# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test pipeline for tfx.dsl.compiler.compiler."""
import os

import tensorflow_model_analysis as tfma
from tfx.components import CsvExampleGen
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import ImporterNode
from tfx.components import Pusher
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components.trainer.executor import GenericExecutor
from tfx.dsl.components.base import executor_spec
from tfx.dsl.components.common import resolver
from tfx.dsl.input_resolution.strategies import latest_blessed_model_strategy
from tfx.orchestration import data_types
from tfx.orchestration import pipeline
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.types import Channel
from tfx.types import standard_artifacts


def create_test_pipeline():
  """Builds an Iris example pipeline with slight changes."""
  pipeline_name = "iris"
  iris_root = "iris_root"
  serving_model_dir = os.path.join(iris_root, "serving_model", pipeline_name)
  tfx_root = "tfx_root"
  data_path = os.path.join(tfx_root, "data_path")
  pipeline_root = os.path.join(tfx_root, "pipelines", pipeline_name)

  example_gen = CsvExampleGen(input_base=data_path)

  statistics_gen = StatisticsGen(examples=example_gen.outputs["examples"])

  importer = ImporterNode(
      source_uri="m/y/u/r/i",
      properties={
          "split_names": "['train', 'eval']",
      },
      custom_properties={
          "int_custom_property": 42,
          "str_custom_property": "42",
      },
      artifact_type=standard_artifacts.Examples).with_id("my_importer")

  schema_gen = SchemaGen(
      statistics=statistics_gen.outputs["statistics"], infer_feature_shape=True)

  example_validator = ExampleValidator(
      statistics=statistics_gen.outputs["statistics"],
      schema=schema_gen.outputs["schema"])

  trainer = Trainer(
      # Use RuntimeParameter as module_file to test out RuntimeParameter in
      # compiler.
      module_file=data_types.RuntimeParameter(
          name="module_file",
          default=os.path.join(iris_root, "iris_utils.py"),
          ptype=str),
      custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
      examples=example_gen.outputs["examples"],
      schema=schema_gen.outputs["schema"],
      train_args=trainer_pb2.TrainArgs(num_steps=2000),
      # Attaching `TrainerArgs` as platform config is not sensible practice,
      # but is only for testing purpose.
      eval_args=trainer_pb2.EvalArgs(num_steps=5)).with_platform_config(
          config=trainer_pb2.TrainArgs(num_steps=2000))

  model_resolver = resolver.Resolver(
      strategy_class=latest_blessed_model_strategy.LatestBlessedModelStrategy,
      baseline_model=Channel(
          type=standard_artifacts.Model, producer_component_id="Trainer"),
      # Cannot add producer_component_id="Evaluator" for model_blessing as it
      # raises "producer component should have already been compiled" error.
      model_blessing=Channel(type=standard_artifacts.ModelBlessing)).with_id(
          "latest_blessed_model_resolver")

  eval_config = tfma.EvalConfig(
      model_specs=[tfma.ModelSpec(signature_name="eval")],
      slicing_specs=[tfma.SlicingSpec()],
      metrics_specs=[
          tfma.MetricsSpec(
              thresholds={
                  "sparse_categorical_accuracy":
                      tfma.config.MetricThreshold(
                          value_threshold=tfma.GenericValueThreshold(
                              lower_bound={"value": 0.6}),
                          change_threshold=tfma.GenericChangeThreshold(
                              direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                              absolute={"value": -1e-10}))
              })
      ])
  evaluator = Evaluator(
      examples=example_gen.outputs["examples"],
      model=trainer.outputs["model"],
      baseline_model=model_resolver.outputs["baseline_model"],
      eval_config=eval_config)

  pusher = Pusher(
      model=trainer.outputs["model"],
      model_blessing=evaluator.outputs["blessing"],
      push_destination=pusher_pb2.PushDestination(
          filesystem=pusher_pb2.PushDestination.Filesystem(
              base_directory=serving_model_dir)))

  return pipeline.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=[
          example_gen,
          statistics_gen,
          importer,
          schema_gen,
          example_validator,
          trainer,
          model_resolver,
          evaluator,
          pusher,
      ],
      enable_cache=False,
      beam_pipeline_args=["--my_testing_beam_pipeline_args=bar"],
      # Attaching `TrainerArgs` as platform config is not sensible practice,
      # but is only for testing purpose.
      platform_config=trainer_pb2.TrainArgs(num_steps=2000),
      execution_mode=pipeline.ExecutionMode.ASYNC)
