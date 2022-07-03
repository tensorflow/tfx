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
"""Test composable pipeline for tfx.dsl.compiler.compiler."""
import os

import tensorflow_model_analysis as tfma
from tfx.components import CsvExampleGen
from tfx.components import Evaluator
from tfx.components import InfraValidator
from tfx.components import Pusher
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components.trainer.executor import GenericExecutor
from tfx.dsl.components.base import executor_spec
from tfx.dsl.experimental.conditionals import conditional
from tfx.dsl.placeholder import placeholder as ph
from tfx.orchestration import pipeline
from tfx.proto import infra_validator_pb2
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.types import channel


def data_ingestion_pipeline(data_path: str) -> pipeline.Pipeline:
  """A data ingestion pipeline that tests input-less composable pipeline."""
  example_gen = CsvExampleGen(input_base=data_path)
  statistics_gen = StatisticsGen(examples=example_gen.outputs["examples"])
  schema_gen = SchemaGen(statistics=statistics_gen.outputs["statistics"])

  return pipeline.Pipeline(
      pipeline_name="data-ingestion-pipeline",
      components=[example_gen, statistics_gen, schema_gen],
      outputs={
          "examples": example_gen.outputs["examples"],
          "schema": schema_gen.outputs["schema"]
      })


def training_pipeline(examples: channel.BaseChannel,
                      schema: channel.BaseChannel) -> pipeline.Pipeline:
  """A pipeline that tests composable pipeline with both inputs and outputs."""
  pipeline_inputs = pipeline.PipelineInputs({
      "examples": examples,
      "schema": schema
  })
  trainer = Trainer(
      module_file="module_file",
      custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
      examples=pipeline_inputs.inputs["examples"],
      schema=pipeline_inputs.inputs["schema"],
      train_args=trainer_pb2.TrainArgs(num_steps=2000),
      eval_args=trainer_pb2.EvalArgs(num_steps=5))
  return pipeline.Pipeline(
      pipeline_name="training-pipeline",
      components=[trainer],
      inputs=pipeline_inputs,
      outputs={"model": trainer.outputs["model"]})


def infra_validator_pipeline(examples: channel.BaseChannel,
                             model: channel.BaseChannel) -> pipeline.Pipeline:
  """A test pipeline that is embedded inside another composable pipeline."""
  # It is to test the nested composable pipeline case where the inner pipeline
  # directly consumes the pipeline inputs of the outer composable pipeline, in
  # which case, the inner pipeline's pipeline begin node should take outer
  # pipeline's pipeline begin node as upstream.
  pipeline_inputs = pipeline.PipelineInputs({
      "examples": examples,
      "model": model
  })
  infra_validator = InfraValidator(
      model=pipeline_inputs.inputs["model"],
      examples=pipeline_inputs.inputs["examples"],
      serving_spec=infra_validator_pb2.ServingSpec(
          tensorflow_serving=infra_validator_pb2.TensorFlowServing(
              tags=["latest"]),
          local_docker=infra_validator_pb2.LocalDockerConfig()),
      request_spec=infra_validator_pb2.RequestSpec(
          tensorflow_serving=infra_validator_pb2.TensorFlowServingRequestSpec())
  )
  return pipeline.Pipeline(
      pipeline_name="infra-validator-pipeline",
      components=[infra_validator],
      inputs=pipeline_inputs,
      outputs={"blessing": infra_validator.outputs["blessing"]})


def validate_and_push_pipeline(examples: channel.BaseChannel,
                               model: channel.BaseChannel,
                               serving_model_dir: str) -> pipeline.Pipeline:
  """An input-only test composable pipeline with conditional."""
  pipeline_inputs = pipeline.PipelineInputs({
      "examples": examples,
      "model": model
  })
  infra_validator = infra_validator_pipeline(pipeline_inputs.inputs["examples"],
                                             pipeline_inputs.inputs["model"])
  with conditional.Cond(
      ph.logical_and(infra_validator.outputs["blessing"].future()[0].value == 1,
                     pipeline_inputs.inputs["model"].future()[0].uri != "")):  # pylint: disable=g-explicit-bool-comparison
    pusher = Pusher(
        model=pipeline_inputs.inputs["model"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir)))

  return pipeline.Pipeline(
      pipeline_name="validate-and-push-pipeline",
      components=[infra_validator, pusher],
      inputs=pipeline_inputs)


def create_test_pipeline():
  """Builds a parent pipeline that contains multiple inner pipelines."""
  pipeline_name = "composable-pipeline"
  pipeline_root = "pipeline_root"
  serving_model_dir = os.path.join(pipeline_root, "serving_model",
                                   pipeline_name)
  tfx_root = "tfx_root"
  data_path = os.path.join(tfx_root, "data_path")
  pipeline_root = os.path.join(tfx_root, "pipelines", pipeline_name)

  data_ingestion = data_ingestion_pipeline(data_path)

  training = training_pipeline(data_ingestion.outputs["examples"],
                               data_ingestion.outputs["schema"])

  eval_config = tfma.EvalConfig(
      model_specs=[tfma.ModelSpec(signature_name="eval")],
      slicing_specs=[tfma.SlicingSpec()],
      metrics_specs=[
          tfma.MetricsSpec(
              thresholds={
                  "sparse_categorical_accuracy":
                      tfma.MetricThreshold(
                          value_threshold=tfma.GenericValueThreshold(
                              lower_bound={"value": 0.6}),
                          change_threshold=tfma.GenericChangeThreshold(
                              direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                              absolute={"value": -1e-10}))
              })
      ])

  evaluator = Evaluator(
      examples=data_ingestion.outputs["examples"],
      model=training.outputs["model"],
      eval_config=eval_config)

  with conditional.Cond(evaluator.outputs["blessing"].future()[0].value == 1):
    validate_and_push = validate_and_push_pipeline(
        data_ingestion.outputs["examples"], training.outputs["model"],
        serving_model_dir)

  return pipeline.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=[data_ingestion, training, evaluator, validate_and_push],
      enable_cache=True,
      beam_pipeline_args=["--my_testing_beam_pipeline_args=foo"],
      execution_mode=pipeline.ExecutionMode.SYNC)
