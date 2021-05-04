# Lint as: python3
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
"""TFX penguin template pipeline definition.

This file defines TFX pipeline and various components in the pipeline.
"""

from typing import List, Optional, Text

import tensorflow_model_analysis as tfma

from tfx.components import CsvExampleGen
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import Pusher
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.dsl.components.common import resolver
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.experimental.templates.penguin.models import features
from tfx.orchestration import pipeline
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing
from ml_metadata.proto import metadata_store_pb2


def create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    data_path: Text,
    preprocessing_fn: Text,
    run_fn: Text,
    train_args: trainer_pb2.TrainArgs,
    eval_args: trainer_pb2.EvalArgs,
    eval_accuracy_threshold: float,
    serving_model_dir: Text,
    metadata_connection_config: Optional[
        metadata_store_pb2.ConnectionConfig] = None,
    beam_pipeline_args: Optional[List[Text]] = None,
) -> pipeline.Pipeline:
  """Implements the penguin pipeline with TFX."""

  components = []

  # Brings data into the pipeline or otherwise joins/converts training data.
  # TODO(step 2): Might use another ExampleGen class for your data.
  example_gen = CsvExampleGen(input_base=data_path)
  components.append(example_gen)

  # Computes statistics over data for visualization and example validation.
  statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
  components.append(statistics_gen)

  # Generates schema based on statistics files.
  schema_gen = SchemaGen(
      statistics=statistics_gen.outputs['statistics'], infer_feature_shape=True)
  components.append(schema_gen)

  # Performs anomaly detection based on statistics and data schema.
  example_validator = ExampleValidator(  # pylint: disable=unused-variable
      statistics=statistics_gen.outputs['statistics'],
      schema=schema_gen.outputs['schema'])
  components.append(example_validator)

  # Performs transformations and feature engineering in training and serving.
  transform = Transform(  # pylint: disable=unused-variable
      examples=example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'],
      preprocessing_fn=preprocessing_fn)
  # TODO(step 3): Uncomment here to add Transform to the pipeline.
  # components.append(transform)

  # Uses user-provided Python function that implements a model using Tensorflow.
  trainer = Trainer(
      run_fn=run_fn,
      examples=example_gen.outputs['examples'],
      # Use outputs of Transform as training inputs if Transform is used.
      # examples=transform.outputs['transformed_examples'],
      # transform_graph=transform.outputs['transform_graph'],
      schema=schema_gen.outputs['schema'],
      train_args=train_args,
      eval_args=eval_args)
  # TODO(step 4): Uncomment here to add Trainer to the pipeline.
  # components.append(trainer)

  # Get the latest blessed model for model validation.
  model_resolver = resolver.Resolver(
      strategy_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
      model=Channel(type=Model),
      model_blessing=Channel(
          type=ModelBlessing)).with_id('latest_blessed_model_resolver')
  # TODO(step 5): Uncomment here to add Resolver to the pipeline.
  # components.append(model_resolver)

  # Uses TFMA to compute a evaluation statistics over features of a model and
  # perform quality validation of a candidate model (compared to a baseline).
  eval_config = tfma.EvalConfig(
      model_specs=[tfma.ModelSpec(label_key=features.LABEL_KEY)],
      slicing_specs=[tfma.SlicingSpec()],
      metrics_specs=[
          tfma.MetricsSpec(metrics=[
              tfma.MetricConfig(
                  class_name='SparseCategoricalAccuracy',
                  threshold=tfma.MetricThreshold(
                      value_threshold=tfma.GenericValueThreshold(
                          lower_bound={'value': eval_accuracy_threshold}),
                      change_threshold=tfma.GenericChangeThreshold(
                          direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                          absolute={'value': -1e-10})))
          ])
      ])
  evaluator = Evaluator(  # pylint: disable=unused-variable
      examples=example_gen.outputs['examples'],
      model=trainer.outputs['model'],
      baseline_model=model_resolver.outputs['model'],
      # Change threshold will be ignored if there is no baseline (first run).
      eval_config=eval_config)
  # TODO(step 5): Uncomment here to add Evaluator to the pipeline.
  # components.append(evaluator)

  # Pushes the model to a file destination if check passed.
  pusher = Pusher(  # pylint: disable=unused-variable
      model=trainer.outputs['model'],
      model_blessing=evaluator.outputs['blessing'],
      push_destination=pusher_pb2.PushDestination(
          filesystem=pusher_pb2.PushDestination.Filesystem(
              base_directory=serving_model_dir)))
  # TODO(step 5): Uncomment here to add Pusher to the pipeline.
  # components.append(pusher)

  return pipeline.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=components,
      # Change this value to control caching of execution results. Default value
      # is `False`.
      # enable_cache=True,
      metadata_connection_config=metadata_connection_config,
      beam_pipeline_args=beam_pipeline_args,
  )
