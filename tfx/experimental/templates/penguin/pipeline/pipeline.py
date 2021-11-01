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

from typing import List, Optional

import tensorflow_model_analysis as tfma
from tfx import v1 as tfx
from tfx.experimental.templates.penguin.models import features

from ml_metadata.proto import metadata_store_pb2


def create_pipeline(
    pipeline_name: str,
    pipeline_root: str,
    data_path: str,
    preprocessing_fn: str,
    run_fn: str,
    train_args: tfx.proto.TrainArgs,
    eval_args: tfx.proto.EvalArgs,
    eval_accuracy_threshold: float,
    serving_model_dir: str,
    schema_path: Optional[str] = None,
    metadata_connection_config: Optional[
        metadata_store_pb2.ConnectionConfig] = None,
    beam_pipeline_args: Optional[List[str]] = None,
) -> tfx.dsl.Pipeline:
  """Implements the penguin pipeline with TFX."""

  components = []

  # Brings data into the pipeline or otherwise joins/converts training data.
  # TODO(step 2): Might use another ExampleGen class for your data.
  example_gen = tfx.components.CsvExampleGen(input_base=data_path)
  components.append(example_gen)

  # Computes statistics over data for visualization and example validation.
  statistics_gen = tfx.components.StatisticsGen(
      examples=example_gen.outputs['examples'])
  components.append(statistics_gen)

  if schema_path is None:
    # Generates schema based on statistics files.
    schema_gen = tfx.components.SchemaGen(
        statistics=statistics_gen.outputs['statistics'])
    components.append(schema_gen)
  else:
    # Import user provided schema into the pipeline.
    schema_gen = tfx.components.ImportSchemaGen(schema_file=schema_path)
    components.append(schema_gen)

    # Performs anomaly detection based on statistics and data schema.
    example_validator = tfx.components.ExampleValidator(  # pylint: disable=unused-variable
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema'])
    components.append(example_validator)

  # Performs transformations and feature engineering in training and serving.
  transform = tfx.components.Transform(  # pylint: disable=unused-variable
      examples=example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'],
      preprocessing_fn=preprocessing_fn)
  # TODO(step 3): Uncomment here to add Transform to the pipeline.
  # components.append(transform)

  # Uses user-provided Python function that implements a model using Tensorflow.
  trainer = tfx.components.Trainer(
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
  model_resolver = tfx.dsl.Resolver(
      strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
      model=tfx.dsl.Channel(type=tfx.types.standard_artifacts.Model),
      model_blessing=tfx.dsl.Channel(
          type=tfx.types.standard_artifacts.ModelBlessing)).with_id(
              'latest_blessed_model_resolver')
  # TODO(step 5): Uncomment here to add Resolver to the pipeline.
  # components.append(model_resolver)

  # Uses TFMA to compute a evaluation statistics over features of a model and
  # perform quality validation of a candidate model (compared to a baseline).
  eval_config = tfma.EvalConfig(
      model_specs=[
          tfma.ModelSpec(
              signature_name='serving_default',
              label_key=features.LABEL_KEY,
              # Use transformed label key if Transform is used.
              # label_key=features.transformed_name(features.LABEL_KEY),
              preprocessing_function_names=['transform_features'])
      ],
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
  evaluator = tfx.components.Evaluator(  # pylint: disable=unused-variable
      examples=example_gen.outputs['examples'],
      model=trainer.outputs['model'],
      baseline_model=model_resolver.outputs['model'],
      # Change threshold will be ignored if there is no baseline (first run).
      eval_config=eval_config)
  # TODO(step 5): Uncomment here to add Evaluator to the pipeline.
  # components.append(evaluator)

  # Pushes the model to a file destination if check passed.
  pusher = tfx.components.Pusher(  # pylint: disable=unused-variable
      model=trainer.outputs['model'],
      model_blessing=evaluator.outputs['blessing'],
      push_destination=tfx.proto.PushDestination(
          filesystem=tfx.proto.PushDestination.Filesystem(
              base_directory=serving_model_dir)))
  # TODO(step 5): Uncomment here to add Pusher to the pipeline.
  # components.append(pusher)

  return tfx.dsl.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=components,
      # Change this value to control caching of execution results. Default value
      # is `False`.
      # enable_cache=True,
      metadata_connection_config=metadata_connection_config,
      beam_pipeline_args=beam_pipeline_args,
  )
