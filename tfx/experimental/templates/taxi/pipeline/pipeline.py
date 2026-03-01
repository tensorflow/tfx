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
"""TFX taxi template pipeline definition.

This file defines TFX pipeline and various components in the pipeline.
"""

from typing import Any, Dict, List, Optional

import tensorflow_model_analysis as tfma
from tfx import v1 as tfx

from ml_metadata.proto import metadata_store_pb2


def create_pipeline(
    pipeline_name: str,
    pipeline_root: str,
    data_path: str,
    # TODO(step 7): (Optional) Uncomment here to use BigQuery as a data source.
    # query: str,
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
    ai_platform_training_args: Optional[Dict[str, str]] = None,
    ai_platform_serving_args: Optional[Dict[str, Any]] = None,
) -> tfx.dsl.Pipeline:
  """Implements the chicago taxi pipeline with TFX."""

  components = []

  # Brings data into the pipeline or otherwise joins/converts training data.
  example_gen = tfx.components.CsvExampleGen(input_base=data_path)
  # TODO(step 7): (Optional) Uncomment here to use BigQuery as a data source.
  # example_gen = tfx.extensions.google_cloud_big_query.BigQueryExampleGen(
  #     query=query)
  components.append(example_gen)

  # Computes statistics over data for visualization and example validation.
  statistics_gen = tfx.components.StatisticsGen(
      examples=example_gen.outputs['examples'])
  # TODO(step 5): Uncomment here to add StatisticsGen to the pipeline.
  # components.append(statistics_gen)

  if schema_path is None:
    # Generates schema based on statistics files.
    schema_gen = tfx.components.SchemaGen(
        statistics=statistics_gen.outputs['statistics'])
    # TODO(step 5): Uncomment here to add SchemaGen to the pipeline.
    # components.append(schema_gen)
  else:
    # Import user provided schema into the pipeline.
    schema_gen = tfx.components.ImportSchemaGen(schema_file=schema_path)
    # TODO(step 5): (Optional) Uncomment here to add ImportSchemaGen to the
    #               pipeline.
    # components.append(schema_gen)

    # Performs anomaly detection based on statistics and data schema.
    example_validator = tfx.components.ExampleValidator(  # pylint: disable=unused-variable
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema'])
    # TODO(step 5): (Optional) Uncomment here to add ExampleValidator to the
    #               pipeline.
    # components.append(example_validator)

  # Performs transformations and feature engineering in training and serving.
  transform = tfx.components.Transform(
      examples=example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'],
      preprocessing_fn=preprocessing_fn)
  # TODO(step 6): Uncomment here to add Transform to the pipeline.
  # components.append(transform)

  # Uses user-provided Python function that implements a model.
  trainer_args = {
      'run_fn': run_fn,
      'examples': transform.outputs['transformed_examples'],
      'schema': schema_gen.outputs['schema'],
      'transform_graph': transform.outputs['transform_graph'],
      'train_args': train_args,
      'eval_args': eval_args,
  }
  if ai_platform_training_args is not None:
    trainer_args['custom_config'] = {
        tfx.extensions.google_cloud_ai_platform.TRAINING_ARGS_KEY:
            ai_platform_training_args,
    }
    trainer = tfx.extensions.google_cloud_ai_platform.Trainer(**trainer_args)
  else:
    trainer = tfx.components.Trainer(**trainer_args)
  # TODO(step 6): Uncomment here to add Trainer to the pipeline.
  # components.append(trainer)

  # Get the latest blessed model for model validation.
  model_resolver = tfx.dsl.Resolver(
      strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
      model=tfx.dsl.Channel(type=tfx.types.standard_artifacts.Model),
      model_blessing=tfx.dsl.Channel(
          type=tfx.types.standard_artifacts.ModelBlessing)).with_id(
              'latest_blessed_model_resolver')
  # TODO(step 6): Uncomment here to add Resolver to the pipeline.
  # components.append(model_resolver)

  # Uses TFMA to compute a evaluation statistics over features of a model and
  # perform quality validation of a candidate model (compared to a baseline).
  eval_config = tfma.EvalConfig(
      model_specs=[
          tfma.ModelSpec(
              signature_name='serving_default',
              label_key='tips_xf',
              preprocessing_function_names=['transform_features'])
      ],
      slicing_specs=[tfma.SlicingSpec()],
      metrics_specs=[
          tfma.MetricsSpec(metrics=[
              tfma.MetricConfig(
                  class_name='BinaryAccuracy',
                  threshold=tfma.MetricThreshold(
                      value_threshold=tfma.GenericValueThreshold(
                          lower_bound={'value': eval_accuracy_threshold}),
                      change_threshold=tfma.GenericChangeThreshold(
                          direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                          absolute={'value': -1e-10})))
          ])
      ])
  evaluator = tfx.components.Evaluator(
      examples=example_gen.outputs['examples'],
      model=trainer.outputs['model'],
      baseline_model=model_resolver.outputs['model'],
      # Change threshold will be ignored if there is no baseline (first run).
      eval_config=eval_config)
  # TODO(step 6): Uncomment here to add Evaluator to the pipeline.
  # components.append(evaluator)

  # Checks whether the model passed the validation steps and pushes the model
  # to a file destination if check passed.
  pusher_args = {
      'model':
          trainer.outputs['model'],
      'model_blessing':
          evaluator.outputs['blessing'],
  }
  if ai_platform_serving_args is not None:
    pusher_args['custom_config'] = {
        tfx.extensions.google_cloud_ai_platform.experimental
        .PUSHER_SERVING_ARGS_KEY:
            ai_platform_serving_args
    }
    pusher = tfx.extensions.google_cloud_ai_platform.Pusher(**pusher_args)  # pylint: disable=unused-variable
  else:
    pusher_args['push_destination'] = tfx.proto.PushDestination(
        filesystem=tfx.proto.PushDestination.Filesystem(
            base_directory=serving_model_dir))
    pusher = tfx.components.Pusher(**pusher_args)  # pylint: disable=unused-variable
  # TODO(step 6): Uncomment here to add Pusher to the pipeline.
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
