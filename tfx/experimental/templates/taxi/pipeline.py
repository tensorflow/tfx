# Lint as: python2, python3
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional, Text, List, Dict, Any

from ml_metadata.proto import metadata_store_pb2
from tfx.components import CsvExampleGen
# TODO(step 7): (Optional) Uncomment here to use BigQuery as a data source.
# from tfx.components import BigQueryExampleGen
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import ModelValidator
from tfx.components import Pusher
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.components.base import executor_spec
from tfx.extensions.google_cloud_ai_platform.pusher import executor as ai_platform_pusher_executor
from tfx.extensions.google_cloud_ai_platform.trainer import executor as ai_platform_trainer_executor
from tfx.orchestration import pipeline
from tfx.proto import evaluator_pb2
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.utils.dsl_utils import external_input


def create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    data_path: Text,
    # TODO(step 7): (Optional) Uncomment here to use BigQuery as a data source.
    # query: Text,
    preprocessing_fn: Text,
    trainer_fn: Text,
    train_args: trainer_pb2.TrainArgs,
    eval_args: trainer_pb2.EvalArgs,
    serving_model_dir: Text,
    metadata_connection_config: Optional[
        metadata_store_pb2.ConnectionConfig] = None,
    beam_pipeline_args: Optional[List[Text]] = None,
    ai_platform_training_args: Optional[Dict[Text, Text]] = None,
    ai_platform_serving_args: Optional[Dict[Text, Any]] = None,
) -> pipeline.Pipeline:
  """Implements the chicago taxi pipeline with TFX."""

  components = []

  # Brings data into the pipeline or otherwise joins/converts training data.
  example_gen = CsvExampleGen(input=external_input(data_path))
  # TODO(step 7): (Optional) Uncomment here to use BigQuery as a data source.
  # example_gen = BigQueryExampleGen(query=query)
  components.append(example_gen)

  # Computes statistics over data for visualization and example validation.
  statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
  # TODO(step 5): Uncomment here to add StatisticsGen to the pipeline.
  # components.append(statistics_gen)

  # Generates schema based on statistics files.
  infer_schema = SchemaGen(
      statistics=statistics_gen.outputs['statistics'],
      infer_feature_shape=False)
  # TODO(step 5): Uncomment here to add SchemaGen to the pipeline.
  # components.append(infer_schema)

  # Performs anomaly detection based on statistics and data schema.
  validate_stats = ExampleValidator(  # pylint: disable=unused-variable
      statistics=statistics_gen.outputs['statistics'],
      schema=infer_schema.outputs['schema'])
  # TODO(step 5): Uncomment here to add ExampleValidator to the pipeline.
  # components.append(validate_stats)

  # Performs transformations and feature engineering in training and serving.
  transform = Transform(
      examples=example_gen.outputs['examples'],
      schema=infer_schema.outputs['schema'],
      preprocessing_fn=preprocessing_fn)
  # TODO(step 6): Uncomment here to add Transform to the pipeline.
  # components.append(transform)

  # Uses user-provided Python function that implements a model using TF-Learn.
  trainer_args = {
      'trainer_fn': trainer_fn,
      'transformed_examples': transform.outputs['transformed_examples'],
      'schema': infer_schema.outputs['schema'],
      'transform_graph': transform.outputs['transform_graph'],
      'train_args': train_args,
      'eval_args': eval_args,
  }
  if ai_platform_training_args is not None:
    trainer_args.update({
        'custom_executor_spec':
            executor_spec.ExecutorClassSpec(
                ai_platform_trainer_executor.Executor),
        'custom_config': {
            ai_platform_trainer_executor.TRAINING_ARGS_KEY:
                ai_platform_training_args
        }
    })
  trainer = Trainer(**trainer_args)
  # TODO(step 6): Uncomment here to add Trainer to the pipeline.
  # components.append(trainer)

  # Uses TFMA to compute a evaluation statistics over features of a model.
  model_analyzer = Evaluator(  # pylint: disable=unused-variable
      examples=example_gen.outputs['examples'],
      model=trainer.outputs['model'],
      feature_slicing_spec=evaluator_pb2.FeatureSlicingSpec(specs=[
          evaluator_pb2.SingleSlicingSpec(
              column_for_slicing=['trip_start_hour'])
      ]))
  # TODO(step 6): Uncomment here to add Evaluator to the pipeline.
  # components.append(model_analyzer)

  # Performs quality validation of a candidate model (compared to a baseline).
  model_validator = ModelValidator(
      examples=example_gen.outputs['examples'], model=trainer.outputs['model'])
  # TODO(step 6): Uncomment here to add ModelValidator to the pipeline.
  # components.append(model_validator)

  # Checks whether the model passed the validation steps and pushes the model
  # to a file destination if check passed.
  pusher_args = {
      'model':
          trainer.outputs['model'],
      'model_blessing':
          model_validator.outputs['blessing'],
      'push_destination':
          pusher_pb2.PushDestination(
              filesystem=pusher_pb2.PushDestination.Filesystem(
                  base_directory=serving_model_dir)),
  }
  if ai_platform_serving_args is not None:
    pusher_args.update({
        'custom_executor_spec':
            executor_spec.ExecutorClassSpec(ai_platform_pusher_executor.Executor
                                           ),
        'custom_config': {
            ai_platform_pusher_executor.SERVING_ARGS_KEY:
                ai_platform_serving_args
        },
    })
  pusher = Pusher(**pusher_args)  # pylint: disable=unused-variable
  # TODO(step 6): Uncomment here to add Pusher to the pipeline.
  # components.append(pusher)

  return pipeline.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=components,
      enable_cache=True,
      metadata_connection_config=metadata_connection_config,
      beam_pipeline_args=beam_pipeline_args,
  )
