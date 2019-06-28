# Copyright 2019 Google LLC. All Rights Reserved.
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
"""Example pipeline to demonstrate custom TFX component.

This example consists of standard TFX components as well as a custom TFX
component requesting for manual review through Slack.

This example along with the custom `SlackComponent` will only serve as an
example and will not be supported by TFX team.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import logging
import os

from slack_component.component import SlackComponent

from tfx.components.evaluator.component import Evaluator
from tfx.components.example_gen.csv_example_gen.component import CsvExampleGen
from tfx.components.example_validator.component import ExampleValidator
from tfx.components.model_validator.component import ModelValidator
from tfx.components.pusher.component import Pusher
from tfx.components.schema_gen.component import SchemaGen
from tfx.components.statistics_gen.component import StatisticsGen
from tfx.components.trainer.component import Trainer
from tfx.components.transform.component import Transform
from tfx.orchestration.airflow.airflow_runner import AirflowDAGRunner
from tfx.orchestration.pipeline import PipelineDecorator
from tfx.proto import evaluator_pb2
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.utils.dsl_utils import csv_input


# This example assumes that the taxi data is stored in ~/taxi/data and the
# taxi utility function is in ~/taxi.  Feel free to customize this as needed.
_taxi_root = os.path.join(os.environ['HOME'], 'taxi')
_data_root = os.path.join(_taxi_root, 'data/simple')
# Python module file to inject customized logic into the TFX components. The
# Transform and Trainer both require user-defined functions to run successfully.
_taxi_module_file = os.path.join(_taxi_root, 'taxi_utils_slack.py')
# Path which can be listened to by the model server.  Pusher will output the
# trained model here.
_serving_model_dir = os.path.join(_taxi_root, 'serving_model/taxi_slack')
# Slack channel to push the model notifications to.
_channel_id = 'my-channel-id'
# Slack token to set up connection.
_slack_token = os.environ['SLACK_BOT_TOKEN']

# Directory and data locations.  This example assumes all of the chicago taxi
# example code and metadata library is relative to $HOME, but you can store
# these files anywhere on your local filesystem.
_tfx_root = os.path.join(os.environ['HOME'], 'tfx')
_pipeline_root = os.path.join(_tfx_root, 'pipelines')
_metadata_db_root = os.path.join(_tfx_root, 'metadata')
_log_root = os.path.join(_tfx_root, 'logs')

# Airflow-specific configs; these will be passed directly to airflow
_airflow_config = {
    'schedule_interval': None,
    'start_date': datetime.datetime(2019, 1, 1),
}

# Logging overrides
logger_overrides = {'log_root': _log_root, 'log_level': logging.INFO}


# TODO(b/124066911): Centralize tfx related config into one place.
@PipelineDecorator(
    pipeline_name='chicago_taxi_slack',
    enable_cache=True,
    metadata_db_root=_metadata_db_root,
    additional_pipeline_args={'logger_args': logger_overrides},
    pipeline_root=_pipeline_root)
def _create_pipeline():
  """Implements the chicago taxi pipeline with TFX."""
  examples = csv_input(_data_root)

  # Brings data into the pipeline or otherwise joins/converts training data.
  example_gen = CsvExampleGen(input_base=examples)

  # Computes statistics over data for visualization and example validation.
  statistics_gen = StatisticsGen(input_data=example_gen.outputs['examples'])

  # Generates schema based on statistics files.
  infer_schema = SchemaGen(stats=statistics_gen.outputs['output'])

  # Performs anomaly detection based on statistics and data schema.
  validate_stats = ExampleValidator(
      stats=statistics_gen.outputs['output'],
      schema=infer_schema.outputs['output'])

  # Performs transformations and feature engineering in training and serving.
  transform = Transform(
      input_data=example_gen.outputs['examples'],
      schema=infer_schema.outputs['output'],
      module_file=_taxi_module_file)

  # Uses user-provided Python function that implements a model using TF-Learn.
  trainer = Trainer(
      module_file=_taxi_module_file,
      transformed_examples=transform.outputs['transformed_examples'],
      schema=infer_schema.outputs['output'],
      transform_output=transform.outputs['transform_output'],
      train_args=trainer_pb2.TrainArgs(num_steps=10000),
      eval_args=trainer_pb2.EvalArgs(num_steps=5000))

  # Uses TFMA to compute a evaluation statistics over features of a model.
  model_analyzer = Evaluator(
      examples=example_gen.outputs['examples'],
      model_exports=trainer.outputs['output'],
      feature_slicing_spec=evaluator_pb2.FeatureSlicingSpec(specs=[
          evaluator_pb2.SingleSlicingSpec(
              column_for_slicing=['trip_start_hour'])
      ]))

  # Performs quality validation of a candidate model (compared to a baseline).
  model_validator = ModelValidator(
      examples=example_gen.outputs['examples'], model=trainer.outputs['output'])

  # This custom component serves as a bridge between pipeline and human model
  # reviewers to enable review-and-push workflow in model development cycle. It
  # utilizes Slack API to send message to user-defined Slack channel with model
  # URI info and wait for go / no-go decision from the same Slack channel:
  #   * To approve the model, users need to reply the thread sent out by the bot
  #     started by SlackComponent with 'lgtm' or 'approve'.
  #   * To reject the model, users need to reply the thread sent out by the bot
  #     started by SlackComponent with 'decline' or 'reject'.
  slack_validator = SlackComponent(
      model_export=trainer.outputs['output'],
      model_blessing=model_validator.outputs['blessing'],
      slack_token=_slack_token,
      channel_id=_channel_id,
      timeout_sec=3600,
  )

  # Checks whether the model passed the validation steps and pushes the model
  # to a file destination if check passed.
  pusher = Pusher(
      model_export=trainer.outputs['output'],
      model_blessing=slack_validator.outputs['slack_blessing'],
      push_destination=pusher_pb2.PushDestination(
          filesystem=pusher_pb2.PushDestination.Filesystem(
              base_directory=_serving_model_dir)))

  return [
      example_gen, statistics_gen, infer_schema, validate_stats, transform,
      trainer, model_analyzer, model_validator, slack_validator, pusher
  ]


pipeline = AirflowDAGRunner(_airflow_config).run(_create_pipeline())
