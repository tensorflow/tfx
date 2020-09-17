# Lint as: python3
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

This example runs in Kubeflow with Google Cloud services..
"""

import datetime
import os

from tfx.components import CsvExampleGen
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import ModelValidator
from tfx.components import Pusher
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.examples.custom_components.slack.slack_component.component import SlackComponent
from tfx.orchestration import pipeline
from tfx.orchestration.kubeflow import kubeflow_dag_runner
from tfx.proto import evaluator_pb2
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.utils.dsl_utils import csv_input

# This example assumes that the taxi data is stored in _input_bucket/data/simple
# and the taxi utility function is in example/taxi_utils_slack.py.
# Feel free to customize this as needed.
_input_bucket = 'gs://my-bucket'
_output_bucket = 'gs://my-bucket'
_taxi_root = __file__
_data_root = os.path.join(_input_bucket, 'data', 'simple')


# Python module file to inject customized logic into the TFX components. The
# Transform and Trainer both require user-defined functions to run successfully.
_taxi_trainer_func = 'example.taxi_utils_slack.trainer_fn'
_taxi_transformer_func = 'example.taxi_utils_slack.preprocessing_fn'
# Path which can be listened to by the model server.  Pusher will output the
# trained model here.
_serving_model_dir = os.path.join(_taxi_root, 'serving_model/taxi_slack')
# Slack channel to push the model notifications to.
_slack_channel_id = os.environ['TFX_SLACK_CHANNEL_ID']
# Slack token to set up connection.
_slack_token = os.environ['TFX_SLACK_BOT_TOKEN']

# Directory and data locations.  This example assumes all of the chicago taxi
# example code and metadata library is relative to $HOME, but you can store
# these files anywhere on your local filesystem.
_tfx_root = os.path.join(os.environ['HOME'], '/tfx')
_pipeline_name = 'chicago_taxi_slack_kubeflow'
_pipeline_root = os.path.join(_input_bucket, _pipeline_name)
_log_root = os.path.join(_tfx_root, 'logs')

# Airflow-specific configs; these will be passed directly to airflow
_airflow_config = {
    'schedule_interval': None,
    'start_date': datetime.datetime(2019, 1, 1),
}


def _create_pipeline():
  """Implements the chicago taxi pipeline with TFX."""
  examples = csv_input(_data_root)

  # Brings data into the pipeline or otherwise joins/converts training data.
  example_gen = CsvExampleGen(input=examples)

  # Computes statistics over data for visualization and example validation.
  statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

  # Generates schema based on statistics files.
  schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'])

  # Performs anomaly detection based on statistics and data schema.
  example_validator = ExampleValidator(
      statistics=statistics_gen.outputs['statistics'],
      schema=schema_gen.outputs['schema'])

  # Performs transformations and feature engineering in training and serving.
  transform = Transform(
      examples=example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'],
      preprocessing_fn=_taxi_transformer_func)

  # Uses user-provided Python function that implements a model using TF-Learn.
  trainer = Trainer(
      trainer_fn=_taxi_trainer_func,
      examples=transform.outputs['transformed_examples'],
      schema=schema_gen.outputs['schema'],
      transform_graph=transform.outputs['transform_graph'],
      train_args=trainer_pb2.TrainArgs(num_steps=10000),
      eval_args=trainer_pb2.EvalArgs(num_steps=5000))

  # Uses TFMA to compute a evaluation statistics over features of a model.
  evaluator = Evaluator(
      examples=example_gen.outputs['examples'],
      model=trainer.outputs['model'],
      feature_slicing_spec=evaluator_pb2.FeatureSlicingSpec(specs=[
          evaluator_pb2.SingleSlicingSpec(
              column_for_slicing=['trip_start_hour'])
      ]))

  # Performs quality validation of a candidate model (compared to a baseline).
  model_validator = ModelValidator(
      examples=example_gen.outputs['examples'], model=trainer.outputs['model'])

  # This custom component serves as a bridge between pipeline and human model
  # reviewers to enable review-and-push workflow in model development cycle. It
  # utilizes Slack API to send message to user-defined Slack channel with model
  # URI info and wait for go / no-go decision from the same Slack channel:
  #   * To approve the model, users need to reply the thread sent out by the bot
  #     started by SlackComponent with 'lgtm' or 'approve'.
  #   * To reject the model, users need to reply the thread sent out by the bot
  #     started by SlackComponent with 'decline' or 'reject'.
  slack_validator = SlackComponent(
      model=trainer.outputs['model'],
      model_blessing=model_validator.outputs['blessing'],
      slack_token=_slack_token,
      slack_channel_id=_slack_channel_id,
      timeout_sec=3600,
  )
  # Checks whether the model passed the validation steps and pushes the model
  # to a file destination if check passed.
  pusher = Pusher(
      model=trainer.outputs['model'],
      model_blessing=slack_validator.outputs['slack_blessing'],
      push_destination=pusher_pb2.PushDestination(
          filesystem=pusher_pb2.PushDestination.Filesystem(
              base_directory=_serving_model_dir)))

  return pipeline.Pipeline(
      pipeline_name=_pipeline_name,
      pipeline_root=_pipeline_root,
      components=[
          example_gen, statistics_gen, schema_gen, example_validator, transform,
          trainer, evaluator, model_validator, slack_validator, pusher
      ],
      enable_cache=True,
  )


if __name__ == '__main__':
  # Metadata config. The defaults works work with the installation of
  # KF Pipelines using Kubeflow. If installing KF Pipelines using the
  # lightweight deployment option, you may need to override the defaults.
  metadata_config = kubeflow_dag_runner.get_default_kubeflow_metadata_config()

  # This pipeline automatically injects the Kubeflow TFX image if the
  # environment variable 'KUBEFLOW_TFX_IMAGE' is defined. Currently, the tfx
  # cli tool exports the environment variable to pass to the pipelines.
  tfx_image = os.environ.get('KUBEFLOW_TFX_IMAGE', None)

  runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
      kubeflow_metadata_config=metadata_config,
      # Specify custom docker image to use.
      tfx_image=tfx_image
  )

  kubeflow_dag_runner.KubeflowDagRunner(config=runner_config).run(
      _create_pipeline())
