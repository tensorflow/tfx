"""Chicago taxi example using TFX DSL."""
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import datetime
import json
import os
from tensorflow.python.lib.io import file_io
from tfx.components import BigQueryExampleGen
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import ModelValidator
from tfx.components import Pusher
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.runtimes.airflow.airflow_runner import AirflowDAGRunner as TfxRunner
from tfx.runtimes.pipeline import PipelineDecorator

# Directory and data locations
home_dir = os.path.join(os.environ['HOME'], 'airflow/')
# Cloud storage
input_bucket = 'gs://my-bucket'
output_bucket = 'gs://my-bucket'
# Helper functions for the taxi pipleine: estimator and preprocessing_fn
taxi_utils = os.path.join(input_bucket, 'taxi_utils.py')
# Path which can be listened by model server. Pusher will output model here.
serving_model_dir = os.path.join(output_bucket, 'serving_model/taxi_bigquery')

# Airflow-specific configs; these will be passed directly to airflow
airflow_config = {
    'schedule_interval': None,
    'start_date': datetime.datetime(2019, 1, 1),
}


# TODO(jyzhao): script documentation.
# Set GOOGLE_APPLICATION_CREDENTIALS in the console that you run
# "airflow webserver" and "airflow scheduler", For more information,
# please see https://cloud.google.com/docs/authentication/getting-started.
def get_project_id():
  if 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ:
    return '''Environment variable GOOGLE_APPLICATION_CREDENTIALS is missing.
    See https://cloud.google.com/docs/authentication/getting-started for more
    information how to set up GCP for your environment.'''
  else:
    path = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
    contents = json.loads(file_io.read_file_to_string(path))
    return contents['project_id']


cmle_training_args = {
    'pythonModule': None,  # Will be populated by TFX
    'args': None,  # Will be populated by TFX
    'region': 'us-central1',
    'jobDir': os.path.join(output_bucket, 'tmp'),
    'runtimeVersion': '1.11',
    'pythonVersion': '2.7',
    'project': get_project_id()
}


# TODO(b/124066911): Centralize tfx related config into one place.
@PipelineDecorator(
    pipeline_name='chicago_taxi_gcp',
    log_root='/var/tmp/tfx/logs',
    metadata_db_root=os.path.join(home_dir, 'data/tfx/metadata'),
    pipeline_root=os.path.join(output_bucket, 'tfx-pipelines'),
    additional_pipeline_args={
        'beam_pipeline_args': [
            '--runner=DataflowRunner', '--experiment=shuffle_mode=auto',
            '--project=' + get_project_id(),
            '--temp_location=' + os.path.join(output_bucket, 'tmp'),
            '--staging_location=' + os.path.join(output_bucket, 'tmp')
        ],
    })
def create_pipeline():
  """Implements the chicago taxi pipeline with TFX."""
  query = """
          SELECT
            pickup_community_area,
            fare,
            EXTRACT(MONTH FROM trip_start_timestamp) AS trip_start_month,
            EXTRACT(HOUR FROM trip_start_timestamp) AS trip_start_hour,
            EXTRACT(DAYOFWEEK FROM trip_start_timestamp) AS trip_start_day,
            UNIX_SECONDS(trip_start_timestamp) AS trip_start_timestamp,
            pickup_latitude,
            pickup_longitude,
            dropoff_latitude,
            dropoff_longitude,
            trip_miles,
            pickup_census_tract,
            dropoff_census_tract,
            payment_type,
            company,
            trip_seconds,
            dropoff_community_area,
            tips
          FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
          WHERE MOD(FARM_FINGERPRINT(unique_key), 3) = 0 LIMIT 15000"""

  # Brings data into the pipeline or otherwise joins/converts training data.
  example_gen = BigQueryExampleGen(query=query)

  # Computes statistics over data for visualization and example validation.
  statistics_gen = StatisticsGen(input_data=example_gen.outputs.examples)

  # Generates schema based on statistics files.
  infer_schema = SchemaGen(stats=statistics_gen.outputs.output)

  # Performs anomaly detection based on statistics and data schema.
  validate_stats = ExampleValidator(  # pylint: disable=unused-variable
      stats=statistics_gen.outputs.output,
      schema=infer_schema.outputs.output)

  # Performs transformations and feature engineering in training and serving.
  transform = Transform(
      input_data=example_gen.outputs.examples,
      schema=infer_schema.outputs.output,
      module_file=taxi_utils)

  # Uses user-provided Python function that implements a model using TF-Learn.
  trainer = Trainer(
      module_file=taxi_utils,
      transformed_examples=transform.outputs.transformed_examples,
      schema=infer_schema.outputs.output,
      transform_output=transform.outputs.transform_output,
      train_steps=10000,
      eval_steps=5000,
      custom_config={'cmle_training_args': cmle_training_args},
      warm_starting=True)

  # Uses TFMA to compute a evaluation statistics over features of a model.
  model_analyzer = Evaluator(  # pylint: disable=unused-variable
      examples=example_gen.outputs.examples,
      model_exports=trainer.outputs.output)

  # Performs quality validation of a candidate model (compared to a baseline).
  model_validator = ModelValidator(
      examples=example_gen.outputs.examples, model=trainer.outputs.output)

  # Checks whether the model passed the validation steps and pushes the model
  # to a file destination if check passed.
  pusher = Pusher(  # pylint: disable=unused-variable
      model_export=trainer.outputs.output,
      model_blessing=model_validator.outputs.blessing,
      serving_model_dir=serving_model_dir)

  return [
      example_gen, statistics_gen, infer_schema, validate_stats, transform,
      trainer, model_analyzer, model_validator, pusher
  ]


pipeline = TfxRunner(airflow_config).run(create_pipeline())
