"""Chicago Taxi example using TFX DSL on Kubeflow."""
# Copyright 2019 Google LLC
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
import os
from tfx.components import BigQueryExampleGen
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import ModelValidator
from tfx.components import Pusher
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.orchestration.kubeflow.runner import KubeflowRunner as TfxRunner
from tfx.orchestration.pipeline import PipelineDecorator

# Cloud storage
input_bucket = 'gs://my-bucket'
output_bucket = 'gs://my-bucket'
# Helper functions for the taxi pipleine: estimator and preprocessing_fn
taxi_utils = os.path.join(output_bucket, 'taxi_utils.py')
# Path which can be listened by model server. Pusher will output model here.
serving_model_dir = os.path.join(output_bucket, 'serving_model/taxi_bigquery')
# Root for all pipeline output.
pipeline_root = os.path.join(output_bucket, 'output')
# GCP project id to use.
project_id = 'my-gcp-project'

cmle_training_args = {
    'pythonModule': None,  # Will be populated by TFX
    'args': None,  # Will be populated by TFX
    'region': 'us-central1',
    'jobDir': os.path.join(output_bucket, 'tmp'),
    'runtimeVersion': '1.11',
    'pythonVersion': '2.7',
    'project': project_id,
}


@PipelineDecorator(
    pipeline_name='chicago_taxi_pipeline_kubeflow_large',
    log_root='/var/tmp/tfx/logs',
    pipeline_root=pipeline_root,
    additional_pipeline_args={
        'beam_pipeline_args': [
            '--runner=DataflowRunner', '--experiment=shuffle_mode=auto',
            '--project=' + project_id,
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
          WHERE MOD(FARM_FINGERPRINT(unique_key), 3) = 0 LIMIT 20000000"""

  # Brings data into the pipeline or otherwise joins/converts training data.
  example_gen = BigQueryExampleGen(query=query)

  # Computes statistics over data for visualization and example validation.
  statistics_gen = StatisticsGen(input_data=example_gen.outputs.examples)

  # Generates schema based on statistics files.
  infer_schema = SchemaGen(stats=statistics_gen.outputs.output)

  # Performs anomaly detection based on statistics and data schema.
  validate_stats = ExampleValidator(
      stats=statistics_gen.outputs.output, schema=infer_schema.outputs.output)

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
  model_analyzer = Evaluator(
      examples=example_gen.outputs.examples,
      model_exports=trainer.outputs.output)

  # Performs quality validation of a candidate model (compared to a baseline).
  model_validator = ModelValidator(
      examples=example_gen.outputs.examples, model=trainer.outputs.output)

  # Checks whether the model passed the validation steps and pushes the model
  # to a file destination if check passed.
  pusher = Pusher(
      model_export=trainer.outputs.output,
      model_blessing=model_validator.outputs.blessing,
      serving_model_dir=serving_model_dir)

  return [
      example_gen, statistics_gen, infer_schema, validate_stats, transform,
      trainer, model_analyzer, model_validator, pusher
  ]


pipeline = TfxRunner().run(create_pipeline())
