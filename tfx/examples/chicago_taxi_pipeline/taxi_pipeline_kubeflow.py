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
"""Chicago Taxi example using TFX DSL on Kubeflow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tfx.components.evaluator.component import Evaluator
from tfx.components.example_gen.big_query_example_gen.component import BigQueryExampleGen
from tfx.components.example_validator.component import ExampleValidator
from tfx.components.model_validator.component import ModelValidator
from tfx.components.pusher.component import Pusher
from tfx.components.schema_gen.component import SchemaGen
from tfx.components.statistics_gen.component import StatisticsGen
from tfx.components.trainer.component import Trainer
from tfx.components.transform.component import Transform
from tfx.orchestration import pipeline
from tfx.orchestration.kubeflow.runner import KubeflowRunner
from tfx.proto import evaluator_pb2
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2

# Directory and data locations (uses Google Cloud Storage).
_input_bucket = 'gs://my-bucket'
_output_bucket = 'gs://my-bucket'
_pipeline_root = os.path.join(_output_bucket, 'tfx')

# Google Cloud Platform project id to use when deploying this pipeline.
_project_id = 'my-gcp-project'

# Python module file to inject customized logic into the TFX components. The
# Transform and Trainer both require user-defined functions to run successfully.
# Copy this from the current directory to a GCS bucket and update the location
# below.
_taxi_utils = os.path.join(_input_bucket, 'taxi_utils.py')

# Path which can be listened to by the model server.  Pusher will output the
# trained model here.
_serving_model_dir = os.path.join(_output_bucket, 'serving_model/taxi_bigquery')

# Region to use for Dataflow jobs and AI Platform training jobs.
#   Dataflow: https://cloud.google.com/dataflow/docs/concepts/regional-endpoints
#   AI Platform: https://cloud.google.com/ml-engine/docs/tensorflow/regions
_gcp_region = 'us-central1'

# A dict which contains the training job parameters to be passed to Google
# Cloud AI Platform. For the full set of parameters supported by Google Cloud AI
# Platform, refer to
# https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#Job
_ai_platform_training_args = {
    'project': _project_id,
    'region': _gcp_region,
    'jobDir': os.path.join(_output_bucket, 'tmp'),
    # Starting from TFX 0.14, 'runtimeVersion' is not relevant anymore.
    # Instead, it will be populated by TFX as <major>.<minor> version of
    # the imported TensorFlow package;
    'runtimeVersion': '1.12',
    # Starting from TFX 0.14, 'pythonVersion' is not relevant anymore.
    # Instead, it will be populated by TFX as the <major>.<minor> version
    # of the running Python interpreter;
    'pythonVersion': '2.7',
    # 'pythonModule' will be populated by TFX;
    # 'args' will be populated by TFX;
    # If 'packageUris' is not empty, AI Platform trainer will assume this
    # will populate this field with an ephemeral TFX package built from current
    # installation.
}

# A dict which contains the serving job parameters to be passed to Google
# Cloud AI Platform. For the full set of parameters supported by Google Cloud AI
# Platform, refer to
# https://cloud.google.com/ml-engine/reference/rest/v1/projects.models
_ai_platform_serving_args = {
    'model_name': 'chicago_taxi',
    'project_id': _project_id,
    # 'runtimeVersion' will be populated by TFX as <major>.<minor> version of
    #   the imported TensorFlow package;
}

# The rate at which to sample rows from the Chicago Taxi dataset using BigQuery.
# The full taxi dataset is > 120M record.  In the interest of resource
# savings and time, we've set the default for this example to be much smaller.
# Feel free to crank it up and process the full dataset!
_query_sample_rate = 0.001  # Generate a 0.1% random sample.

# This is the upper bound of FARM_FINGERPRINT in Bigquery (ie the max value of
# signed int64).
_max_int64 = '0x7FFFFFFFFFFFFFFF'

# The query that extracts the examples from BigQuery.  The Chicago Taxi dataset
# used for this example is a public dataset available on Google AI Platform.
# https://console.cloud.google.com/marketplace/details/city-of-chicago-public-data/chicago-taxi-trips
_query = """
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
         WHERE (ABS(FARM_FINGERPRINT(unique_key)) / {max_int64})
           < {query_sample_rate}""".format(
               max_int64=_max_int64, query_sample_rate=_query_sample_rate)


def _create_pipeline():
  """Implements the chicago taxi pipeline with TFX and Kubeflow Pipelines."""

  # Brings data into the pipeline or otherwise joins/converts training data.
  example_gen = BigQueryExampleGen(query=_query)

  # Computes statistics over data for visualization and example validation.
  statistics_gen = StatisticsGen(input_data=example_gen.outputs.examples)

  # Generates schema based on statistics files.
  infer_schema = SchemaGen(stats=statistics_gen.outputs.output)

  # Performs anomaly detection based on statistics and data schema.
  validate_stats = ExampleValidator(
      stats=statistics_gen.outputs.output,
      schema=infer_schema.outputs.output)

  # Performs transformations and feature engineering in training and serving.
  transform = Transform(
      input_data=example_gen.outputs.examples,
      schema=infer_schema.outputs.output,
      module_file=_taxi_utils)

  # Uses user-provided Python function that implements a model using TF-Learn
  # to train a model on Google Cloud AI Platform.
  try:
    from tfx.extensions.google_cloud_ai_platform.trainer import executor as ai_platform_trainer_executor  # pylint: disable=g-import-not-at-top
    # Train using a custom executor. This requires TFX >= 0.14.
    trainer = Trainer(
        executor_class=ai_platform_trainer_executor.Executor,
        module_file=_taxi_utils,
        transformed_examples=transform.outputs.transformed_examples,
        schema=infer_schema.outputs.output,
        transform_output=transform.outputs.transform_output,
        train_args=trainer_pb2.TrainArgs(num_steps=10000),
        eval_args=trainer_pb2.EvalArgs(num_steps=5000),
        custom_config={'ai_platform_training_args': _ai_platform_training_args})
  except ImportError:
    # Train using a deprecated flag.
    trainer = Trainer(
        module_file=_taxi_utils,
        transformed_examples=transform.outputs.transformed_examples,
        schema=infer_schema.outputs.output,
        transform_output=transform.outputs.transform_output,
        train_args=trainer_pb2.TrainArgs(num_steps=10000),
        eval_args=trainer_pb2.EvalArgs(num_steps=5000),
        custom_config={'cmle_training_args': _ai_platform_training_args})

  # Uses TFMA to compute a evaluation statistics over features of a model.
  model_analyzer = Evaluator(
      examples=example_gen.outputs.examples,
      model_exports=trainer.outputs.output,
      feature_slicing_spec=evaluator_pb2.FeatureSlicingSpec(specs=[
          evaluator_pb2.SingleSlicingSpec(
              column_for_slicing=['trip_start_hour'])
      ]))

  # Performs quality validation of a candidate model (compared to a baseline).
  model_validator = ModelValidator(
      examples=example_gen.outputs.examples, model=trainer.outputs.output)

  # Checks whether the model passed the validation steps and pushes the model
  # to a destination if check passed.
  try:
    from tfx.extensions.google_cloud_ai_platform.pusher import executor as ai_platform_pusher_executor  # pylint: disable=g-import-not-at-top
    # Deploy the model on Google Cloud AI Platform. This requires TFX >=0.14.
    pusher = Pusher(
        executor_class=ai_platform_pusher_executor.Executor,
        model_export=trainer.outputs.output,
        model_blessing=model_validator.outputs.blessing,
        custom_config={'ai_platform_serving_args': _ai_platform_serving_args})
  except ImportError:
    # Deploy the model on Google Cloud AI Platform, using a deprecated flag.
    pusher = Pusher(
        model_export=trainer.outputs.output,
        model_blessing=model_validator.outputs.blessing,
        custom_config={'cmle_serving_args': _ai_platform_serving_args},
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=_serving_model_dir)))

  return pipeline.Pipeline(
      pipeline_name='chicago_taxi_pipeline_kubeflow',
      pipeline_root=_pipeline_root,
      components=[
          example_gen, statistics_gen, infer_schema, validate_stats,
          transform, trainer, model_analyzer, model_validator, pusher
      ],
      additional_pipeline_args={
          'beam_pipeline_args': [
              '--runner=DataflowRunner',
              '--experiments=shuffle_mode=auto',
              '--project=' + _project_id,
              '--temp_location=' + os.path.join(_output_bucket, 'tmp'),
              '--region=' + _gcp_region,
          ],
          # Optional args:
          # 'tfx_image': custom docker image to use for components.
          # This is needed if TFX package is not installed from an RC
          # or released version.
      },
      log_root='/var/tmp/tfx/logs',
  )


_ = KubeflowRunner().run(_create_pipeline())
