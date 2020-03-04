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
"""TFX taxi template configurations.

This file defines environments for a TFX taxi pipeline.
"""

import os  # pylint: disable=unused-import

from tfx.proto import trainer_pb2

# TODO(b/149347293): Move more TFX CLI flags into python configuration.

# Pipeline name will be used to identify this pipeline.
PIPELINE_NAME = '{{PIPELINE_NAME}}'

# GCP related configs.
# These configs are only useful if you are using Google Cloud.
# TODO(step 4,step 8): Specify your GCS bucket name here.
#                      You have to use GCS to store output files when running a
#                      pipeline with Kubeflow Pipeline on GCP or when running a
#                      job using Dataflow.
GCS_BUCKET_NAME = 'YOUR_GCS_BUCKET_NAME'

# TODO(step 7,step 8,step 9): (Optional) Set your project ID and region to use
#                             GCP services including BigQuery, Dataflow and
#                             Cloud AI Platform.
# GCP_PROJECT_ID = 'YOUR_GCP_PROJECT_ID'
# GCP_REGION = 'YOUR_GCP_REGION'  # ex) 'us-central1'

PREPROCESSING_FN = 'preprocessing.preprocessing_fn'
TRAINER_FN = 'model.trainer_fn'

TRAIN_ARGS = trainer_pb2.TrainArgs(num_steps=100)
EVAL_ARGS = trainer_pb2.EvalArgs(num_steps=100)

# Beam args to use BigQueryExampleGen.
# TODO(step 7): (Optional) Uncomment here to provide GCP related configs for
#               BigQuery.
# BIG_QUERY_BEAM_PIPELINE_ARGS = [
#    '--project=' + GCP_PROJECT_ID,
#    ]

# The rate at which to sample rows from the Chicago Taxi dataset using BigQuery.
# The full taxi dataset is > 120M record.  In the interest of resource
# savings and time, we've set the default for this example to be much smaller.
# Feel free to crank it up and process the full dataset!
_query_sample_rate = 0.0001  # Generate a 0.01% random sample.

# The query that extracts the examples from BigQuery.  The Chicago Taxi dataset
# used for this example is a public dataset available on Google AI Platform.
# https://console.cloud.google.com/marketplace/details/city-of-chicago-public-data/chicago-taxi-trips
# TODO(step 7): (Optional) Uncomment here to use BigQuery.
# BIG_QUERY_QUERY = """
#         SELECT
#           pickup_community_area,
#           fare,
#           EXTRACT(MONTH FROM trip_start_timestamp) AS trip_start_month,
#           EXTRACT(HOUR FROM trip_start_timestamp) AS trip_start_hour,
#           EXTRACT(DAYOFWEEK FROM trip_start_timestamp) AS trip_start_day,
#           UNIX_SECONDS(trip_start_timestamp) AS trip_start_timestamp,
#           pickup_latitude,
#           pickup_longitude,
#           dropoff_latitude,
#           dropoff_longitude,
#           trip_miles,
#           pickup_census_tract,
#           dropoff_census_tract,
#           payment_type,
#           company,
#           trip_seconds,
#           dropoff_community_area,
#           tips
#         FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
#         WHERE (ABS(FARM_FINGERPRINT(unique_key)) / 0x7FFFFFFFFFFFFFFF)
#           < {query_sample_rate}""".format(
#    query_sample_rate=_query_sample_rate)

# Beam args to run data processing on DataflowRunner.
# TODO(step 8): (Optional) Uncomment below to use Dataflow.
# BEAM_PIPELINE_ARGS = [
#    '--project=' + GCP_PROJECT_ID,
#    '--runner=DataflowRunner',
#    '--experiments=shuffle_mode=auto',
#    '--temp_location=' + os.path.join('gs://', GCS_BUCKET_NAME, 'tmp'),
#    '--region=' + GCP_REGION,
#    ]

# A dict which contains the training job parameters to be passed to Google
# Cloud AI Platform. For the full set of parameters supported by Google Cloud AI
# Platform, refer to
# https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#Job
# TODO(step 9): (Optional) Uncomment below to use AI Platform training.
# GCP_AI_PLATFORM_TRAINING_ARGS = {
#     'project': GCP_PROJECT_ID,
#     'region': GCP_REGION,
#     # Starting from TFX 0.14, training on AI Platform uses custom containers:
#     # https://cloud.google.com/ml-engine/docs/containers-overview
#     # You can specify a custom container here. If not specified, TFX will use
#     # a public container image matching the installed version of TFX.
#     # TODO(step 9): (Optional) Set your container name below.
#     'masterConfig': {
#       'imageUri': 'gcr.io/' + GCP_PROJECT_ID + '/tfx-pipeline'
#     },
#     # Note that if you do specify a custom container, ensure the entrypoint
#     # calls into TFX's run_executor script (tfx/scripts/run_executor.py)
# }

# A dict which contains the serving job parameters to be passed to Google
# Cloud AI Platform. For the full set of parameters supported by Google Cloud AI
# Platform, refer to
# https://cloud.google.com/ml-engine/reference/rest/v1/projects.models
# TODO(step 9): (Optional) Uncomment below to use AI Platform serving.
# GCP_AI_PLATFORM_SERVING_ARGS = {
#     'model_name': PIPELINE_NAME,
#     'project_id': GCP_PROJECT_ID,
#     # The region to use when serving the model. See available regions here:
#     # https://cloud.google.com/ml-engine/docs/regions
#     # Note that serving currently only supports a single region:
#     # https://cloud.google.com/ml-engine/reference/rest/v1/projects.models#Model  # pylint: disable=line-too-long
#     'regions': [GCP_REGION],
# }
