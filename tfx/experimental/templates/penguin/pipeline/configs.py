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
"""TFX penguin template configurations.

This file defines environments for a TFX penguin pipeline.
"""

import os  # pylint: disable=unused-import

# TODO(b/149347293): Move more TFX CLI flags into python configuration.

# Pipeline name will be used to identify this pipeline.
PIPELINE_NAME = '{{PIPELINE_NAME}}'

# GCP related configs.

# Following code will retrieve your GCP project. You can choose which project
# to use by setting GOOGLE_CLOUD_PROJECT environment variable.
try:
  import google.auth  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error
  try:
    _, GOOGLE_CLOUD_PROJECT = google.auth.default()
  except google.auth.exceptions.DefaultCredentialsError:
    GOOGLE_CLOUD_PROJECT = ''
except ImportError:
  GOOGLE_CLOUD_PROJECT = ''

# Specify your GCS bucket name here. You have to use GCS to store output files
# when running a pipeline with Kubeflow Pipeline on GCP or when running a job
# using Dataflow. Default is '<gcp_project_name>-kubeflowpipelines-default'.
# This bucket is created automatically when you deploy KFP from marketplace.
GCS_BUCKET_NAME = GOOGLE_CLOUD_PROJECT + '-kubeflowpipelines-default'

# Following image will be used to run pipeline components run if Kubeflow
# Pipelines used.
# This image will be automatically built by CLI if we use --build-image flag.
PIPELINE_IMAGE = f'gcr.io/{GOOGLE_CLOUD_PROJECT}/{PIPELINE_NAME}'

PREPROCESSING_FN = 'models.preprocessing.preprocessing_fn'
RUN_FN = 'models.model.run_fn'

TRAIN_NUM_STEPS = 100
EVAL_NUM_STEPS = 15

# Change this value according to your use cases.
EVAL_ACCURACY_THRESHOLD = 0.6

# Google Cloud BigQuery related configs.
# Use following configs to use BigQueryExampleGen as a data source.
#
# Beam args to use BigQueryExampleGen with Beam DirectRunner.
#
# BIG_QUERY_WITH_DIRECT_RUNNER_BEAM_PIPELINE_ARGS = [
#    '--project=' + GOOGLE_CLOUD_PROJECT,
#    '--temp_location=' + os.path.join('gs://', GCS_BUCKET_NAME, 'tmp'),
#    ]

# The query that extracts the examples from BigQuery.
#
# BIG_QUERY_QUERY = """
#         SELECT ...
#         FROM
#         WHERE
# """
