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
"""Common Flags."""

ENGINE_FLAG = 'engine'
PIPELINE_DSL_PATH = 'pipeline_dsl_path'
PIPELINE_NAME = 'pipeline_name'
AIRFLOW_PACKAGE_NAME = 'apache-airflow'
KUBEFLOW_PACKAGE_NAME = 'kfp'
RUN_ID = 'run_id'
AIRFLOW_ENGINE = 'airflow'
BEAM_ENGINE = 'beam'
KUBEFLOW_ENGINE = 'kubeflow'
LOCAL_ENGINE = 'local'
VERTEX_ENGINE = 'vertex'

# Path to root directory of the pipeline.
PIPELINE_ROOT = 'pipeline_root'
# List of components in the pipeline.
PIPELINE_COMPONENTS = 'pipeline_components'

# Kubeflow specific labels.
# Base container image path.
BASE_IMAGE = 'build_base_image'
# Client ID for IAP protected endpoint.
IAP_CLIENT_ID = 'iap_client_id'
# Endpoint of the KFP API service to connect.
ENDPOINT = 'endpoint'
# Kubernetes namespace to connect to the KFP API.
NAMESPACE = 'namespace'
# Pipeline id generated when pipeline is uploaded to KFP server.
PIPELINE_ID = 'pipeline_id'
# Pipeline version id generated when pipeline is created or updated.
PIPELINE_VERSION_ID = 'pipeline_version_id'
# Experiment id generated when a new experiment is created on KFP server.
EXPERIMENT_ID = 'experiment_id'
# Flag to decide whether an image build is needed
BUILD_IMAGE = 'build_image'
# GCP Project ID for GCP API call.
GCP_PROJECT_ID = 'gcp_project_id'
# GCP Region for GCP API call.
GCP_REGION = 'gcp_region'
# Runtime parameter for next pipeline run.
RUNTIME_PARAMETER = 'runtime_parameter'

# Template specific labels.
# Destination directory path to copy files
DESTINATION_PATH = 'destination_path'
# Model kind of the copying template
MODEL = 'model'
