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
"""Helper class to start TFX training jobs on AI Platform."""
# TODO(b/168926785): Consider to move some methods to a utility file.

import time
from typing import Any, Dict, List, Optional

from absl import logging
from googleapiclient import discovery

from tfx import types
from tfx.extensions.google_cloud_ai_platform import prediction_clients
from tfx.extensions.google_cloud_ai_platform import training_clients
from tfx.utils import version_utils

_POLLING_INTERVAL_IN_SECONDS = 30

_CONNECTION_ERROR_RETRY_LIMIT = 5

# Default container image being used for CAIP training jobs.
# TODO(b/139934802) Ensure mirroring of released TFX containers in Docker Hub
# and gcr.io/tfx-oss-public/ registries.
_TFX_IMAGE = 'gcr.io/tfx-oss-public/tfx:{}'.format(
    version_utils.get_image_version())

# Entrypoint of cloud AI platform training. The module comes from `tfx`
# package installation into a default location of 'python'.
_CONTAINER_COMMAND = ['python', '-m', 'tfx.scripts.run_executor']

# Default endpoint for v1 API.
DEFAULT_ENDPOINT = 'https://ml.googleapis.com'
# Default API version.
_DEFAULT_API_VERSION = 'v1'


def _launch_cloud_training(project: str,
                           training_job: Dict[str, Any],
                           enable_vertex: Optional[bool] = False,
                           vertex_region: Optional[str] = None) -> None:
  """Launches and monitors a Cloud custom training job.

  Args:
    project: The GCP project under which the training job will be executed.
    training_job: Training job argument for AI Platform training job. See
      https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.customJobs#CustomJob
        for detailed schema for the Vertex CustomJob. See
      https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs for the
        detailed schema for CAIP Job.
    enable_vertex: Whether to enable Vertex or not.
    vertex_region: Region for endpoint in Vertex training.

  Raises:
    RuntimeError: if the Google Cloud AI Platform training job failed/cancelled.
    ConnectionError: if the status polling of the training job failed due to
      connection issue.
  """
  # TODO(b/185159702): Migrate all training jobs to Vertex and remove the
  #                    enable_vertex switch.
  client = training_clients.get_job_client(enable_vertex, vertex_region)
  # Configure and launch AI Platform training job.
  client.launch_job(project, training_job)

  # Wait for Cloud Training job to finish
  response = client.get_job()
  retry_count = 0

  job_id = client.get_job_name()
  # Monitors the long-running operation by polling the job state periodically,
  # and retries the polling when a transient connectivity issue is encountered.
  #
  # Long-running operation monitoring:
  #   The possible states of "get job" response can be found at
  #   https://cloud.google.com/ai-platform/training/docs/reference/rest/v1/projects.jobs#State
  #   where SUCCEEDED/FAILED/CANCELLED are considered to be final states.
  #   The following logic will keep polling the state of the job until the job
  #   enters a final state.
  #
  # During the polling, if a connection error was encountered, the GET request
  # will be retried by recreating the Python API client to refresh the lifecycle
  # of the connection being used. See
  # https://github.com/googleapis/google-api-python-client/issues/218
  # for a detailed description of the problem. If the error persists for
  # _CONNECTION_ERROR_RETRY_LIMIT consecutive attempts, the function will raise
  # ConnectionError.
  while client.get_job_state(response) not in client.JOB_STATES_COMPLETED:
    time.sleep(_POLLING_INTERVAL_IN_SECONDS)
    try:
      response = client.get_job()
      retry_count = 0
    # Handle transient connection error.
    except ConnectionError as err:
      if retry_count < _CONNECTION_ERROR_RETRY_LIMIT:
        retry_count += 1
        logging.warning(
            'ConnectionError (%s) encountered when polling job: %s. Trying to '
            'recreate the API client.', err, job_id)
        # Recreate the Python API client.
        client.create_client()
      else:
        logging.error('Request failed after %s retries.',
                      _CONNECTION_ERROR_RETRY_LIMIT)
        raise

  if client.get_job_state(response) in client.JOB_STATES_FAILED:
    err_msg = 'Job \'{}\' did not succeed.  Detailed response {}.'.format(
        client.get_job_name(), response)
    logging.error(err_msg)
    raise RuntimeError(err_msg)

  # Cloud training complete
  logging.info('Job \'%s\' successful.', client.get_job_name())


def _wait_for_operation(api: discovery.Resource, operation: Dict[str, Any],
                        method_name: str) -> Dict[str, Any]:
  """Wait for a long running operation.

  Args:
    api: Google API client resource.
    operation: The operation to wait for.
    method_name: Operation method name for logging.

  Returns:
    Operation completion status.

  Raises:
    RuntimeError: If the operation completed with an error.
  """
  status_resc = api.projects().operations().get(name=operation['name'])
  while not status_resc.execute().get('done'):
    time.sleep(_POLLING_INTERVAL_IN_SECONDS)
    logging.info('Method %s still being executed...', method_name)
  result = status_resc.execute()
  if result.get('error'):
    # The operation completed with an error.
    raise RuntimeError('Failed to execute {}: {}'.format(
        method_name, result['error']))
  return result


# TODO(b/168926785): Consider to change executor_class_path to job_labels.
def start_cloud_training(
    input_dict: Dict[str, List[types.Artifact]],
    output_dict: Dict[str, List[types.Artifact]],
    exec_properties: Dict[str, Any],
    executor_class_path: str,
    job_args: Dict[str, Any],
    job_id: Optional[str],
    job_labels: Optional[Dict[str, Any]] = None,  # AI Platform only.
    enable_vertex: Optional[bool] = False,
    vertex_region: Optional[str] = None):
  """Start a trainer job on AI Platform (AIP).

  This is done by forwarding the inputs/outputs/exec_properties to the
  tfx.scripts.run_executor module on a AI Platform training job interpreter.

  Args:
    input_dict: Passthrough input dict for tfx.components.Trainer.executor.
    output_dict: Passthrough input dict for tfx.components.Trainer.executor.
    exec_properties: Passthrough input dict for tfx.components.Trainer.executor.
    executor_class_path: class path for TFX core default trainer.
    job_args: Training argument for AI Platform training job. 'pythonModule',
      'pythonVersion' and 'runtimeVersion' will be inferred. For the full set of
      parameters supported by Vertex AI CustomJob, refer to
       https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.customJobs#CustomJob
         For the full set of parameters supported by Google Cloud AI Platform
         (CAIP) TrainingInput, refer to
       https://cloud.google.com/ml-engine/docs/tensorflow/training-jobs#configuring_the_job
    job_id: Job ID for AI Platform Training job. If not supplied,
      system-determined unique ID is given. Refer to
      https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#resource-job.
        In Vertex AI, the job_id corresponds to the display name, a unique ID is
        always given to the created job.
    job_labels: Labels for AI Platform training job.
    enable_vertex: Whether to enable Vertex or not.
    vertex_region: Region for endpoint in Vertex training.

  Returns:
    None
  """
  # Project was stowaway in job_args and has finally reached its destination.
  project = job_args.pop('project')

  client = training_clients.get_job_client(enable_vertex, vertex_region)
  training_job = client.create_training_job(input_dict, output_dict,
                                            exec_properties,
                                            executor_class_path, job_args,
                                            job_id, job_labels)

  _launch_cloud_training(
      project=project,
      training_job=training_job,
      enable_vertex=enable_vertex,
      vertex_region=vertex_region)


# TODO(zhitaoli): remove this function since we are not going to support
# more API versions on existing Cloud AI Platform.
def get_service_name_and_api_version(
    ai_platform_serving_args: Dict[str, Any]):  # -> Tuple[Text, Text]
  """Gets service name and api version from ai_platform_serving_args.

  Args:
    ai_platform_serving_args: Dictionary containing arguments for pushing to AI
      Platform.

  Returns:
    Service name and API version.
  """
  del ai_platform_serving_args
  return ('ml', _DEFAULT_API_VERSION)


def create_model_for_aip_prediction_if_not_exist(
    labels: Dict[str, str],
    ai_platform_serving_args: Dict[str, Any],
    api: Optional[discovery.Resource] = None,
    enable_vertex: Optional[bool] = False) -> bool:
  """Creates a new CAIP model or Vertex endpoint for serving with AI Platform if not exists.

  Args:
    labels: The dict of labels that will be attached to this CAIP job or Vertex
      endpoint.
    ai_platform_serving_args: Dictionary containing arguments for pushing to AI
      Platform.
    api: (CAIP only, required) Google API client resource.
    enable_vertex: Whether to enable Vertex or not.

  Returns:
    Whether a new CAIP model or Vertex endpoint is created.

  Raises:
    RuntimeError if creation failed.
  """

  client = prediction_clients.get_prediction_client(
      api=api, enable_vertex=enable_vertex)
  return client.create_model_for_aip_prediction_if_not_exist(
      labels=labels, ai_platform_serving_args=ai_platform_serving_args)


def deploy_model_for_aip_prediction(
    serving_path: str,
    model_version_name: str,
    ai_platform_serving_args: Dict[str, Any],
    labels: Dict[str, str],
    api: Optional[discovery.Resource] = None,
    serving_container_image_uri: Optional[str] = None,
    endpoint_region: Optional[str] = None,
    skip_model_endpoint_creation: Optional[bool] = False,
    set_default: Optional[bool] = True,
    enable_vertex: Optional[bool] = False) -> str:
  """Deploys a model for serving with AI Platform.

  Args:
    serving_path: The path to the model. Must be a GCS URI.
    model_version_name: Model version for CAIP model being deployed, or model
      name for the Vertex model being deployed. Must be different from what is
      currently being served.
    ai_platform_serving_args: Dictionary containing arguments for pushing to AI
      Platform. The full set of parameters supported can be found at,
      for CAIP:
      https://cloud.google.com/ml-engine/reference/rest/v1/projects.models.versions#Version.
      for Vertex:
      https://googleapis.dev/python/aiplatform/latest/aiplatform.html?highlight=deploy#google.cloud.aiplatform.Model.deploy
        Most keys are forwarded as-is, but following keys are handled specially.
      For CAIP:
        - name: this must be empty (and will be filled by pusher).
        - deployment_uri: this must be empty (and will be filled by pusher).
        - python_version: when left empty, this will be filled by python version
          of the environment being used.
        - runtime_version: when left empty, this will be filled by TensorFlow
          version from the environment.
        - labels: a list of job labels will be merged with user's input.
      For Vertex:
        - endpoint_name: Name of the endpoint.
        - traffic_percentage: Desired traffic to newly deployed model. Forwarded
          as-is if specified. If not specified, it is set to 100 if
          set_default_version is True, or set to 0 otherwise.
        - labels: a list of job labels will be merged with user's input.
    labels: The dict of labels that will be attached to this CAIP job or Vertex
      endpoint. They are merged with optional labels from
      `ai_platform_serving_args`.
    api: (CAIP only, required) Google API client resource.
    serving_container_image_uri: (Vertex only, required) The path to the serving
      container image URI. Container registry for prediction is available at:
      https://gcr.io/cloud-aiplatform/prediction.
    endpoint_region: (Vertex only, required) Region for Vertex endpoint. For
      available regions, please see
      https://cloud.google.com/vertex-ai/docs/general/locations
    skip_model_endpoint_creation: If true, the method assumes CAIP model or
      Vertex endpoint already exists in AI platform, therefore skipping its
      creation.
    set_default: Whether set the newly deployed CAIP model version or Vertex
      model as the default.
    enable_vertex: Whether to enable Vertex or not.

  Returns:
    For Vertex, the resource name of the deployed model.

  Raises:
    RuntimeError: if an error is encountered when trying to push.
  """
  client = prediction_clients.get_prediction_client(
      api=api, enable_vertex=enable_vertex)
  if enable_vertex:
    return client.deploy_model(
        serving_path=serving_path,
        model_version_name=model_version_name,
        ai_platform_serving_args=ai_platform_serving_args,
        labels=labels,
        serving_container_image_uri=serving_container_image_uri,
        endpoint_region=endpoint_region,
        skip_model_endpoint_creation=skip_model_endpoint_creation,
        set_default=set_default)
  else:
    return client.deploy_model(
        serving_path=serving_path,
        model_version_name=model_version_name,
        ai_platform_serving_args=ai_platform_serving_args,
        labels=labels,
        skip_model_endpoint_creation=skip_model_endpoint_creation,
        set_default=set_default)


def delete_model_from_aip_if_exists(
    ai_platform_serving_args: Dict[str, Any],
    api: Optional[discovery.Resource] = None,
    model_version_name: Optional[str] = None,
    delete_model_endpoint: Optional[bool] = False,
    enable_vertex: Optional[bool] = False,
) -> None:
  """Deletes a model version from Google Cloud AI Platform if version exists.

  Args:
    ai_platform_serving_args: Dictionary containing arguments for pushing to AI
      Platform. For the full set of parameters supported, refer to
      https://cloud.google.com/ml-engine/reference/rest/v1/projects.models
    api: (CAIP only, required) Google API client resource.
    model_version_name: Model version for CAIP model being deployed, or model
      name for the Vertex model to be deleted. Required if delete_model_endpoint
      is False, otherwise not needed.
    delete_model_endpoint: Whether CAIP model or Vertex endpoint should be
      deleted.
    enable_vertex: Whether to enable Vertex or not.

  Raises:
    RuntimeError: if an error is encountered when trying to delete.
  """
  client = prediction_clients.get_prediction_client(
      api=api, enable_vertex=enable_vertex)
  client.delete_model_from_aip_if_exists(
      ai_platform_serving_args=ai_platform_serving_args,
      model_version_name=model_version_name,
      delete_model_endpoint=delete_model_endpoint)
