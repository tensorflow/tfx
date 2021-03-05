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
"""Helper class to start TFX training jobs on AI Platform."""
# TODO(b/168926785): Consider to move some methods to a utility file.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
from typing import Any, Dict, List, Optional, Text

from absl import logging
from googleapiclient import discovery
from googleapiclient import errors
import tensorflow as tf

from tfx import types
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

_TF_COMPATIBILITY_OVERRIDE = {
    # Generally, runtimeVersion should be same as <major>.<minor> of currently
    # installed tensorflow version, with certain compatibility hacks since
    # some TensorFlow runtime versions are not explicitly supported by
    # CAIP pusher. See:
    # https://cloud.google.com/ai-platform/prediction/docs/runtime-version-list
    '2.0': '1.15',
    # TODO(b/168249383) Update this once CAIP model support TF 2.4 runtime.
    '2.4': '2.3',
    '2.5': '2.3',
}

# Default endpoint for v1 API.
DEFAULT_ENDPOINT = 'https://ml.googleapis.com'
# Default API version.
_DEFAULT_API_VERSION = 'v1'


def _get_tf_runtime_version(tf_version: Text) -> Text:
  """Returns the tensorflow runtime version used in Cloud AI Platform.

  This is only used for prediction service.

  Args:
    tf_version: version string returned from `tf.__version__`.
  Returns: same major.minor version of installed tensorflow, except when
    overriden by _TF_COMPATIBILITY_OVERRIDE.
  """
  tf_version = '.'.join(tf_version.split('.')[0:2])
  return _TF_COMPATIBILITY_OVERRIDE.get(tf_version) or tf_version


# TODO(b/180967044): This can be removed, and use 3.7 as default.
def _get_caip_python_version(caip_tf_runtime_version: Text) -> Text:
  """Returns supported python version on Cloud AI Platform.

  See
  https://cloud.google.com/ml-engine/docs/tensorflow/versioning#set-python-version-training

  Args:
    caip_tf_runtime_version: version string returned from
      _get_tf_runtime_version().

  Returns:
    '2.7' for PY2. '3.5' or '3.7' for PY3 depending on caip_tf_runtime_version.
  """
  if sys.version_info.major == 2:
    return '2.7'
  (major, minor) = caip_tf_runtime_version.split('.')[0:2]
  if (int(major), int(minor)) >= (1, 15):
    return '3.7'
  return '3.5'


def _launch_aip_training(
    job_id: Text,
    project: Text,
    training_input: Dict[Text, Any],
    job_labels: Optional[Dict[Text, Text]] = None) -> None:
  """Launches and monitors a AIP custom training job.

  Args:
    job_id: The job ID of the AI Platform training job.
    project: The GCP project under which the training job will be executed.
    training_input: Training input argument for AI Platform training job. See
      https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#TrainingInput
      for the detailed schema.
    job_labels: The dict of labels that will be attached to this job.

  Raises:
    RuntimeError: if the Google Cloud AI Platform training job failed/cancelled.
    ConnectionError: if the status polling of the training job failed due to
      connection issue.
  """
  client = training_clients.get_job_client()
  # Configure AI Platform training job
  project_id = 'projects/{}'.format(project)

  client.launch_job(job_id, project_id, training_input, job_labels)

  # Wait for AIP Training job to finish
  job_name = '{}/jobs/{}'.format(project_id, job_id)
  request = client.get_job_request()
  response = request.execute()
  retry_count = 0

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
  while response['state'] not in client.JOB_STATES_COMPLETED:
    time.sleep(_POLLING_INTERVAL_IN_SECONDS)
    try:
      response = request.execute()
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
        request = client.get_job_request()
      else:
        logging.error('Request failed after %s retries.',
                      _CONNECTION_ERROR_RETRY_LIMIT)
        raise

  if response['state'] in client.JOB_STATES_FAILED:
    err_msg = 'Job \'{}\' did not succeed.  Detailed response {}.'.format(
        job_name, response)
    logging.error(err_msg)
    raise RuntimeError(err_msg)

  # AIP training complete
  logging.info('Job \'%s\' successful.', job_name)


def _wait_for_operation(api: discovery.Resource, operation: Dict[Text, Any],
                        method_name: Text) -> Dict[Text, Any]:
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
def start_aip_training(input_dict: Dict[Text, List[types.Artifact]],
                       output_dict: Dict[Text, List[types.Artifact]],
                       exec_properties: Dict[Text,
                                             Any], executor_class_path: Text,
                       training_inputs: Dict[Text,
                                             Any], job_id: Optional[Text]):
  """Start a trainer job on AI Platform (AIP).

  This is done by forwarding the inputs/outputs/exec_properties to the
  tfx.scripts.run_executor module on a AI Platform training job interpreter.

  Args:
    input_dict: Passthrough input dict for tfx.components.Trainer.executor.
    output_dict: Passthrough input dict for tfx.components.Trainer.executor.
    exec_properties: Passthrough input dict for tfx.components.Trainer.executor.
    executor_class_path: class path for TFX core default trainer.
    training_inputs: Training input argument for AI Platform training job.
      'pythonModule', 'pythonVersion' and 'runtimeVersion' will be inferred. For
      the full set of parameters, refer to
      https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#TrainingInput
    job_id: Job ID for AI Platform Training job. If not supplied,
      system-determined unique ID is given. Refer to
    https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#resource-job

  Returns:
    None
  """
  client = training_clients.get_job_client()
  training_args = client.create_training_args(input_dict, output_dict,
                                              exec_properties,
                                              executor_class_path,
                                              training_inputs, job_id)

  _launch_aip_training(
      job_id=training_args['job_id'],
      project=training_args['project'],
      training_input=training_args['training_input'],
      job_labels=training_args['job_labels'])


# TODO(zhitaoli): remove this function since we are not going to support
# more API versions on existing Cloud AI Platform.
def get_service_name_and_api_version(
    ai_platform_serving_args: Dict[Text, Any]):  # -> Tuple[Text, Text]
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
    api: discovery.Resource,
    job_labels: Dict[Text, Text],
    ai_platform_serving_args: Dict[Text, Any],
) -> bool:
  """Creates a new model for serving with AI Platform if not exists.

  Args:
    api: Google API client resource.
    job_labels: The dict of labels that will be attached to this job.
    ai_platform_serving_args: Dictionary containing arguments for pushing to AI
      Platform.

  Returns:
    Whether a new model is created.

  Raises:
    RuntimeError if model creation failed.
  """

  model_name = ai_platform_serving_args['model_name']
  project_id = ai_platform_serving_args['project_id']
  regions = ai_platform_serving_args.get('regions', [])
  body = {'name': model_name, 'regions': regions, 'labels': job_labels}
  parent = 'projects/{}'.format(project_id)
  result = True
  try:
    api.projects().models().create(body=body, parent=parent).execute()
  except errors.HttpError as e:
    # If the error is to create an already existing model, it's ok to ignore.
    if e.resp.status == 409:
      logging.warn('Model %s already exists', model_name)
      result = False
    else:
      raise RuntimeError('Creating model to AI Platform failed: {}'.format(e))
  return result


def deploy_model_for_aip_prediction(api: discovery.Resource,
                                    serving_path: Text,
                                    model_version: Text,
                                    ai_platform_serving_args: Dict[Text, Any],
                                    job_labels: Dict[Text, Text],
                                    skip_model_creation: bool = False,
                                    set_default_version: bool = True) -> None:
  """Deploys a model for serving with AI Platform.

  Args:
    api: Google API client resource.
    serving_path: The path to the model. Must be a GCS URI.
    model_version: Version of the model being deployed. Must be different from
      what is currently being served.
    ai_platform_serving_args: Dictionary containing arguments for pushing to AI
      Platform. The full set of parameters supported can be found at
      https://cloud.google.com/ml-engine/reference/rest/v1/projects.models.versions#Version.
      Most keys are forwarded as-is, but following keys are handled specially:
        - name: this must be empty (and will be filled by pusher).
        - deployment_uri: this must be empty (and will be filled by pusher).
        - python_version: when left empty, this will be filled by python version
            of the environment being used.
        - runtime_version: when left empty, this will be filled by TensorFlow
            version from the environment.
        - labels: a list of job labels will be merged with user's input.
    job_labels: The dict of labels that will be attached to this job. They are
      merged with optional labels from `ai_platform_serving_args`.
    skip_model_creation: If true, the method assuem model already exist in
      AI platform, therefore skipping model creation.
    set_default_version: Whether set the newly deployed model version as the
      default version.

  Raises:
    RuntimeError: if an error is encountered when trying to push.
  """
  logging.info(
      'Deploying to model with version %s to AI Platform for serving: %s',
      model_version, ai_platform_serving_args)

  model_name = ai_platform_serving_args['model_name']
  project_id = ai_platform_serving_args['project_id']
  default_runtime_version = _get_tf_runtime_version(tf.__version__)
  runtime_version = ai_platform_serving_args.get('runtime_version',
                                                 default_runtime_version)
  python_version = _get_caip_python_version(runtime_version)

  if not skip_model_creation:
    create_model_for_aip_prediction_if_not_exist(api, job_labels,
                                                 ai_platform_serving_args)
  version_body = dict(ai_platform_serving_args)
  for model_only_key in ['model_name', 'project_id', 'regions']:
    version_body.pop(model_only_key, None)
  version_body['name'] = model_version
  version_body['deployment_uri'] = serving_path
  version_body['runtime_version'] = version_body.get('runtime_version',
                                                     runtime_version)
  version_body['python_version'] = version_body.get('python_version',
                                                    python_version)
  version_body['labels'] = {**version_body.get('labels', {}), **job_labels}
  logging.info(
      'Creating new version of model_name %s in project %s, request body: %s',
      model_name, project_id, version_body)

  # Push to AIP, and record the operation name so we can poll for its state.
  model_name = 'projects/{}/models/{}'.format(project_id, model_name)
  try:
    operation = api.projects().models().versions().create(
        body=version_body, parent=model_name).execute()
    _wait_for_operation(api, operation, 'projects.models.versions.create')
  except errors.HttpError as e:
    # If the error is to create an already existing model version, it's ok to
    # ignore.
    if e.resp.status == 409:
      logging.warn('Model version %s already exists', model_version)
    else:
      raise RuntimeError('Creating model verseion to AI Platform failed: {}'
                         .format(e))

  if set_default_version:
    # Set the new version as default.
    # By API specification, if Long-Running-Operation is done and there is
    # no error, 'response' is guaranteed to exist.
    api.projects().models().versions().setDefault(name='{}/versions/{}'.format(
        model_name, model_version)).execute()

  logging.info(
      'Successfully deployed model %s with version %s, serving from %s',
      model_name, model_version, serving_path)


def delete_model_version_from_aip_if_exists(
    api: discovery.Resource,
    model_version: Text,
    ai_platform_serving_args: Dict[Text, Any],
) -> None:
  """Deletes a model version from Google Cloud AI Platform if version exists.

  Args:
    api: Google API client resource.
    model_version: Version of the model being deleted.
    ai_platform_serving_args: Dictionary containing arguments for pushing to AI
      Platform. For the full set of parameters supported, refer to
      https://cloud.google.com/ml-engine/reference/rest/v1/projects.models

  Raises:
    RuntimeError: if an error is encountered when trying to delete.
  """
  logging.info('Deleting model version %s from AI Platform: %s', model_version,
               ai_platform_serving_args)
  model_name = ai_platform_serving_args['model_name']
  project_id = ai_platform_serving_args['project_id']
  version_name = 'projects/{}/models/{}/versions/{}'.format(
      project_id, model_name, model_version)
  try:
    operation = api.projects().models().versions().delete(
        name=version_name).execute()
    _wait_for_operation(api, operation, 'projects.models.versions.delete')
  except errors.HttpError as e:
    # If the error is to delete an non-exist model version, it's ok to ignore.
    if e.resp.status == 404:
      logging.warn('Model version %s does not exist', version_name)
    if e.resp.status == 400:
      logging.warn('Model version %s won\'t be deleted because it is the '
                   'default version and not the only version in the model',
                   version_name)
    else:
      raise RuntimeError(
          'Deleting model version {} from AI Platform failed: {}'.format(
              version_name, e))


def delete_model_from_aip_if_exists(
    api: discovery.Resource,
    ai_platform_serving_args: Dict[Text, Any],
) -> None:
  """Deletes a model from Google Cloud AI Platform if exists.

  Args:
    api: Google API client resource.
    ai_platform_serving_args: Dictionary containing arguments for pushing to AI
      Platform. For the full set of parameters supported, refer to
      https://cloud.google.com/ml-engine/reference/rest/v1/projects.models

  Raises:
    RuntimeError: if an error is encountered when trying to delete.
  """
  logging.info('Deleting model with from AI Platform: %s',
               ai_platform_serving_args)
  model_name = ai_platform_serving_args['model_name']
  project_id = ai_platform_serving_args['project_id']
  name = 'projects/{}/models/{}'.format(project_id, model_name)
  try:
    operation = api.projects().models().delete(name=name).execute()
    _wait_for_operation(api, operation, 'projects.models.delete')
  except errors.HttpError as e:
    # If the error is to delete an non-exist model, it's ok to ignore.
    if e.resp.status == 404:
      logging.warn('Model %s does not exist', model_name)
    else:
      raise RuntimeError('Deleting model from AI Platform failed: {}'.format(e))
