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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import json
import sys
import time
from typing import Any, Dict, List, Optional, Text

from absl import logging
from googleapiclient import discovery
from googleapiclient import errors
import tensorflow as tf

from tfx import types
from tfx import version
from tfx.types import artifact_utils
from tfx.utils import telemetry_utils

_POLLING_INTERVAL_IN_SECONDS = 30

_CONNECTION_ERROR_RETRY_LIMIT = 5

# TODO(b/139934802) Ensure mirroring of released TFX containers in Docker Hub
# and gcr.io/tfx-oss-public/ registries.
_TFX_IMAGE = 'gcr.io/tfx-oss-public/tfx:{}'.format(version.__version__)

_TF_COMPATIBILITY_OVERRIDE = {
    # Generally, runtimeVersion should be same as <major>.<minor> of currently
    # installed tensorflow version, with certain compatibility hacks since
    # some TensorFlow runtime versions are not explicitly supported by
    # CAIP pusher. See:
    # https://cloud.google.com/ai-platform/prediction/docs/runtime-version-list
    '2.0': '1.15',
    # TODO(b/157039850) Remove this once CAIP model support TF 2.2 runtime.
    '2.2': '2.1',
    '2.3': '2.1',
    '2.4': '2.1'
}


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
    job_id: the job ID of the AI Platform training job.
    project: the GCP project under which the training job will be executed.
    training_input: Training input argument for AI Platform training job. See
      https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#TrainingInput
      for the detailed schema.
    job_labels: the dict of labels that will be attached to this job.

  Raises:
    RuntimeError: if the Google Cloud AI Platform training job failed/cancelled.
    ConnectionError: if the status polling of the training job failed due to
      connection issue.
  """
  # Configure AI Platform training job
  api_client = discovery.build('ml', 'v1')
  project_id = 'projects/{}'.format(project)
  job_spec = {
      'jobId': job_id,
      'trainingInput': training_input,
      'labels': job_labels,
  }

  # Submit job to AIP Training
  logging.info('TrainingInput=%s', training_input)
  logging.info('Submitting job=\'%s\', project=\'%s\' to AI Platform.', job_id,
               project)
  request = api_client.projects().jobs().create(
      body=job_spec, parent=project_id)
  request.execute()

  # Wait for AIP Training job to finish
  job_name = '{}/jobs/{}'.format(project_id, job_id)
  request = api_client.projects().jobs().get(name=job_name)
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
  while response['state'] not in ('SUCCEEDED', 'FAILED', 'CANCELLED'):
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
        api_client = discovery.build('ml', 'v1')
        request = api_client.projects().jobs().get(name=job_name)
      else:
        logging.error('Request failed after %s retries.',
                      _CONNECTION_ERROR_RETRY_LIMIT)
        raise

  if response['state'] in ('FAILED', 'CANCELLED'):
    err_msg = 'Job \'{}\' did not succeed.  Detailed response {}.'.format(
        job_name, response)
    logging.error(err_msg)
    raise RuntimeError(err_msg)

  # AIP training complete
  logging.info('Job \'%s\' successful.', job_name)


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
  training_inputs = training_inputs.copy()

  json_inputs = artifact_utils.jsonify_artifact_dict(input_dict)
  logging.info('json_inputs=\'%s\'.', json_inputs)
  json_outputs = artifact_utils.jsonify_artifact_dict(output_dict)
  logging.info('json_outputs=\'%s\'.', json_outputs)
  json_exec_properties = json.dumps(exec_properties, sort_keys=True)
  logging.info('json_exec_properties=\'%s\'.', json_exec_properties)

  # We use custom containers to launch training on AI Platform, which invokes
  # the specified image using the container's entrypoint. The default
  # entrypoint for TFX containers is to call scripts/run_executor.py. The
  # arguments below are passed to this run_executor entry to run the executor
  # specified in `executor_class_path`.
  job_args = [
      '--executor_class_path', executor_class_path, '--inputs', json_inputs,
      '--outputs', json_outputs, '--exec-properties', json_exec_properties
  ]

  if not training_inputs.get('masterConfig'):
    training_inputs['masterConfig'] = {
        'imageUri': _TFX_IMAGE,
    }

  training_inputs['args'] = job_args

  # Pop project_id so AIP doesn't complain about an unexpected parameter.
  # It's been a stowaway in aip_args and has finally reached its destination.
  project = training_inputs.pop('project')
  with telemetry_utils.scoped_labels(
      {telemetry_utils.LABEL_TFX_EXECUTOR: executor_class_path}):
    job_labels = telemetry_utils.get_labels_dict()

  # 'tfx_YYYYmmddHHMMSS' is the default job ID if not explicitly specified.
  job_id = job_id or 'tfx_{}'.format(
      datetime.datetime.now().strftime('%Y%m%d%H%M%S'))

  _launch_aip_training(
      job_id=job_id,
      project=project,
      training_input=training_inputs,
      job_labels=job_labels)


def deploy_model_for_aip_prediction(
    serving_path: Text,
    model_version: Text,
    ai_platform_serving_args: Dict[Text, Any],
    executor_class_path: Text,
):
  """Deploys a model for serving with AI Platform.

  Args:
    serving_path: The path to the model. Must be a GCS URI.
    model_version: Version of the model being deployed. Must be different from
      what is currently being served.
    ai_platform_serving_args: Dictionary containing arguments for pushing to AI
      Platform. For the full set of parameters supported, refer to
      https://cloud.google.com/ml-engine/reference/rest/v1/projects.models.versions#Version
    executor_class_path: class path for TFX core default trainer.

  Raises:
    RuntimeError: if an error is encountered when trying to push.
  """
  logging.info(
      'Deploying to model with version %s to AI Platform for serving: %s',
      model_version, ai_platform_serving_args)

  model_name = ai_platform_serving_args['model_name']
  project_id = ai_platform_serving_args['project_id']
  regions = ai_platform_serving_args.get('regions', [])
  default_runtime_version = _get_tf_runtime_version(tf.__version__)
  runtime_version = ai_platform_serving_args.get('runtime_version',
                                                 default_runtime_version)
  python_version = _get_caip_python_version(runtime_version)

  api = discovery.build('ml', 'v1')
  body = {'name': model_name, 'regions': regions}
  parent = 'projects/{}'.format(project_id)
  try:
    api.projects().models().create(body=body, parent=parent).execute()
  except errors.HttpError as e:
    # If the error is to create an already existing model, it's ok to ignore.
    # TODO(b/135211463): Remove the disable once the pytype bug is fixed.
    if e.resp.status == 409:  # pytype: disable=attribute-error
      logging.warn('Model %s already exists', model_name)
    else:
      raise RuntimeError('AI Platform Push failed: {}'.format(e))
  with telemetry_utils.scoped_labels(
      {telemetry_utils.LABEL_TFX_EXECUTOR: executor_class_path}):
    job_labels = telemetry_utils.get_labels_dict()
  body = {
      'name': model_version,
      'deployment_uri': serving_path,
      'runtime_version': runtime_version,
      'python_version': python_version,
      'labels': job_labels,
  }

  # Push to AIP, and record the operation name so we can poll for its state.
  model_name = 'projects/{}/models/{}'.format(project_id, model_name)
  response = api.projects().models().versions().create(
      body=body, parent=model_name).execute()
  op_name = response['name']

  deploy_status_resc = api.projects().operations().get(name=op_name)
  while not deploy_status_resc.execute().get('done'):
    time.sleep(_POLLING_INTERVAL_IN_SECONDS)
    logging.info('Model still being deployed...')

  deploy_status = deploy_status_resc.execute()

  if deploy_status.get('error'):
    # The operation completed with an error.
    raise RuntimeError(
        'Failed to deploy model to AI Platform for serving: {}'.format(
            deploy_status['error']))

  # Set the new version as default.
  # By API specification, if Long-Running-Operation is done and there is
  # no error, 'response' is guaranteed to exist.
  api.projects().models().versions().setDefault(name='{}/versions/{}'.format(
      model_name, deploy_status['response']['name'])).execute()

  logging.info(
      'Successfully deployed model %s with version %s, serving from %s',
      model_name, model_version, serving_path)
