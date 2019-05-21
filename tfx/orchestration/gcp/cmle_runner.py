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
"""Helper class to start TFX training jobs on CMLE."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import json
import os
import sys
import time
from googleapiclient import discovery
from googleapiclient import errors
import tensorflow as tf
from typing import Any, Dict, List, Text

from tfx.utils import deps_utils
from tfx.utils import io_utils
from tfx.utils import types

_POLLING_INTERVAL_IN_SECONDS = 30


def _get_tf_runtime_version() -> Text:
  # runtimeVersion should be same as <major>.<minor> of currently
  # installed tensorflow version.
  return '.'.join(tf.__version__.split('.')[0:2])


def _get_python_version() -> Text:
  """Return <major>.<minor> version for current Python interpreter."""
  return '%d.%d' % (sys.version_info.major, sys.version_info.minor)


def start_cmle_training(input_dict: Dict[Text, List[types.TfxArtifact]],
                        output_dict: Dict[Text, List[types.TfxArtifact]],
                        exec_properties: Dict[Text, Any],
                        executor_class_path: Text,
                        training_inputs: Dict[Text, Any]):
  """Start a trainer job on CMLE.

  This is done by forwarding the inputs/outputs/exec_properties to the
  tfx.scripts.run_executor module on a CMLE training job interpreter.

  Args:
    input_dict: Passthrough input dict for tfx.components.Trainer.executor.
    output_dict: Passthrough input dict for tfx.components.Trainer.executor.
    exec_properties: Passthrough input dict for tfx.components.Trainer.executor.
    executor_class_path: class path for TFX core default trainer.
    training_inputs: Training input for CMLE training job. 'pythonModule',
      'pythonVersion' and 'runtimeVersion' will be inferred by the runner. For
      the full set of parameters supported, refer to
        https://cloud.google.com/ml-engine/docs/tensorflow/deploying-models#creating_a_model_version.

  Returns:
    None
  Raises:
    RuntimeError: if the Google Cloud AI Platform training job failed.
  """
  training_inputs = training_inputs.copy()
  # Remove cmle_args from exec_properties so CMLE trainer doesn't call itself
  for gaip_training_key in ['cmle_training_args', 'gaip_training_args']:
    if gaip_training_key in exec_properties.get('custom_config'):
      exec_properties['custom_config'].pop(gaip_training_key)

  json_inputs = types.jsonify_tfx_type_dict(input_dict)
  tf.logging.info('json_inputs=\'%s\'.', json_inputs)
  json_outputs = types.jsonify_tfx_type_dict(output_dict)
  tf.logging.info('json_outputs=\'%s\'.', json_outputs)
  json_exec_properties = json.dumps(exec_properties)
  tf.logging.info('json_exec_properties=\'%s\'.', json_exec_properties)

  # Configure CMLE job
  api_client = discovery.build('ml', 'v1')
  job_args = [
      '--executor_class_path', executor_class_path, '--inputs', json_inputs,
      '--outputs', json_outputs, '--exec-properties', json_exec_properties
  ]
  training_inputs['args'] = job_args
  training_inputs['pythonModule'] = 'tfx.scripts.run_executor'
  training_inputs['pythonVersion'] = _get_python_version()
  # runtimeVersion should be same as <major>.<minor> of currently
  # installed tensorflow version.
  training_inputs['runtimeVersion'] = _get_tf_runtime_version()

  # Pop project_id so CMLE doesn't complain about an unexpected parameter.
  # It's been a stowaway in cmle_args and has finally reached its destination.
  project = training_inputs.pop('project')
  project_id = 'projects/{}'.format(project)

  package_uris = training_inputs.get('packageUris', [])
  if package_uris:
    tf.logging.info('Following packageUris \'%s\' are provided by user.',
                    package_uris)
  else:
    local_package = deps_utils.build_ephemeral_package()
    # TODO(b/125451545): Use a safe temp dir instead of jobDir.
    cloud_package = os.path.join(training_inputs['jobDir'],
                                 os.path.basename(local_package))
    io_utils.copy_file(local_package, cloud_package, True)
    training_inputs['packageUris'] = [cloud_package]
    tf.logging.info('Package %s will be used', training_inputs['packageUris'])

  job_name = 'tfx_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
  job_spec = {'jobId': job_name, 'trainingInput': training_inputs}

  # Submit job to CMLE
  tf.logging.info('Submitting job=\'{}\', project=\'{}\' to CMLE.'.format(
      job_name, project))
  request = api_client.projects().jobs().create(
      body=job_spec, parent=project_id)
  request.execute()

  # Wait for CMLE job to finish
  job_id = '{}/jobs/{}'.format(project_id, job_name)
  request = api_client.projects().jobs().get(name=job_id)
  response = request.execute()
  while response['state'] not in ('SUCCEEDED', 'FAILED'):
    time.sleep(_POLLING_INTERVAL_IN_SECONDS)
    response = request.execute()

  if response['state'] == 'FAILED':
    err_msg = 'Job \'{}\' did not succeed.  Detailed response {}.'.format(
        job_name, response)
    tf.logging.error(err_msg)
    raise RuntimeError(err_msg)

  # CMLE training complete
  tf.logging.info('Job \'{}\' successful.'.format(job_name))


def deploy_model_for_cmle_serving(serving_path: Text, model_version: Text,
                                  cmle_serving_args: Dict[Text, Any]):
  """Deploys a model for serving with CMLE.

  Args:
    serving_path: The path to the model. Must be a GCS URI.
    model_version: Version of the model being deployed. Must be different
      from what is currently being served.
    cmle_serving_args: Dictionary containing arguments for pushing to CMLE.
      For the full set of parameters supported, refer to
      https://cloud.google.com/ml-engine/reference/rest/v1/projects.models.versions#Version

  Raises:
    RuntimeError: if an error is encountered when trying to push.
  """
  tf.logging.info(
      'Deploying to model with version {} to CMLE for serving: {}'.format(
          model_version, cmle_serving_args))

  model_name = cmle_serving_args['model_name']
  project_id = cmle_serving_args['project_id']
  runtime_version = _get_tf_runtime_version()
  python_version = _get_python_version()

  api = discovery.build('ml', 'v1')
  body = {'name': model_name}
  parent = 'projects/{}'.format(project_id)
  try:
    api.projects().models().create(body=body, parent=parent).execute()
  except errors.HttpError as e:
    # If the error is to create an already existing model, it's ok to ignore.
    if e.resp.status == 409:
      tf.logging.warn('Model {} already exists'.format(model_name))
    else:
      raise RuntimeError('CMLE Push failed: {}'.format(e))

  body = {
      'name': 'v{}'.format(model_version),
      'deployment_uri': serving_path,
      'runtime_version': runtime_version,
      'python_version': python_version,
  }

  # Push to CMLE, and record the operation name so we can poll for its state.
  model_name = 'projects/{}/models/{}'.format(project_id, model_name)
  response = api.projects().models().versions().create(
      body=body, parent=model_name).execute()
  op_name = response['name']

  while True:
    deploy_status = api.projects().operations().get(name=op_name).execute()
    if deploy_status.get('done'):
      break
    if 'error' in deploy_status:
      # The operation completed with an error.
      tf.logging.error(deploy_status['error'])
      raise RuntimeError(
          'Failed to deploy model to CMLE for serving: {}'.format(
              deploy_status['error']))

    time.sleep(_POLLING_INTERVAL_IN_SECONDS)
    tf.logging.info('Model still being deployed...')

  tf.logging.info(
      'Successfully deployed model {} with version {}, serving from {}'.format(
          model_name, model_version, serving_path))
