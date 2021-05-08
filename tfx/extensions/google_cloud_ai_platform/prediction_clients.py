# Copyright 2021 Google LLC. All Rights Reserved.
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
"""An abstract class for the Pusher for both CAIP and uCAIP."""

import abc
import sys
import time
from typing import Any, Dict, Optional, Text

from absl import logging
from googleapiclient import discovery
from googleapiclient import errors
import tensorflow as tf

_POLLING_INTERVAL_IN_SECONDS = 30

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


class AbstractPredictionClient(abc.ABC):
  """Abstract class interacting with CAIP or uCAIP Prediction service."""

  @abc.abstractmethod
  def deploy_model(self,
                   serving_path: Text,
                   model_version: Text,
                   ai_platform_serving_args: Dict[Text, Any],
                   job_labels: Dict[Text, Text],
                   skip_model_creation: Optional[bool] = False,
                   set_default_version: Optional[bool] = True) -> None:
    """Deploys a model for serving with AI Platform.

    Args:
      serving_path: The path to the model. Must be a GCS URI.
      model_version: Version of the model being deployed. Must be different from
        what is currently being served.
      ai_platform_serving_args: Dictionary containing arguments for pushing to
        AI Platform.
      job_labels: The dict of labels that will be attached to this job. They are
        merged with optional labels from `ai_platform_serving_args`.
      skip_model_creation: If true, the method assumes model already exist in
        AI platform, therefore skipping model creation.
      set_default_version: Whether set the newly deployed model version as the
        default version.

    Raises:
      RuntimeError: if an error is encountered when trying to push.
    """
    pass


class CAIPTfxPredictionClient(AbstractPredictionClient):
  """Class for interacting with CAIP Pusher."""

  def __init__(self, api: Optional[discovery.Resource] = None):
    if api is None:
      raise ValueError('Google API client resource required.')
    self._client = api

  def deploy_model(self,
                   serving_path: Text,
                   model_version: Text,
                   ai_platform_serving_args: Dict[Text, Any],
                   job_labels: Dict[Text, Text],
                   skip_model_creation: Optional[bool] = False,
                   set_default_version: Optional[bool] = True
                   ) -> None:
    """Deploys a model for serving with AI Platform.

    Args:
      serving_path: The path to the model. Must be a GCS URI.
      model_version: Version of the model being deployed. Must be different from
        what is currently being served.
      ai_platform_serving_args: Dictionary containing arguments for pushing to
        AI Platform. The full set of parameters supported can be found at
        https://cloud.google.com/ml-engine/reference/rest/v1/projects.models.versions#Version.
        Most keys are forwarded as-is, but following keys are handled specially:
          - name: this must be empty (and will be filled by pusher).
          - deployment_uri: this must be empty (and will be filled by pusher).
          - python_version: when left empty, this will be filled by python
              version of the environment being used.
          - runtime_version: when left empty, this will be filled by TensorFlow
              version from the environment.
          - labels: a list of job labels will be merged with user's input.
      job_labels: The dict of labels that will be attached to this job. They are
        merged with optional labels from `ai_platform_serving_args`.
      skip_model_creation: If true, the method assumes model already exist in
        AI platform, therefore skipping model creation.
      set_default_version: Whether set the newly deployed model version as the
        default version.

    Raises:
      RuntimeError: if an error is encountered when trying to push.
    """
    logging.info(
        'Deploying to model with version %s to AI Platform for serving: %s',
        model_version, ai_platform_serving_args)
    if (sys.version_info.major != 3) and (sys.version_info.minor != 7):
      logging.warn('Current python version is not the same as default of 3.7.')

    model_name = ai_platform_serving_args['model_name']
    project_id = ai_platform_serving_args['project_id']
    default_runtime_version = _get_tf_runtime_version(tf.__version__)
    runtime_version = ai_platform_serving_args.get('runtime_version',
                                                   default_runtime_version)
    python_version = '3.7'

    if not skip_model_creation:
      self.create_model_for_aip_prediction_if_not_exist(
          job_labels, ai_platform_serving_args)
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
      operation = self._client.projects().models().versions().create(
          body=version_body, parent=model_name).execute()
      self._wait_for_operation(
          operation, 'projects.models.versions.create')
    except errors.HttpError as e:
      # If the error is to create an already existing model version, it's ok to
      # ignore.
      if e.resp.status == 409:
        logging.warn('Model version %s already exists', model_version)
      else:
        raise RuntimeError('Creating model version to AI Platform failed: {}'
                           .format(e))

    if set_default_version:
      # Set the new version as default.
      # By API specification, if Long-Running-Operation is done and there is
      # no error, 'response' is guaranteed to exist.
      self._client.projects().models().versions().setDefault(
          name='{}/versions/{}'.format(model_name, model_version)).execute()

    logging.info(
        'Successfully deployed model %s with version %s, serving from %s',
        model_name, model_version, serving_path)

  def create_model_for_aip_prediction_if_not_exist(
      self,
      job_labels: Dict[Text, Text],
      ai_platform_serving_args: Dict[Text, Any]
  ) -> bool:
    """Creates a new model for serving with AI Platform if not exists.

    Args:
      job_labels: The dict of labels that will be attached to this job.
      ai_platform_serving_args: Dictionary containing arguments for pushing to
        AI Platform.

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
      self._client.projects().models().create(
          body=body, parent=parent).execute()
    except errors.HttpError as e:
      # If the error is to create an already existing model, it's ok to ignore.
      if e.resp.status == 409:
        logging.warn('Model %s already exists', model_name)
        result = False
      else:
        raise RuntimeError('Creating model to AI Platform failed: {}'.format(e))
    return result

  def _wait_for_operation(self,
                          operation: Dict[Text, Any],
                          method_name: Text
                          ) -> Dict[Text, Any]:
    """Wait for a long running operation.

    Args:
      operation: The operation to wait for.
      method_name: Operation method name for logging.

    Returns:
      Operation completion status.

    Raises:
      RuntimeError: If the operation completed with an error.
    """
    status_resc = self._client.projects().operations().get(
        name=operation['name'])
    while not status_resc.execute().get('done'):
      time.sleep(_POLLING_INTERVAL_IN_SECONDS)
      logging.info('Method %s still being executed...', method_name)
    result = status_resc.execute()
    if result.get('error'):
      # The operation completed with an error.
      raise RuntimeError('Failed to execute {}: {}'.format(
          method_name, result['error']))
    return result

  def delete_model_version_from_aip_if_exists(
      self,
      model_version: Text,
      ai_platform_serving_args: Dict[Text, Any]
  ) -> None:
    """Deletes a model version from Google Cloud AI Platform if version exists.

    Args:
      model_version: Version of the model being deleted.
      ai_platform_serving_args: Dictionary containing arguments for pushing to
        AI Platform. For the full set of parameters supported, refer to
        https://cloud.google.com/ml-engine/reference/rest/v1/projects.models

    Raises:
      RuntimeError: if an error is encountered when trying to delete.
    """
    logging.info('Deleting model version %s from AI Platform: %s',
                 model_version, ai_platform_serving_args)
    model_name = ai_platform_serving_args['model_name']
    project_id = ai_platform_serving_args['project_id']
    version_name = 'projects/{}/models/{}/versions/{}'.format(
        project_id, model_name, model_version)
    try:
      operation = self._client.projects().models().versions().delete(
          name=version_name).execute()
      self._wait_for_operation(
          operation, 'projects.models.versions.delete')
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
      self,
      ai_platform_serving_args: Dict[Text, Any]
  ) -> None:
    """Deletes a model from Google Cloud AI Platform if exists.

    Args:
      ai_platform_serving_args: Dictionary containing arguments for pushing to
        AI Platform. For the full set of parameters supported, refer to
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
      operation = self._client.projects().models().delete(name=name).execute()
      self._wait_for_operation(operation, 'projects.models.delete')
    except errors.HttpError as e:
      # If the error is to delete an non-exist model, it's ok to ignore.
      if e.resp.status == 404:
        logging.warn('Model %s does not exist', model_name)
      else:
        raise RuntimeError(
            'Deleting model from AI Platform failed: {}'.format(e))


def get_prediction_client(
    api: Optional[discovery.Resource] = None,
    enable_ucaip: Optional[bool] = False) -> CAIPTfxPredictionClient:
  """Gets the job client.

  Args:
    api: Google API client resource.
    enable_ucaip: Whether to enable uCAIP

  Returns:
    The corresponding prediction client.
  """
  if enable_ucaip:
    raise NotImplementedError('uCAIP Prediction support not yet implemented')
  return CAIPTfxPredictionClient(api)
