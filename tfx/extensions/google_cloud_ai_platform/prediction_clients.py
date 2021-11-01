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
"""An abstract class for the Pusher for both CAIP and Vertex."""

import abc
import sys
import time
from typing import Any, Dict, Optional, Union

from absl import logging
from google.cloud import aiplatform
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
    # TODO(b/168249383) Update this once CAIP model support TF 2.6 runtime.
    '2.6': '2.5',
}

# Google Cloud AI Platform's ModelVersion resource path format.
# https://cloud.google.com/ai-platform/prediction/docs/reference/rest/v1/projects.models.versions/get
_CAIP_MODEL_VERSION_PATH_FORMAT = (
    'projects/{project_id}/models/{model}/versions/{version}')

_VERTEX_ENDPOINT_SUFFIX = '-aiplatform.googleapis.com'


def _get_tf_runtime_version(tf_version: str) -> str:
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
  """Abstract class interacting with CAIP or Vertex Prediction service."""

  @abc.abstractmethod
  def deploy_model(self,
                   serving_path: str,
                   model_version_name: str,
                   ai_platform_serving_args: Dict[str, Any],
                   labels: Dict[str, str],
                   skip_model_endpoint_creation: Optional[bool] = False,
                   set_default: Optional[bool] = True,
                   **kwargs) -> str:
    """Deploys a model for serving with AI Platform.

    Args:
      serving_path: The path to the model. Must be a GCS URI.
      model_version_name: Model version for CAIP model being deployed, or model
        name for the Vertex model being deployed. Must be different from what is
        currently being served.
      ai_platform_serving_args: Dictionary containing arguments for pushing to
        AI Platform.
      labels: The dict of labels that will be attached to this CAIP job or
        Vertex endpoint. They are merged with optional labels from
        `ai_platform_serving_args`.
      skip_model_endpoint_creation: If true, the method assumes CAIP model or
        Vertex endpoint already exists in AI platform, therefore skipping
        model/endpoint creation.
      set_default: Whether set the newly deployed CAIP model version or Vertex
        model as the default.
      **kwargs: Extra keyword args.

    Returns:
      The resource name of the deployed CAIP model version or Vertex model.

    Raises:
      RuntimeError: if an error is encountered when trying to push.
    """
    pass

  @abc.abstractmethod
  def create_model_for_aip_prediction_if_not_exist(
      self, labels: Dict[str, str],
      ai_platform_serving_args: Dict[str, Any]) -> bool:
    """Creates a new CAIP model or Vertex endpoint for serving with AI Platform if not exists.

    Args:
      labels: The dict of labels that will be attached to this CAIP job or
        Vertex endpoint.
      ai_platform_serving_args: Dictionary containing arguments for pushing to
        AI Platform.

    Returns:
      Whether a new CAIP model or Vertex endpoint is created.

    Raises:
      RuntimeError if CAIP model or Vertex endpoint creation failed.
    """
    pass

  @abc.abstractmethod
  def delete_model_from_aip_if_exists(
      self,
      ai_platform_serving_args: Dict[str, Any],
      model_version_name: str,
      delete_model_endpoint: Optional[bool] = False,
  ) -> None:
    """Deletes a CAIP model and model version or Vertex endpoint and model if exists.

    Args:
      ai_platform_serving_args: Dictionary containing arguments for pushing to
        AI Platform.
      model_version_name: Model version or Vertex endpoint being deleted.
        Required if delete_model_endpoint is False, otherwise not needed.
      delete_model_endpoint: Whether CAIP model or Vertex endpoint should be
        deleted.

    Raises:
      RuntimeError: if an error is encountered when trying to delete.
    """
    pass


class CAIPTfxPredictionClient(AbstractPredictionClient):
  """Class for interacting with CAIP Prediction service."""

  def __init__(self, api: discovery.Resource):
    self._client = api
    super().__init__()

  def deploy_model(self,
                   serving_path: str,
                   model_version_name: str,
                   ai_platform_serving_args: Dict[str, Any],
                   labels: Dict[str, str],
                   skip_model_endpoint_creation: Optional[bool] = False,
                   set_default: Optional[bool] = True,
                   **kwargs) -> str:
    """Deploys a model for serving with AI Platform.

    Args:
      serving_path: The path to the model. Must be a GCS URI.
      model_version_name: Version of the model being deployed. Must be different
        from what is currently being served.
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
      labels: The dict of labels that will be attached to this job. They are
        merged with optional labels from `ai_platform_serving_args`.
      skip_model_endpoint_creation: If true, the method assumes model already
        exists in AI platform, therefore skipping model creation.
      set_default: Whether set the newly deployed model version as the
        default version.
      **kwargs: Extra keyword args.

    Returns:
      The resource name of the deployed model version.

    Raises:
      RuntimeError: if an error is encountered when trying to push.
    """
    logging.info(
        'Deploying to model with version %s to AI Platform for serving: %s',
        model_version_name, ai_platform_serving_args)
    if (sys.version_info.major != 3) and (sys.version_info.minor != 7):
      logging.warn('Current python version is not the same as default of 3.7.')

    model_name = ai_platform_serving_args['model_name']
    project_id = ai_platform_serving_args['project_id']
    default_runtime_version = _get_tf_runtime_version(tf.__version__)
    runtime_version = ai_platform_serving_args.get('runtime_version',
                                                   default_runtime_version)
    python_version = '3.7'

    if not skip_model_endpoint_creation:
      self.create_model_for_aip_prediction_if_not_exist(
          labels, ai_platform_serving_args)
    version_body = dict(ai_platform_serving_args)
    for model_only_key in ['model_name', 'project_id', 'regions']:
      version_body.pop(model_only_key, None)
    version_body['name'] = model_version_name
    version_body['deployment_uri'] = serving_path
    version_body['runtime_version'] = version_body.get('runtime_version',
                                                       runtime_version)
    version_body['python_version'] = version_body.get('python_version',
                                                      python_version)
    version_body['labels'] = {**version_body.get('labels', {}), **labels}
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
        logging.warn('Model version %s already exists', model_version_name)
      else:
        raise RuntimeError('Creating model version to AI Platform failed: {}'
                           .format(e))

    if set_default:
      # Set the new version as default.
      # By API specification, if Long-Running-Operation is done and there is
      # no error, 'response' is guaranteed to exist.
      self._client.projects().models().versions().setDefault(
          name='{}/versions/{}'.format(model_name,
                                       model_version_name)).execute()

    logging.info(
        'Successfully deployed model %s with version %s, serving from %s',
        model_name, model_version_name, serving_path)

    return _CAIP_MODEL_VERSION_PATH_FORMAT.format(
        project_id=project_id, model=model_name, version=model_version_name)

  def create_model_for_aip_prediction_if_not_exist(
      self, labels: Dict[str, str],
      ai_platform_serving_args: Dict[str, Any]) -> bool:
    """Creates a new model for serving with AI Platform if not exists.

    Args:
      labels: The dict of labels that will be attached to this job.
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
    body = {'name': model_name, 'regions': regions, 'labels': labels}
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

  def _wait_for_operation(self, operation: Dict[str, Any],
                          method_name: str) -> Dict[str, Any]:
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

  def delete_model_from_aip_if_exists(
      self,
      ai_platform_serving_args: Dict[str, Any],
      model_version_name: Optional[str] = None,
      delete_model_endpoint: Optional[bool] = False,
  ) -> None:
    """Deletes a model from Google Cloud AI Platform if exists.

    Args:
      ai_platform_serving_args: Dictionary containing arguments for pushing to
        AI Platform. For the full set of parameters supported, refer to
        https://cloud.google.com/ml-engine/reference/rest/v1/projects.models
      model_version_name: Version of the CAIP model being deleted. Required if
        delete_model_endpoint is False, otherwise not needed.
      delete_model_endpoint: If True, deletes the CAIP model, which
        automatically deletes all model versions. If False, deletes only the
        model version with model_version_name.

    Raises:
      RuntimeError: if an error is encountered when trying to delete.
    """
    model_name = ai_platform_serving_args['model_name']
    project_id = ai_platform_serving_args['project_id']

    if delete_model_endpoint:
      logging.info('Deleting model with from AI Platform: %s',
                   ai_platform_serving_args)
      name = 'projects/{}/models/{}'.format(project_id, model_name)
      try:
        operation = self._client.projects().models().delete(name=name).execute()
        self._wait_for_operation(operation, 'projects.models.delete')
      except errors.HttpError as e:
        # If the error is to delete an non-existent model, it's ok to ignore.
        if e.resp.status == 404:
          logging.warn('Model %s does not exist', model_name)
        else:
          raise RuntimeError(
              'Deleting model from AI Platform failed.') from e
    else:
      if model_version_name is None:
        raise ValueError('model_version_name is required if'
                         ' delete_model_endpoint is False')
      logging.info('Deleting model version %s from AI Platform: %s',
                   model_version_name, ai_platform_serving_args)
      version_name = 'projects/{}/models/{}/versions/{}'.format(
          project_id, model_name, model_version_name)
      try:
        operation = self._client.projects().models().versions().delete(
            name=version_name).execute()
        self._wait_for_operation(
            operation, 'projects.models.versions.delete')
      except errors.HttpError as e:
        # If the error is to delete an non-existent model version,
        # it's ok to ignore.
        if e.resp.status == 404:
          logging.warn('Model version %s does not exist', version_name)
        if e.resp.status == 400:
          logging.warn('Model version %s won\'t be deleted because it is the '
                       'default version and not the only version in the model',
                       version_name)
        else:
          raise RuntimeError(
              'Deleting model version {} from AI Platform failed.'.format(
                  version_name)) from e


class VertexPredictionClient(AbstractPredictionClient):
  """Class for interacting with Vertex Prediction service."""

  def deploy_model(self,
                   serving_path: str,
                   model_version_name: str,
                   ai_platform_serving_args: Dict[str, Any],
                   labels: Dict[str, str],
                   serving_container_image_uri: str,
                   endpoint_region: str,
                   skip_model_endpoint_creation: Optional[bool] = False,
                   set_default: Optional[bool] = True,
                   **kwargs) -> str:
    """Deploys a model for serving with AI Platform.

    Args:
      serving_path: The path to the model. Must be a GCS URI. Required for model
        creation. If not specified, it is assumed that model with model_name
        exists in AIP.
      model_version_name: Name of the Vertex model being deployed. Must be
        different from what is currently being served, if there is an existing
        model at the specified endpoint with endpoint_name.
      ai_platform_serving_args: Dictionary containing arguments for pushing to
        AI Platform. The full set of parameters supported can be found at
        https://googleapis.dev/python/aiplatform/latest/aiplatform.html?highlight=deploy#google.cloud.aiplatform.Model.deploy.
        Most keys are forwarded as-is, but following keys are handled specially:
          - endpoint_name: Name of the endpoint.
          - traffic_percentage: Desired traffic to newly deployed model.
            Forwarded as-is if specified. If not specified, it is set to 100 if
            set_default_version is True, or set to 0 otherwise.
          - labels: a list of job labels will be merged with user's input.
      labels: The dict of labels that will be attached to this endpoint. They
        are merged with optional labels from `ai_platform_serving_args`.
      serving_container_image_uri: The path to the serving container image URI.
        Container registry for prediction is available at:
        https://gcr.io/cloud-aiplatform/prediction.
      endpoint_region: Region for Vertex Endpoint. For available regions, please
        see https://cloud.google.com/vertex-ai/docs/general/locations
      skip_model_endpoint_creation: If true, the method assumes endpoint already
        exists in AI platform, therefore skipping endpoint creation.
      set_default: Whether set the newly deployed model as the default (i.e.
        100% traffic).
      **kwargs: Extra keyword args.

    Returns:
      The resource name of the deployed model.
    """
    logging.info(
        'Deploying to model to AI Platform for serving: %s',
        ai_platform_serving_args)
    if sys.version_info[:2] != (3, 7):
      logging.warn('Current python version is not the same as default of 3.7.')

    if ai_platform_serving_args.get('project_id'):
      assert 'project' not in ai_platform_serving_args, ('`project` and '
                                                         '`project_id` should '
                                                         'not be set at the '
                                                         'same time in serving '
                                                         'args')
      logging.warn('Replacing `project_id` with `project` in serving args.')
      ai_platform_serving_args['project'] = ai_platform_serving_args[
          'project_id']
      ai_platform_serving_args.pop('project_id')
    project = ai_platform_serving_args['project']

    # Initialize the AI Platform client
    # location defaults to 'us-central-1' if not specified
    aiplatform.init(project=project, location=endpoint_region)

    endpoint_name = ai_platform_serving_args['endpoint_name']
    if not skip_model_endpoint_creation:
      self.create_model_for_aip_prediction_if_not_exist(
          labels, ai_platform_serving_args)
    endpoint = self._get_endpoint(ai_platform_serving_args)

    deploy_body = dict(ai_platform_serving_args)
    for unneeded_key in ['endpoint_name', 'project', 'regions', 'labels']:
      deploy_body.pop(unneeded_key, None)
    deploy_body['traffic_percentage'] = deploy_body.get(
        'traffic_percentage', 100 if set_default else 0)
    logging.info(
        'Creating model_name %s in project %s at endpoint %s, request body: %s',
        model_version_name, project, endpoint_name, deploy_body)

    model = aiplatform.Model.upload(
        display_name=model_version_name,
        artifact_uri=serving_path,
        serving_container_image_uri=serving_container_image_uri)
    model.wait()

    try:
      # Push to AI Platform and wait for deployment to be complete.
      model.deploy(endpoint=endpoint, **deploy_body)
      model.wait()
    except errors.HttpError as e:
      # If the error is to create an already existing model, it's ok to
      # ignore.
      if e.resp.status == 409:
        logging.warn('Model %s already exists at endpoint %s',
                     model_version_name, endpoint_name)
      else:
        raise RuntimeError(
            'Creating model version to AI Platform failed.') from e

    logging.info(
        'Successfully deployed model %s to endpoint %s, serving from %s',
        model_version_name, endpoint_name, endpoint.resource_name)

    return model.resource_name

  def create_model_for_aip_prediction_if_not_exist(
      self, labels: Dict[str, str],
      ai_platform_serving_args: Dict[str, Any]) -> bool:
    """Creates a new endpoint for serving with AI Platform if not exists.

    Args:
      labels: The dict of labels that will be attached to this Vertex endpoint.
      ai_platform_serving_args: Dictionary containing arguments for pushing to
        AI Platform.

    Returns:
      The endpoint if it's created, otherwise None.

    Raises:
      RuntimeError if endpoint creation failed.
    """
    endpoint_name = ai_platform_serving_args['endpoint_name']
    endpoint_labels = {**ai_platform_serving_args.get('labels', {}),
                       **labels}
    endpoint = None
    try:
      endpoint = aiplatform.Endpoint.create(
          display_name=endpoint_name, labels=endpoint_labels)
    except errors.HttpError as e:
      # If the error is to create an already existing endpoint,
      # it's ok to ignore.
      if e.resp.status == 409:
        logging.warn('Endpoint %s already exists', endpoint_name)
      else:
        raise RuntimeError(
            'Creating endpoint in AI Platform failed.') from e
    return endpoint is not None

  def delete_model_from_aip_if_exists(
      self,
      ai_platform_serving_args: Dict[str, Any],
      model_version_name: Optional[str] = None,
      delete_model_endpoint: Optional[bool] = False,
  ) -> None:
    """Deletes a model from Google Cloud AI Platform if model exists.

    Args:
      ai_platform_serving_args: Dictionary containing arguments for pushing to
        AI Platform. For the full set of parameters supported, refer to
        https://googleapis.dev/python/aiplatform/latest/aiplatform.html?highlight=deploy#google.cloud.aiplatform.Model.deploy.
      model_version_name: Name of the Vertex model being deleted. Required if
        delete_model_endpoint is False, otherwise not needed.
      delete_model_endpoint: If True, deletes the Vertex endpoint, which
        automatically deletes all models at the endpoint. If False, deletes only
        the model with model_version_name at the endpoint.

    Raises:
      RuntimeError: if an error is encountered when trying to delete.
    """
    endpoint = self._get_endpoint(ai_platform_serving_args)

    if endpoint is None:
      return

    if delete_model_endpoint:
      logging.info('Deleting endpoint with from AI Platform: %s',
                   ai_platform_serving_args)
      try:
        endpoint.delete(force=True, sync=True)
      except errors.HttpError as e:
        if e.resp.status == 404:
          logging.warn('Endpoint %s does not exist',
                       ai_platform_serving_args['endpoint_name'])
        else:
          raise RuntimeError(
              'Deleting endpoint from AI Platform failed.') from e
    else:
      if model_version_name is None:
        raise ValueError('model_version_name is required if'
                         ' delete_model_endpoint is False')
      logging.info('Deleting model %s from AI Platform: %s',
                   model_version_name, ai_platform_serving_args)
      deployed_models = endpoint.list_models()
      models = [
          model for model in deployed_models
          if model.display_name == model_version_name
      ]
      if models:
        model_to_undeploy = models[0]
      else:
        logging.warn('Model %s does not exist at endpoint %s',
                     model_version_name,
                     ai_platform_serving_args['endpoint_name'])
        return
      deployed_model_id = model_to_undeploy.id
      try:
        endpoint.undeploy(deployed_model_id=deployed_model_id, sync=True)
      except errors.HttpError as e:
        # If the error is to delete an non-existent model version,
        # it's ok to ignore.
        if e.resp.status == 404:
          logging.warn('Model %s does not exist', model_version_name)
        if e.resp.status == 400:
          logging.warn('Model %s won\'t be deleted because it is the '
                       'default version and not the only version in the model',
                       model_version_name)
        else:
          raise RuntimeError(
              'Deleting model {} from AI Platform failed.'.format(
                  model_version_name)) from e

  def _get_endpoint(
      self, ai_platform_serving_args: Dict[str, Any]) -> aiplatform.Endpoint:
    """Gets an endpoint from Google Cloud AI Platform if endpoint exists.

    Args:
      ai_platform_serving_args: Dictionary containing arguments for pushing to
        AI Platform. For the full set of parameters supported, refer to
        https://googleapis.dev/python/aiplatform/latest/aiplatform.html?highlight=deploy#google.cloud.aiplatform.Model.deploy.

    Raises:
      RuntimeError: if an error is encountered when trying to get the endpoint

    Returns:
      The endpoint
    """
    endpoint_name = ai_platform_serving_args['endpoint_name']
    endpoint = None

    endpoints = aiplatform.Endpoint.list(filter='display_name="{}"'.format(
        endpoint_name))

    if endpoints:
      endpoint = endpoints[0]
    else:
      raise RuntimeError('Error getting endpoint {}'.format(endpoint_name))

    return endpoint


def get_prediction_client(
    api: Optional[discovery.Resource] = None,
    enable_vertex: Optional[bool] = False
) -> Union[CAIPTfxPredictionClient, VertexPredictionClient]:
  """Gets the job client.

  Args:
    api: Google API client resource.
    enable_vertex: Whether to enable Vertex

  Returns:
    The corresponding prediction client.
  """
  if enable_vertex:
    return VertexPredictionClient()
  return CAIPTfxPredictionClient(api)
