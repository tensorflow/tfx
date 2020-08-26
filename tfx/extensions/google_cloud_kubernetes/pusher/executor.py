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
"""Custom executor to serve on Kubernetes."""

from absl import logging
from os import path
import json
from typing import Any, Dict, List, Text
import yaml

from google.protobuf import text_format
from kubernetes.client import rest
from tensorflow_serving.config import model_server_config_pb2
from tensorflow_serving.sources.storage_path import file_system_storage_path_source_pb2
from tfx import types
from tfx.components.pusher import executor as tfx_pusher_executor
from tfx.types import artifact_utils
from tfx.utils import json_utils
from tfx.utils import path_utils
from tfx.utils import kube_utils


# Keys to the items in custom_config passed as a part of exec_properties.
TF_SERVING_ARGS_KEY = 'tf_serving_args'

# Keys for custom_config.
_CUSTOM_CONFIG_KEY = 'custom_config'

# Number of serving replicas.
NUM_REPLICAS_KEY = 'num_replicas'

# Environment variable key to pass in server config.
_SERVER_CONFIG_ENV_KEY = 'TF_SERVING_CONFIG'


def _create_model_service_configuration(
    model_name: Text,
    model_uri: Text,
    model_version: int,
) -> Text:
  """Helper function to create a serialized configuration for serving."""
  # Create the base model config Protobuf definition.
  model_server_config = model_server_config_pb2.ModelServerConfig()
  config_list = model_server_config_pb2.ModelConfigList()       
  one_config = config_list.config.add()
  one_config.name = model_name
  one_config.base_path = model_uri
  one_config.model_platform = 'tensorflow'

  # Create the model version policy Protobuf definition.
  path_config = (file_system_storage_path_source_pb2.
      FileSystemStoragePathSourceConfig)
  version_policy = path_config.ServableVersionPolicy(
    specific=path_config.ServableVersionPolicy.Specific(
      versions = [model_version]))

  one_config.model_version_policy.CopyFrom(version_policy)
  model_server_config.model_config_list.CopyFrom(config_list)
  return text_format.MessageToString(model_server_config)


class Executor(tfx_pusher_executor.Executor):
  """Deploy a model to GKE cluster with Tensorflow Serving."""

  def DeployTFServingDeployment(
      self,
      model_name: Text,
      model_uri: Text,
      model_version: Text,
      num_replicas: int,
  ) -> None:
    """Creates or updates the model serving deployment with TF Serving.

    Args:
      model_name: Name of the model being served, used as part of the model
        path in the serving container as well as the exposed api endpoint
        to the client.
      model_uri: Uri of the serving model output from which the container
        will download the model.
      model_version: Version of the model being served.
      num_replicas: Number of serving replicas.
  """
    client_api = kube_utils.make_extensions_v1_beta1_api()

    with open(path.join(path.dirname(__file__),
                        'yaml',
                        'serving-deployment.yaml')) as f:
      deployment = yaml.safe_load(f)
    # Pass through model server configuration protobuf string as an
    # environment variable.
    env_vars = [{'name': _SERVER_CONFIG_ENV_KEY,
                 'value': _create_model_service_configuration(
                     model_name, model_uri, model_version)}]
    spec = deployment['spec']
    # Configure the number of replicas and environment variables.
    spec['replicas'] = num_replicas
    spec['template']['spec']['containers'][0]['env'] = env_vars
    try:
      response = client_api.create_namespaced_deployment(
          body=deployment, namespace='default')
      logging.info('Deployment created. status="%s"' % str(response.status))
    except rest.ApiException:
      logging.info(deployment)
      response = client_api.patch_namespaced_deployment(
          name=deployment['metadata']['name'],
          body=deployment, namespace='default')
      logging.info('Deployment updated. status="%s"' % str(response.status))

  def DeployTFServingService(self) -> None:
    """Creates the model serving service with TF Serving."""
    client_api = kube_utils.make_core_v1_api()
    with open(path.join(path.dirname(__file__),
                        'yaml', 'serving-service.yaml')) as f:
      service = yaml.safe_load(f)
    try:
      response = client_api.create_namespaced_service(
          body=service, namespace='default')
      logging.info('Model Service created. status="%s"' % str(response.status))
    except rest.ApiException:
      # Since the model service yaml is static, no update is needed.
      logging.info('Model Service unchanged.')

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """Overrides the tfx_pusher_executor.

    Args:
      input_dict: Input dict from input key to a list of artifacts, including:
        - model_export: Exported model from trainer.
        - model_blessing: Model blessing path from evaluator.
      output_dict: Output dict from key to a list of artifacts, including:
        - model_push: A list of 'ModelPushPath' artifact of size one. It will
          include the model in this push execution if the model was pushed.
      exec_properties: Mostly a passthrough input dict for
        tfx.components.Pusher.executor. custom_config.tf_serving_args
        is consumed by this class.

    Raises:
      ValueError:
        If Serving model path does not start with gs://.
      kubernetes.client.rest.ApiException:
        if deployment to GKE cluster failed.
    """
    super().Do(input_dict=input_dict, output_dict=output_dict,
               exec_properties=exec_properties)
    model_push = artifact_utils.get_single_instance(
        output_dict[tfx_pusher_executor.PUSHED_MODEL_KEY])
    if model_push.get_int_custom_property('pushed') == 0:
      return

    custom_config = json_utils.loads(
        exec_properties.get(_CUSTOM_CONFIG_KEY, '{}'))
    if custom_config is not None and not isinstance(custom_config, dict):
      raise ValueError('custom_config in execution properties needs to be a '
                       'dict.')

    tf_serving_args = custom_config.get(TF_SERVING_ARGS_KEY, {})
    if not tf_serving_args:
      logging.info('\'tf_serving_args\' is missing in \'custom_config\'')

    # Parse the number of replicas and model name used as part of serving path.
    if not tf_serving_args.get(NUM_REPLICAS_KEY):
      logging.info('\'%s\' not specified in \'tf_serving_args\', using 1',
                   NUM_REPLICAS_KEY)
    num_replicas = tf_serving_args.get(NUM_REPLICAS_KEY, 1)

    # Assume the following Tensorflow Serving model path format:
    # /base_path/{model_name}/{model_version}
    # See: https://www.tensorflow.org/tfx/serving/serving_kubernetes
    model_path = model_push.get_string_custom_property(
        tfx_pusher_executor._PUSHED_DESTINATION_KEY)
    split_path = model_path.split(path.sep)
    model_base_path = path.sep.join(split_path[:-1])
    model_name = split_path[-2]
    model_version = int(model_push.get_string_custom_property(
        tfx_pusher_executor._PUSHED_VERSION_KEY))

    # Deploy the service and pods.
    self.DeployTFServingDeployment(
        model_name=model_name,
        model_uri=model_base_path,
        model_version=model_version,
        num_replicas=num_replicas,
    )
    self.DeployTFServingService()

    logging.info('Push to Kubernetes successful.')
