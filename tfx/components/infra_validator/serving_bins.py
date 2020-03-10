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
"""Modules for organizing various model server binaries."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from typing import Any, Dict, List, Optional, Text

from docker import types as docker_types
import six

from tfx.components.infra_validator.model_server_clients import base_client
from tfx.components.infra_validator.model_server_clients import tensorflow_serving_client
from tfx.proto import infra_validator_pb2


def parse_serving_binaries(  # pylint: disable=invalid-name
    serving_spec: infra_validator_pb2.ServingSpec) -> List['ServingBinary']:
  """Parse `ServingBinary`s from `ServingSpec`."""
  result = []
  serving_binary = serving_spec.WhichOneof('serving_binary')
  if serving_binary == 'tensorflow_serving':
    config = serving_spec.tensorflow_serving
    for tag in config.tags:
      result.append(TensorFlowServing(model_name=serving_spec.model_name,
                                      tag=tag))
    for digest in config.digests:
      result.append(TensorFlowServing(model_name=serving_spec.model_name,
                                      digest=digest))
    return result
  else:
    raise ValueError('Invalid serving_binary {}'.format(serving_binary))


class ServingBinary(six.with_metaclass(abc.ABCMeta, object)):
  """Base class for serving binaries."""

  @abc.abstractproperty
  def container_port(self) -> int:
    """Container port of the model server.

    Only applies to docker compatible serving binaries.
    """
    raise NotImplementedError('{} is not docker compatible.'.format(
        type(self).__name__))

  @abc.abstractproperty
  def image(self) -> Text:
    """Container image of the model server.

    Only applies to docker compatible serving binaries.
    """
    raise NotImplementedError('{} is not docker compatible.'.format(
        type(self).__name__))

  @abc.abstractmethod
  def MakeEnvVars(self, *args: Any) -> Dict[Text, Text]:
    """Construct environment variables to be used in container image.

    Only applies to docker compatible serving binaries.

    Args:
      *args: List of unresolved variables to configure environment variables.

    Returns:
      A dictionary of environment variables inside container.
    """
    raise NotImplementedError('{} is not docker compatible.'.format(
        type(self).__name__))

  @abc.abstractmethod
  def MakeDockerRunParams(self, *args: Any) -> Dict[Text, Text]:
    """Make parameters for docker `client.containers.run`.

    Only applies to docker compatible serving binaries.

    Args:
      *args: List of unresolved variables to configure docker run parameters.

    Returns:
      A dictionary of docker run parameters.
    """
    raise NotImplementedError('{} is not docker compatible.'.format(
        type(self).__name__))

  @abc.abstractmethod
  def MakeClient(self, endpoint: Text) -> base_client.BaseModelServerClient:
    """Create a model server client of this serving binary."""
    raise NotImplementedError('{} does not implement MakeClient.'.format(
        type(self).__name__))


class TensorFlowServing(ServingBinary):
  """TensorFlow Serving binary."""

  _base_docker_run_args = {
      # Enable auto-removal of the container on docker daemon after container
      # process exits.
      'auto_remove': True,
      # Run container in the background instead of streaming its output.
      'detach': True,
  }
  _DEFAULT_IMAGE_NAME = 'tensorflow/serving'
  _DEFAULT_GRPC_PORT = 8500
  _DEFAULT_MODEL_BASE_PATH = '/model'

  def __init__(
      self,
      model_name: Text,
      tag: Optional[Text] = None,
      digest: Optional[Text] = None,
  ):
    super(TensorFlowServing, self).__init__()
    self._model_name = model_name
    if (tag is None) == (digest is None):
      raise ValueError('Exactly one of `tag` or `digest` should be used.')
    if tag is not None:
      self._image = '{}:{}'.format(self._DEFAULT_IMAGE_NAME, tag)
    else:
      self._image = '{}@{}'.format(self._DEFAULT_IMAGE_NAME, digest)

  @property
  def container_port(self) -> int:
    return self._DEFAULT_GRPC_PORT

  @property
  def image(self) -> Text:
    return self._image

  def MakeEnvVars(
      self, model_base_path: Optional[Text] = None) -> Dict[Text, Text]:
    if model_base_path is None:
      model_base_path = self._DEFAULT_MODEL_BASE_PATH
    return {
        'MODEL_NAME': self._model_name,
        'MODEL_BASE_PATH': model_base_path
    }

  def MakeDockerRunParams(
      self,
      host_port: int,
      remote_model_base_path: Optional[Text] = None,
      host_model_base_path: Optional[Text] = None
  ):
    """Make parameters for docker `client.containers.run`.

    Args:
      host_port: Available port in the host to bind with container port.
      remote_model_base_path: (Optional) Model base path in the remote
          destination. (e.g. `gs://your_bucket/model_base_path`.) Use this
          argument if you have model in the remote place.
      host_model_base_path: (Optional) Model base path in the host machine.
          (i.e. local path during the execution.) This would create a volume
          mount from `host_model_base_path` to the container model base path
          (i.e. `/model`).

    Returns:
      A dictionary of docker run parameters.
    """
    result = dict(
        self._base_docker_run_args,
        image=self._image,
        ports={
            '{}/tcp'.format(self.container_port): host_port
        },
        environment=self.MakeEnvVars(model_base_path=remote_model_base_path))

    if host_model_base_path is not None:
      # TODO(b/149534564): Replace os.path to pathlib.PurePosixPath after py3.
      result.update(mounts=[
          docker_types.Mount(
              type='bind',
              target=self._DEFAULT_MODEL_BASE_PATH,
              source=host_model_base_path,
              read_only=True)
      ])

    return result

  def MakeClient(self, endpoint: Text) -> base_client.BaseModelServerClient:
    return tensorflow_serving_client.TensorFlowServingClient(
        endpoint=endpoint, model_name=self._model_name)
