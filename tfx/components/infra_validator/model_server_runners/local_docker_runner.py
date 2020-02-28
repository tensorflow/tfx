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
"""Module for LocalDockerModelServerRunner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import os
import socket
import time

from absl import logging
import docker
from typing import Text

from tfx.components.infra_validator import binary_kinds
from tfx.components.infra_validator.model_server_clients import base_client
from tfx.components.infra_validator.model_server_runners import base_runner
from tfx.proto import infra_validator_pb2
from tfx.types import standard_artifacts
from tfx.utils import path_utils

ModelState = base_client.ModelState

_MODEL_STATE_POLLING_INTERVALS_SECONDS = 1


def _make_docker_client(config: infra_validator_pb2.LocalDockerConfig):
  params = {}
  if config.client_timeout_seconds:
    params['timeout'] = config.client_timeout_seconds
  if config.client_base_url:
    params['base_url'] = config.client_base_url
  if config.client_api_version:
    params['version'] = config.client_api_version
  return docker.DockerClient(**params)


def _find_available_port():
  """Find available port in the host machine."""
  with contextlib.closing(
      socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
    sock.bind(('localhost', 0))
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    _, port = sock.getsockname()
    return port


def _parse_model_path(model_path: Text):
  """Parse model path into a base path, model name, and a version.

  Args:
    model_path: Path to the SavedModel (or other format) in the structure of
        `{model_base_path}/{model_name}/{version}`, where version is an integer.
  Raises:
    ValueError: if the model_path does not conform to the expected directory
        structure.
  Returns:
    `model_base_path`, `model_name`, and integer `version`.
  """
  model_path, version = os.path.split(model_path)
  if not version.isdigit():
    raise ValueError(
        '{} does not conform to tensorflow serving directory structure: '
        'BASE_PATH/model_name/int_version.'.format(model_path))
  base_path, model_name = os.path.split(model_path)
  return base_path, model_name, int(version)


class LocalDockerRunner(base_runner.BaseModelServerRunner):
  """A model server runner that runs in a local docker runtime.

  You need to pre-install docker in the machine that is running InfraValidator
  component. For that reason, it is recommended to use this runner only for
  testing purpose.
  """

  def __init__(self, model: standard_artifacts.Model,
               binary_kind: binary_kinds.BinaryKind,
               serving_spec: infra_validator_pb2.ServingSpec):
    """Make a local docker runner.

    Args:
      model: A model artifact to infra validate.
      binary_kind: A BinaryKind to run.
      serving_spec: A ServingSpec instance.
    """
    base_path, model_name, version = _parse_model_path(
        path_utils.serving_model_path(model.uri))

    if model_name != serving_spec.model_name:
      raise ValueError(
          'ServingSpec.model_name ({}) does not match the model name ({}) from'
          'the Model artifact.'.format(
              serving_spec.model_name, model_name))

    self._model_base_path = base_path
    self._model_name = model_name
    self._model_version = version
    self._binary_kind = binary_kind
    self._serving_spec = serving_spec
    self._docker = _make_docker_client(serving_spec.local_docker)
    self._container = None
    self._client = None

  def __repr__(self):
    return 'LocalDockerRunner(image: {image})'.format(
        image=self._binary_kind.image)

  @property
  def client(self):
    return self._client

  @property
  def model_path(self):
    return os.path.join(self._model_base_path, self._model_name)

  @property
  def model_version_path(self):
    return os.path.join(self._model_base_path, self._model_name,
                        str(self._model_version))

  def Start(self):
    if self._container:
      raise RuntimeError('You cannot start model server multiple times.')

    host_port = _find_available_port()
    endpoint = 'localhost:{}'.format(host_port)

    if isinstance(self._binary_kind, binary_kinds.TensorFlowServing):
      is_local_model = os.path.exists(self.model_version_path)
      if is_local_model:
        run_params = self._binary_kind.MakeDockerRunParams(
            host_port=host_port,
            host_model_path=self.model_path)
      else:
        run_params = self._binary_kind.MakeDockerRunParams(
            host_port=host_port,
            model_base_path=self._model_base_path)
    else:
      raise ValueError('Unsupported binary kind {}'.format(
          type(self._binary_kind).__name__))

    logging.info('Running container with parameter %s', run_params)
    self._container = self._docker.containers.run(**run_params)
    self._client = self._binary_kind.MakeClient(endpoint)

  def WaitUntilModelAvailable(self, timeout_secs):
    if not self._container:
      raise RuntimeError('container is not started.')

    deadline = time.time() + timeout_secs
    while time.time() < deadline:
      # Reload container attributes from server. This is the only right way to
      # retrieve the latest container status from docker engine.
      self._container.reload()
      # Once container is up and running, use a client to wait until model is
      # available.
      if self._container.status == 'running':
        state = self._client.GetModelState()
        if state == ModelState.AVAILABLE:
          return True
        elif state == ModelState.UNAVAILABLE:
          return False
        else:
          time.sleep(_MODEL_STATE_POLLING_INTERVALS_SECONDS)
      # Docker status is one of 'created', 'restarting', 'running', 'removing',
      # 'paused', 'exited', or 'dead'. Status other than 'created' and 'running'
      # indicates failure.
      elif self._container.status != 'created':
        logging.error('Container has reached %s state before available; marking'
                      ' model as not blessed.', self._container.status)
        return False
      else:
        time.sleep(_MODEL_STATE_POLLING_INTERVALS_SECONDS)

    # Deadline exceeded.
    logging.error('Deadline has exceeded; marking model as not blessed.')
    return False

  def Stop(self):
    if self._container:
      logging.info('Stopping container.')
      self._container.stop()
    self._docker.close()
