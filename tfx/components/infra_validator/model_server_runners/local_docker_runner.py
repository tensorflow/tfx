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
from docker import errors as docker_errors

from tfx.components.infra_validator import error_types
from tfx.components.infra_validator import serving_binary_lib
from tfx.components.infra_validator.model_server_runners import base_runner
from tfx.proto import infra_validator_pb2
from tfx.utils import path_utils
from tfx.utils import time_utils

_POLLING_INTERVAL_SEC = 1


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


class LocalDockerRunner(base_runner.BaseModelServerRunner):
  """A model server runner that runs in a local docker runtime.

  You need to pre-install docker in the machine that is running InfraValidator
  component. For that reason, it is recommended to use this runner only for
  testing purpose.
  """

  def __init__(self, standard_model_path: path_utils.StandardModelPath,
               serving_binary: serving_binary_lib.ServingBinary,
               serving_spec: infra_validator_pb2.ServingSpec):
    """Make a local docker runner.

    Args:
      standard_model_path: A StandardModelPath.
      serving_binary: A ServingBinary to run.
      serving_spec: A ServingSpec instance.
    """
    self._standard_model_path = standard_model_path
    self._serving_binary = serving_binary
    self._serving_spec = serving_spec
    self._docker = _make_docker_client(serving_spec.local_docker)
    self._container = None
    self._endpoint = None

  def __repr__(self):
    return 'LocalDockerRunner(image: {image})'.format(
        image=self._serving_binary.image)

  def GetEndpoint(self):
    if not self._endpoint:
      raise error_types.IllegalState(
          'Endpoint is not yet created. You should call Start() first.')
    return self._endpoint

  def Start(self):
    if self._container:
      raise error_types.IllegalState(
          'You cannot start model server multiple times.')

    host_port = _find_available_port()
    self._endpoint = 'localhost:{}'.format(host_port)

    if isinstance(self._serving_binary, serving_binary_lib.TensorFlowServing):
      is_local = os.path.exists(self._standard_model_path.full_path)
      if is_local:
        run_params = self._serving_binary.MakeDockerRunParams(
            host_port=host_port,
            host_model_base_path=self._standard_model_path.base_path)
      else:
        run_params = self._serving_binary.MakeDockerRunParams(
            host_port=host_port,
            remote_model_base_path=self._standard_model_path.base_path)
    else:
      raise NotImplementedError('Unsupported serving binary {}'.format(
          type(self._serving_binary).__name__))

    logging.info('Running container with parameter %s', run_params)
    self._container = self._docker.containers.run(**run_params)

  def WaitUntilRunning(self, deadline):
    if not self._container:
      raise error_types.IllegalState('container is not started.')

    while time_utils.utc_timestamp() < deadline:
      try:
        # Reload container attributes from server. This is the only right way to
        # retrieve the latest container status from docker engine.
        self._container.reload()
        status = self._container.status
      except docker_errors.NotFound:
        # If the job has been aborted and container has specified auto_removal
        # to True, we might get a NotFound error during container.reload().
        raise error_types.JobAborted(
            'Container not found. Possibly removed after the job has been '
            'aborted.')
      # The container is just created and not yet in the running status.
      if status == 'created':
        time.sleep(_POLLING_INTERVAL_SEC)
        continue
      # The container is running :)
      if status == 'running':
        return
      # Other docker status ('created', 'restarting', 'running', 'removing',
      # 'paused', 'exited', or 'dead') indicates failure.
      raise error_types.JobAborted(
          'Job has been aborted (container status={})'.format(status))

    raise error_types.DeadlineExceeded(
        'Deadline exceeded while waiting for the container to be running.')

  def Stop(self):
    if self._container:
      logging.info('Stopping container.')
      self._container.stop()
    self._docker.close()
