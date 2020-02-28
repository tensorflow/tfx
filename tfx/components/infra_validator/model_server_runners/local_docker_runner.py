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
from six.moves.urllib import parse
from typing import Text

from tfx.components.infra_validator.model_server_clients import base_client
from tfx.components.infra_validator.model_server_clients import factory
from tfx.components.infra_validator.model_server_runners import base_runner
from tfx.proto import infra_validator_pb2
from tfx.types import standard_artifacts
from tfx.utils import path_utils

ModelState = base_client.ModelState

_MODEL_STATE_POLLING_INTERVALS_SECONDS = 1
# Default tensorflow/serving grpc port
# https://www.tensorflow.org/tfx/serving/docker#running_a_serving_image
_TENSORFLOW_SERVING_GRPC_PORT = '8500/tcp'


class LocalDockerModelServerRunner(base_runner.BaseModelServerRunner):
  """A model server runner that runs in a local docker runtime.

  You need to pre-install docker in the machine that is running InfraValidator
  component. For that reason, it is recommended to use this runner only for
  testing purpose.
  """

  def __init__(self, model: standard_artifacts.Model,
               image_uri: Text,
               serving_spec: infra_validator_pb2.ServingSpec,
               client_factory: factory.ClientFactory):
    self._model_dir = os.path.dirname(path_utils.serving_model_path(model.uri))
    self._image_uri = image_uri
    self._serving_spec = serving_spec
    self._docker = self._MakeDockerClientFromConfig(serving_spec.local_docker)
    self._client_factory = client_factory
    self._container = None
    self._client = None

  def __repr__(self):
    attrs = dict(image_uri=self._image_uri)
    return '<{class_name} {attrs}>'.format(
        class_name=self.__class__.__name__,
        attrs=' '.join('{}={}'.format(key, value)
                       for key, value in attrs.items()))

  def _MakeDockerClientFromConfig(
      self, config: infra_validator_pb2.LocalDockerConfig):
    params = {}
    params['timeout'] = (config.client_timeout_seconds
                         or docker.constants.DEFAULT_TIMEOUT_SECONDS)
    if config.client_base_url:
      params['base_url'] = config.client_base_url
    if config.client_api_version:
      params['version'] = config.client_api_version
    logging.info('Initializing docker client with parameter %s', params)
    return docker.DockerClient(**params)

  def Start(self):
    if self._container:
      raise RuntimeError('You cannot start model server multiple times.')

    # TODO(jjong): Current implementation assumes tensorflow serving image.
    # Needs refactoring.
    model_name = self._serving_spec.model_name
    if model_name != os.path.basename(self._model_dir):
      raise ValueError('model_name does not match Model artifact directory.')
    grpc_port = self._FindAvailablePort()
    endpoint = 'localhost:{}'.format(grpc_port)

    run_args = dict(
        image=self._image_uri,
        ports={_TENSORFLOW_SERVING_GRPC_PORT: grpc_port},
        environment={
            'MODEL_NAME': model_name
        },
        auto_remove=True,
        detach=True)
    if self._IsLocalUri(self._model_dir):
      # If model is in the host machine, we need to bind the local model
      # directory to the container. Tensorflow serving uses /models/ directory
      # as a default model base directory.
      logging.info(os.listdir(self._model_dir))
      run_args['mounts'] = [
          docker.types.Mount(
              type='bind',
              target='/models/{}'.format(model_name),
              source=self._model_dir,
              read_only=True)
      ]
    else:
      # Else the model is in the remote location and will be retrieved using
      # tensorflow gfile abstraction.
      model_base_dir = os.path.dirname(self._model_dir)
      run_args['environment']['MODEL_BASE_DIR'] = model_base_dir

    logging.info('Running container with argument %s', run_args)
    self._container = self._docker.containers.run(**run_args)

    self._client = self._client_factory(endpoint)

  @staticmethod
  def _FindAvailablePort():
    with contextlib.closing(
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
      sock.bind(('localhost', 0))
      sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
      _, port = sock.getsockname()
      return port

  @staticmethod
  def _IsLocalUri(uri):
    parsed = parse.urlparse(uri)
    return parsed.scheme == ''  # pylint: disable=g-explicit-bool-comparison

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

  @property
  def client(self):
    return self._client
