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
"""Tests for tfx.components.infra_validator.model_server_runners.local_docker_runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import mock
import tensorflow as tf
from typing import Any, Dict, Text

from google.protobuf import json_format
from tfx.components.infra_validator.model_server_clients import base_client
from tfx.components.infra_validator.model_server_runners import local_docker_runner
from tfx.proto import infra_validator_pb2
from tfx.types import standard_artifacts

LocalDockerConfig = infra_validator_pb2.LocalDockerConfig
ModelState = base_client.ModelState
LocalDockerModelServerRunner = local_docker_runner.LocalDockerModelServerRunner


def _create_local_docker_config(payload: Dict[Text, Any]):
  config = LocalDockerConfig()
  json_format.ParseDict(payload, config)
  return config


class LocalDockerRunnerTest(tf.test.TestCase):

  def setUp(self):
    super(LocalDockerRunnerTest, self).setUp()

    base_dir = os.path.join(
        os.path.dirname(  # components/
            os.path.dirname(  # infra_validator/
                os.path.dirname(__file__))),  # model_server_runners/
        'testdata'
    )
    self.model = standard_artifacts.Model()
    self.model.uri = os.path.join(base_dir, 'trainer', 'current')

    # Mock LocalDockerModelServerRunner._FindAvailablePort
    self.find_available_port_patcher = mock.patch.object(
        LocalDockerModelServerRunner, '_FindAvailablePort')
    self.find_available_port = self.find_available_port_patcher.start()
    self.find_available_port.return_value = 1234

    # Mock docker.DockerClient
    self.docker_client_patcher = mock.patch('docker.DockerClient')
    self.docker_client_cls = self.docker_client_patcher.start()
    self.docker_client = self.docker_client_cls.return_value

    # Mock client factory
    self.client_factory = mock.Mock()
    self.client = self.client_factory.return_value

  def tearDown(self):
    super(LocalDockerRunnerTest, self).tearDown()
    self.find_available_port_patcher.stop()
    self.docker_client_patcher.stop()

  def _CreateLocalDockerRunner(self, image_uri='docker_image_uri',
                               config_dict=None):
    return LocalDockerModelServerRunner(
        model=self.model,
        image_uri=image_uri,
        config=_create_local_docker_config(config_dict or {}),
        client_factory=self.client_factory
    )

  def testStart(self):
    # Prepare mocks and variables.
    runner = self._CreateLocalDockerRunner(
        image_uri='tensorflow/serving:1.15.0')

    # Act.
    runner.Start()

    # Check calls.
    self.docker_client.containers.run.assert_called()
    _, run_kwargs = self.docker_client.containers.run.call_args
    self.assertDictContainsSubset(dict(
        image='tensorflow/serving:1.15.0',
        ports={'8500/tcp': 1234},
        environment={
            'MODEL_NAME': 'chicago-taxi',
        },
        auto_remove=True,
        detach=True
    ), run_kwargs)
    self.client_factory.assert_called_with('localhost:1234')

  def testStartMultipleTimesFail(self):
    # Prepare mocks and variables.
    runner = self._CreateLocalDockerRunner()

    # Act.
    runner.Start()
    with self.assertRaises(RuntimeError) as err:
      runner.Start()

    # Check errors.
    self.assertEqual(
        str(err.exception), 'You cannot start model server multiple times.')

  def testWaitUntilModelAvailable(self):
    # Prepare mocks and variables.
    container = self.docker_client.containers.run.return_value
    runner = self._CreateLocalDockerRunner()

    # Setup state.
    runner.Start()
    container.status = 'running'
    self.client.GetModelState.return_value = ModelState.AVAILABLE

    # Act.
    succeeded = runner.WaitUntilModelAvailable(timeout_secs=10)

    # Check states.
    self.assertTrue(succeeded)
    container.reload.assert_called()
    self.client.GetModelState.assert_called()

  def testWaitUntilModelAvailable_FailWithoutStartingFirst(self):
    # Prepare runner.
    runner = self._CreateLocalDockerRunner()

    # Act.
    with self.assertRaises(RuntimeError) as err:
      runner.WaitUntilModelAvailable(timeout_secs=10)

    # Check errors.
    self.assertEqual(str(err.exception), 'container is not started.')

  def testWaitUntilModelAvailable_FailWhenBadContainerStatus(self):
    # Prepare mocks and variables.
    container = self.docker_client.containers.run.return_value
    runner = self._CreateLocalDockerRunner()

    # Setup state.
    runner.Start()
    container.status = 'dead'  # Bad status

    # Act.
    succeeded = runner.WaitUntilModelAvailable(timeout_secs=10)

    # Check result.
    self.assertFalse(succeeded)

  def testWaitUntilModelAvailable_FailWhenModelUnavailable(self):
    # Prepare mocks and variables.
    container = self.docker_client.containers.run.return_value
    runner = self._CreateLocalDockerRunner()

    # Setup state.
    runner.Start()
    container.status = 'running'
    self.client.return_value = ModelState.UNAVAILABLE

    # Act.
    succeeded = runner.WaitUntilModelAvailable(timeout_secs=10)

    # Check result.
    self.assertFalse(succeeded)

  @mock.patch('time.time')
  @mock.patch('time.sleep')
  def testWaitUntilModelAvailable_FailIfStatusNotReadyUntilDeadline(
      self, mock_sleep, mock_time):
    # Prepare mocks and variables.
    container = self.docker_client.containers.run.return_value
    runner = self._CreateLocalDockerRunner()

    # Setup state.
    runner.Start()
    container.status = 'running'
    self.client.return_value = ModelState.NOT_READY
    mock_time.side_effect = list(range(20))

    # Act.
    succeeded = runner.WaitUntilModelAvailable(timeout_secs=10)

    # Check result.
    self.assertFalse(succeeded)

  @mock.patch('time.time')
  @mock.patch('time.sleep')
  def testWaitUntilModelAvailable_FailIfContainerNotRunningUntilDeadline(
      self, mock_sleep, mock_time):
    # Prepare mocks and variables.
    container = self.docker_client.containers.run.return_value
    runner = self._CreateLocalDockerRunner()

    # Setup state.
    runner.Start()
    container.status = 'running'
    self.client.return_value = ModelState.NOT_READY
    mock_time.side_effect = list(range(20))

    # Act.
    succeeded = runner.WaitUntilModelAvailable(timeout_secs=10)

    # Check result.
    self.assertFalse(succeeded)


if __name__ == '__main__':
  tf.test.main()
