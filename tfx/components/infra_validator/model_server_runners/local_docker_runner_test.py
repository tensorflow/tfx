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
from tfx.components.infra_validator import binary_kinds
from tfx.components.infra_validator.model_server_clients import base_client
from tfx.components.infra_validator.model_server_runners import local_docker_runner
from tfx.proto import infra_validator_pb2
from tfx.types import standard_artifacts

ModelState = base_client.ModelState


def _create_serving_spec(payload: Dict[Text, Any]):
  result = infra_validator_pb2.ServingSpec()
  json_format.ParseDict(payload, result)
  return result


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
    self.model_name = 'chicago-taxi'

    # Mock _find_available_port
    patcher = mock.patch.object(local_docker_runner, '_find_available_port')
    self.find_available_port = patcher.start()
    self.find_available_port.return_value = 1234
    self.addCleanup(patcher.stop)

    # Mock docker.DockerClient
    patcher = mock.patch('docker.DockerClient')
    self.docker_client_cls = patcher.start()
    self.docker_client = self.docker_client_cls.return_value
    self.addCleanup(patcher.stop)

    self.serving_spec = _create_serving_spec({
        'tensorflow_serving': {
            'tags': ['1.15.0']},
        'local_docker': {},
        'model_name': 'chicago-taxi',
    })
    self.model_server_client = None

  def _ParseBinaryKind(self, serving_spec: infra_validator_pb2.ServingSpec):
    binary_kind = binary_kinds.parse_binary_kinds(serving_spec)[0]
    patcher = mock.patch.object(binary_kind, 'MakeClient')
    self.make_client_mock = patcher.start()
    self.model_server_client = self.make_client_mock.return_value
    self.addCleanup(patcher.stop)
    return binary_kind

  def _CreateLocalDockerRunner(self):
    binary_kind = self._ParseBinaryKind(self.serving_spec)
    return local_docker_runner.LocalDockerRunner(
        model=self.model,
        binary_kind=binary_kind,
        serving_spec=self.serving_spec)

  def testStart(self):
    # Prepare mocks and variables.
    runner = self._CreateLocalDockerRunner()

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
            'MODEL_BASE_PATH': '/model'
        },
        auto_remove=True,
        detach=True
    ), run_kwargs)
    self.make_client_mock.assert_called_with('localhost:1234')

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
    self.model_server_client.GetModelState.return_value = ModelState.AVAILABLE

    # Act.
    succeeded = runner.WaitUntilModelAvailable(timeout_secs=10)

    # Check states.
    self.assertTrue(succeeded)
    container.reload.assert_called()
    self.model_server_client.GetModelState.assert_called()

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
    container.status = 'dead'  # Bad status.

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
    self.model_server_client.GetModelState.return_value = ModelState.UNAVAILABLE

    # Act.
    succeeded = runner.WaitUntilModelAvailable(timeout_secs=10)

    # Check result.
    self.model_server_client.GetModelState.assert_called()
    self.assertFalse(succeeded)

  @mock.patch('time.time')
  @mock.patch('time.sleep')
  def testWaitUntilModelAvailable_FailIfStatusNotReadyUntilDeadline(
      self, mock_sleep, mock_time):
    del mock_sleep

    # Prepare mocks and variables.
    container = self.docker_client.containers.run.return_value
    runner = self._CreateLocalDockerRunner()

    # Setup state.
    runner.Start()
    container.status = 'running'
    self.model_server_client.GetModelState.return_value = ModelState.NOT_READY
    mock_time.side_effect = list(range(20))

    # Act.
    succeeded = runner.WaitUntilModelAvailable(timeout_secs=10)

    # Check result.
    self.model_server_client.GetModelState.assert_called()
    self.assertFalse(succeeded)

  @mock.patch('time.time')
  @mock.patch('time.sleep')
  def testWaitUntilModelAvailable_FailIfContainerNotRunningUntilDeadline(
      self, mock_sleep, mock_time):
    del mock_sleep

    # Prepare mocks and variables.
    container = self.docker_client.containers.run.return_value
    runner = self._CreateLocalDockerRunner()

    # Setup state.
    runner.Start()
    container.status = 'created'  # container not running.
    mock_time.side_effect = list(range(20))

    # Act.
    succeeded = runner.WaitUntilModelAvailable(timeout_secs=10)

    # Check result.
    self.assertFalse(succeeded)


if __name__ == '__main__':
  tf.test.main()
