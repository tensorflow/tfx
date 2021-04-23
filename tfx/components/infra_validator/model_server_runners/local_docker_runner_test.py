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
from typing import Any, Dict, Text
from unittest import mock

from docker import errors as docker_errors
import tensorflow as tf
from tfx.components.infra_validator import error_types
from tfx.components.infra_validator import serving_bins
from tfx.components.infra_validator.model_server_runners import local_docker_runner
from tfx.proto import infra_validator_pb2
from tfx.types import standard_artifacts
from tfx.utils import path_utils

from google.protobuf import json_format


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
    self._model = standard_artifacts.Model()
    self._model.uri = os.path.join(base_dir, 'trainer', 'current')
    self._model_name = 'chicago-taxi'
    self._model_path = path_utils.serving_model_path(self._model.uri)

    # Mock docker.DockerClient
    patcher = mock.patch('docker.DockerClient')
    self._docker_client = patcher.start().return_value
    self.addCleanup(patcher.stop)

    self._serving_spec = _create_serving_spec({
        'tensorflow_serving': {
            'tags': ['1.15.0']},
        'local_docker': {},
        'model_name': self._model_name,
    })
    self._serving_binary = serving_bins.parse_serving_binaries(
        self._serving_spec)[0]
    patcher = mock.patch.object(self._serving_binary, 'MakeClient')
    self._model_server_client = patcher.start().return_value
    self.addCleanup(patcher.stop)

  def _CreateLocalDockerRunner(self):
    return local_docker_runner.LocalDockerRunner(
        model_path=self._model_path,
        serving_binary=self._serving_binary,
        serving_spec=self._serving_spec)

  def testStart(self):
    # Prepare mocks and variables.
    runner = self._CreateLocalDockerRunner()

    # Act.
    runner.Start()

    # Check calls.
    self._docker_client.containers.run.assert_called()
    _, run_kwargs = self._docker_client.containers.run.call_args
    self.assertDictContainsSubset(dict(
        image='tensorflow/serving:1.15.0',
        environment={
            'MODEL_NAME': 'chicago-taxi',
            'MODEL_BASE_PATH': '/model'
        },
        publish_all_ports=True,
        auto_remove=True,
        detach=True
    ), run_kwargs)

  def testStartMultipleTimesFail(self):
    # Prepare mocks and variables.
    runner = self._CreateLocalDockerRunner()

    # Act.
    runner.Start()
    with self.assertRaises(AssertionError) as err:
      runner.Start()

    # Check errors.
    self.assertEqual(
        str(err.exception), 'You cannot start model server multiple times.')

  @mock.patch('time.time')
  def testGetEndpoint_AfterWaitUntilRunning(self, mock_time):
    # Prepare mocks and variables.
    runner = self._CreateLocalDockerRunner()
    mock_time.side_effect = list(range(10))
    container = self._docker_client.containers.run.return_value
    container.status = 'running'
    container.ports = {
        '8500/tcp': [{'HostIp': '0.0.0.0', 'HostPort': '1234'}],  # gRPC port.
        '8501/tcp': [{'HostIp': '0.0.0.0', 'HostPort': '5678'}]   # REST port.
    }

    # Act.
    runner.Start()
    runner.WaitUntilRunning(deadline=10)
    endpoint = runner.GetEndpoint()

    # Check result.
    self.assertEqual(endpoint, 'localhost:1234')

  def testGetEndpoint_FailWithoutStartingFirst(self):
    # Prepare mocks and variables.
    runner = self._CreateLocalDockerRunner()

    # Act.
    with self.assertRaises(AssertionError):
      runner.GetEndpoint()

  @mock.patch('time.time')
  def testWaitUntilRunning(self, mock_time):
    # Prepare mocks and variables.
    container = self._docker_client.containers.run.return_value
    runner = self._CreateLocalDockerRunner()
    mock_time.side_effect = list(range(10))

    # Setup state.
    runner.Start()
    container.status = 'running'

    # Act.
    try:
      runner.WaitUntilRunning(deadline=10)
    except Exception as e:  # pylint: disable=broad-except
      self.fail(e)

    # Check states.
    container.reload.assert_called()

  @mock.patch('time.time')
  def testWaitUntilRunning_FailWithoutStartingFirst(self, mock_time):
    # Prepare runner.
    runner = self._CreateLocalDockerRunner()
    mock_time.side_effect = list(range(10))

    # Act.
    with self.assertRaises(AssertionError) as err:
      runner.WaitUntilRunning(deadline=10)

    # Check errors.
    self.assertEqual(str(err.exception), 'container has not been started.')

  @mock.patch('time.time')
  def testWaitUntilRunning_FailWhenBadContainerStatus(self, mock_time):
    # Prepare mocks and variables.
    container = self._docker_client.containers.run.return_value
    runner = self._CreateLocalDockerRunner()
    mock_time.side_effect = list(range(10))

    # Setup state.
    runner.Start()
    container.status = 'dead'  # Bad status.

    # Act.
    with self.assertRaises(error_types.JobAborted):
      runner.WaitUntilRunning(deadline=10)

  @mock.patch('time.time')
  @mock.patch('time.sleep')
  def testWaitUntilRunning_FailIfNotRunningUntilDeadline(
      self, mock_sleep, mock_time):
    # Prepare mocks and variables.
    container = self._docker_client.containers.run.return_value
    runner = self._CreateLocalDockerRunner()
    mock_time.side_effect = list(range(20))

    # Setup state.
    runner.Start()
    container.status = 'created'

    # Act.
    with self.assertRaises(error_types.DeadlineExceeded):
      runner.WaitUntilRunning(deadline=10)

    # Check result.
    mock_sleep.assert_called()

  @mock.patch('time.time')
  def testWaitUntilRunning_FailIfContainerNotFound(self, mock_time):
    # Prepare mocks and variables.
    container = self._docker_client.containers.run.return_value
    container.reload.side_effect = docker_errors.NotFound('message required.')
    runner = self._CreateLocalDockerRunner()
    mock_time.side_effect = list(range(20))

    # Setup state.
    runner.Start()

    # Act.
    with self.assertRaises(error_types.JobAborted):
      runner.WaitUntilRunning(deadline=10)


if __name__ == '__main__':
  tf.test.main()
