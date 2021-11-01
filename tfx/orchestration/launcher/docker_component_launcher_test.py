# Copyright 2019 Google LLC. All Rights Reserved.
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
"""Tests for tfx.orchestration.launcher.docker_component_launcher."""

import os
from unittest import mock

import docker
import tensorflow as tf
from tfx.dsl.components.base import base_executor
from tfx.dsl.components.base import executor_spec
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.orchestration import publisher
from tfx.orchestration.config import docker_component_config
from tfx.orchestration.launcher import docker_component_launcher
from tfx.orchestration.launcher import test_utils
from tfx.types import channel_utils

from ml_metadata.proto import metadata_store_pb2


# TODO(hongyes): add e2e testing to cover docker launcher in beam/airflow.
class DockerComponentLauncherTest(tf.test.TestCase):

  def testCanLaunch(self):
    self.assertTrue(
        docker_component_launcher.DockerComponentLauncher.can_launch(
            executor_spec.ExecutorContainerSpec(image='test'),
            component_config=None))
    self.assertFalse(
        docker_component_launcher.DockerComponentLauncher.can_launch(
            executor_spec.ExecutorClassSpec(base_executor.BaseExecutor),
            component_config=None))

  @mock.patch.object(publisher, 'Publisher', autospec=True)
  @mock.patch.object(docker, 'from_env', autospec=True)
  def testLaunchSucceedsWithoutConfig(self, mock_docker_client, mock_publisher):
    mock_publisher.return_value.publish_execution.return_value = {}
    mock_run = mock_docker_client.return_value.containers.run
    mock_run.return_value.logs.return_value = []
    mock_run.return_value.wait.return_value = {'StatusCode': 0}
    context = self._create_launcher_context()

    context['launcher'].launch()

    mock_run.assert_called_once()
    _, mock_kwargs = mock_run.call_args
    self.assertEqual('gcr://test', mock_kwargs['image'])
    self.assertListEqual([context['input_artifact'].uri],
                         mock_kwargs['command'])

  @mock.patch.object(publisher, 'Publisher', autospec=True)
  @mock.patch.object(docker, 'DockerClient', autospec=True)
  def testLaunchSucceedsWithConfig(self, mock_docker_client, mock_publisher):
    mock_publisher.return_value.publish_execution.return_value = {}
    mock_run = mock_docker_client.return_value.containers.run
    mock_run.return_value.logs.return_value = []
    mock_run.return_value.wait.return_value = {'StatusCode': 0}
    docker_config = docker_component_config.DockerComponentConfig(
        docker_server_url='http://mock.docker.server',
        environment={'name': 'value'},
        privileged=True,
        volumes=['/local/etc:/local/etc'],
        ports={'2222/tcp': 3333})
    context = self._create_launcher_context(docker_config)

    context['launcher'].launch()

    mock_run.assert_called_once()
    _, mock_kwargs = mock_run.call_args
    self.assertEqual('gcr://test', mock_kwargs['image'])
    self.assertListEqual([context['input_artifact'].uri],
                         mock_kwargs['command'])
    mock_docker_client.assert_called_with(base_url='http://mock.docker.server')
    self.assertDictEqual({'name': 'value'}, mock_kwargs['environment'])
    self.assertTrue(mock_kwargs['privileged'])
    self.assertListEqual(['/local/etc:/local/etc'], mock_kwargs['volumes'])
    self.assertDictEqual({'2222/tcp': 3333}, mock_kwargs['ports'])

  @mock.patch.object(publisher, 'Publisher', autospec=True)
  @mock.patch.object(docker, 'from_env', autospec=True)
  def testLaunchWithErrorCode(self, mock_docker_client, mock_publisher):
    mock_publisher.return_value.publish_execution.return_value = {}
    mock_run = mock_docker_client.return_value.containers.run
    mock_run.return_value.logs.return_value = []
    mock_run.return_value.wait.return_value = {'StatusCode': 1}
    launcher = self._create_launcher_context()['launcher']

    with self.assertRaises(RuntimeError):
      launcher.launch()

  def _create_launcher_context(self, component_config=None):
    test_dir = self.get_temp_dir()

    connection_config = metadata_store_pb2.ConnectionConfig()
    connection_config.sqlite.SetInParent()
    metadata_connection = metadata.Metadata(connection_config)

    pipeline_root = os.path.join(test_dir, 'Test')

    input_artifact = test_utils._InputArtifact()
    input_artifact.uri = os.path.join(test_dir, 'input')

    component = test_utils._FakeComponent(
        name='FakeComponent',
        input_channel=channel_utils.as_channel([input_artifact]),
        custom_executor_spec=executor_spec.ExecutorContainerSpec(
            image='gcr://test', args=['{{input_dict["input"][0].uri}}']))

    pipeline_info = data_types.PipelineInfo(
        pipeline_name='Test', pipeline_root=pipeline_root, run_id='123')

    driver_args = data_types.DriverArgs(enable_cache=True)

    launcher = docker_component_launcher.DockerComponentLauncher.create(
        component=component,
        pipeline_info=pipeline_info,
        driver_args=driver_args,
        metadata_connection=metadata_connection,
        beam_pipeline_args=[],
        additional_pipeline_args={},
        component_config=component_config)

    return {'launcher': launcher, 'input_artifact': input_artifact}


if __name__ == '__main__':
  tf.test.main()
