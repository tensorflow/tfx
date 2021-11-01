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
"""Tests for tfx.experimental.pipeline_testing.stub_component_launcher."""

import os

from unittest import mock
import tensorflow as tf

from tfx.dsl.io import fileio
from tfx.experimental.pipeline_testing import stub_component_launcher
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.orchestration import publisher
from tfx.orchestration.launcher import test_utils
from tfx.types import channel_utils
from tfx.utils import io_utils

from ml_metadata.proto import metadata_store_pb2


class StubComponentLauncherTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    test_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    connection_config = metadata_store_pb2.ConnectionConfig()
    connection_config.sqlite.SetInParent()
    self.metadata_connection = metadata.Metadata(connection_config)

    self.pipeline_root = os.path.join(test_dir, 'Test')

    self.input_key = 'input'
    self.output_key = 'output'

    self.input_dir = os.path.join(test_dir, self.input_key)
    self.output_dir = os.path.join(test_dir, self.output_key)
    self.record_dir = os.path.join(test_dir, 'record')
    fileio.makedirs(self.input_dir)
    fileio.makedirs(self.output_dir)
    fileio.makedirs(self.record_dir)

    input_artifact = test_utils._InputArtifact()  # pylint: disable=protected-access
    input_artifact.uri = os.path.join(self.input_dir, 'result.txt')
    output_artifact = test_utils._OutputArtifact()  # pylint: disable=protected-access
    output_artifact.uri = os.path.join(self.output_dir, 'result.txt')
    self.component = test_utils._FakeComponent(  # pylint: disable=protected-access
        name='FakeComponent',
        input_channel=channel_utils.as_channel([input_artifact]),
        output_channel=channel_utils.as_channel([output_artifact]))
    self.driver_args = data_types.DriverArgs(enable_cache=True)

    self.pipeline_info = data_types.PipelineInfo(
        pipeline_name='Test', pipeline_root=self.pipeline_root, run_id='123')

  @mock.patch.object(publisher, 'Publisher')
  def testStubExecutor(self, mock_publisher):
    # verify whether base stub executor substitution works
    mock_publisher.return_value.publish_execution.return_value = {}

    record_file = os.path.join(self.record_dir, self.component.id,
                               self.output_key, '0', 'recorded.txt')
    io_utils.write_string_file(record_file, 'hello world')

    stub_component_launcher.StubComponentLauncher.initialize(
        test_data_dir=self.record_dir, test_component_ids=[])

    launcher = stub_component_launcher.StubComponentLauncher.create(
        component=self.component,
        pipeline_info=self.pipeline_info,
        driver_args=self.driver_args,
        metadata_connection=self.metadata_connection,
        beam_pipeline_args=[],
        additional_pipeline_args={})
    launcher.launch()

    output_path = self.component.outputs[self.output_key].get()[0].uri
    copied_file = os.path.join(output_path, 'recorded.txt')
    self.assertTrue(fileio.exists(copied_file))
    contents = io_utils.read_string_file(copied_file)
    self.assertEqual('hello world', contents)

  @mock.patch.object(publisher, 'Publisher')
  def testExecutor(self, mock_publisher):
    # verify whether original executors can run
    mock_publisher.return_value.publish_execution.return_value = {}

    io_utils.write_string_file(
        os.path.join(self.input_dir, 'result.txt'), 'test')

    stub_component_launcher.StubComponentLauncher.initialize(
        test_data_dir=self.record_dir, test_component_ids=[self.component.id])

    launcher = stub_component_launcher.StubComponentLauncher.create(
        component=self.component,
        pipeline_info=self.pipeline_info,
        driver_args=self.driver_args,
        metadata_connection=self.metadata_connection,
        beam_pipeline_args=[],
        additional_pipeline_args={})
    self.assertEqual(
        launcher._component_info.component_type,  # pylint: disable=protected-access
        '.'.join([  # pylint: disable=protected-access
            test_utils._FakeComponent.__module__,  # pylint: disable=protected-access
            test_utils._FakeComponent.__name__  # pylint: disable=protected-access
        ]))
    launcher.launch()

    output_path = self.component.outputs[self.output_key].get()[0].uri
    self.assertTrue(fileio.exists(output_path))
    contents = io_utils.read_string_file(output_path)
    self.assertEqual('test', contents)


if __name__ == '__main__':
  tf.test.main()
