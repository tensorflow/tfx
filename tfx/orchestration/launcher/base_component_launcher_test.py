# Lint as: python2, python3
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
"""Tests for tfx.orchestration.component_launcher."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from unittest import mock
import tensorflow as tf
from tfx.dsl.io import fileio
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.orchestration import publisher
from tfx.orchestration.launcher import in_process_component_launcher
from tfx.orchestration.launcher import test_utils
from tfx.types import channel_utils

from ml_metadata.proto import metadata_store_pb2
from tensorflow.python.lib.io import file_io  # pylint: disable=g-direct-tensorflow-import


class ComponentRunnerTest(tf.test.TestCase):

  @mock.patch.object(publisher, 'Publisher')
  def testRun(self, mock_publisher):
    mock_publisher.return_value.publish_execution.return_value = {}

    test_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    connection_config = metadata_store_pb2.ConnectionConfig()
    connection_config.sqlite.SetInParent()
    metadata_connection = metadata.Metadata(connection_config)

    pipeline_root = os.path.join(test_dir, 'Test')
    input_path = os.path.join(test_dir, 'input')
    fileio.makedirs(os.path.dirname(input_path))
    file_io.write_string_to_file(input_path, 'test')

    input_artifact = test_utils._InputArtifact()
    input_artifact.uri = input_path

    component = test_utils._FakeComponent(
        name='FakeComponent',
        input_channel=channel_utils.as_channel([input_artifact]))

    pipeline_info = data_types.PipelineInfo(
        pipeline_name='Test', pipeline_root=pipeline_root, run_id='123')

    driver_args = data_types.DriverArgs(enable_cache=True)

    # We use InProcessComponentLauncher to test BaseComponentLauncher logics.
    launcher = in_process_component_launcher.InProcessComponentLauncher.create(
        component=component,
        pipeline_info=pipeline_info,
        driver_args=driver_args,
        metadata_connection=metadata_connection,
        beam_pipeline_args=[],
        additional_pipeline_args={})
    self.assertEqual(
        launcher._component_info.component_type, '.'.join([
            test_utils._FakeComponent.__module__,
            test_utils._FakeComponent.__name__
        ]))
    launcher.launch()

    output_path = component.outputs['output'].get()[0].uri
    self.assertTrue(fileio.exists(output_path))
    contents = file_io.read_file_to_string(output_path)
    self.assertEqual('test', contents)


if __name__ == '__main__':
  tf.test.main()
