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
import mock
import tensorflow as tf
from typing import Any, Dict, List, Optional, Text
from ml_metadata.proto import metadata_store_pb2
from tensorflow.python.lib.io import file_io  # pylint: disable=g-direct-tensorflow-import
from tfx import types
from tfx.components.base import base_component
from tfx.components.base import base_driver
from tfx.components.base import base_executor
from tfx.orchestration import component_launcher
from tfx.orchestration import data_types
from tfx.orchestration import publisher
from tfx.types import artifact_utils
from tfx.types import channel_utils


class _FakeDriver(base_driver.BaseDriver):

  def pre_execution(
      self,
      input_dict: Dict[Text, types.Channel],
      output_dict: Dict[Text, types.Channel],
      exec_properties: Dict[Text, Any],
      driver_args: data_types.DriverArgs,
      pipeline_info: data_types.PipelineInfo,
      component_info: data_types.ComponentInfo,
  ) -> data_types.ExecutionDecision:
    input_artifacts = channel_utils.unwrap_channel_dict(input_dict)
    output_artifacts = channel_utils.unwrap_channel_dict(output_dict)
    tf.gfile.MakeDirs(pipeline_info.pipeline_root)
    artifact_utils.get_single_instance(
        output_artifacts['output']).uri = os.path.join(
            pipeline_info.pipeline_root, 'output')
    return data_types.ExecutionDecision(input_artifacts, output_artifacts,
                                        exec_properties, 123, False)


class _FakeExecutor(base_executor.BaseExecutor):

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    input_path = artifact_utils.get_single_uri(input_dict['input'])
    output_path = artifact_utils.get_single_uri(output_dict['output'])
    tf.gfile.Copy(input_path, output_path)


class _FakeComponentSpec(base_component.ComponentSpec):
  PARAMETERS = {}
  INPUTS = {'input': base_component.ChannelParameter(type_name='InputPath')}
  OUTPUTS = {'output': base_component.ChannelParameter(type_name='OutputPath')}


class _FakeComponent(base_component.BaseComponent):
  SPEC_CLASS = _FakeComponentSpec
  EXECUTOR_CLASS = _FakeExecutor
  DRIVER_CLASS = _FakeDriver

  def __init__(self,
               name: Text,
               input_channel: types.Channel,
               output_channel: Optional[types.Channel] = None):
    output_channel = output_channel or types.Channel(
        type_name='OutputPath', artifacts=[types.Artifact('OutputPath')])
    spec = _FakeComponentSpec(input=input_channel, output=output_channel)
    super(_FakeComponent, self).__init__(spec=spec, name=name)


class ComponentRunnerTest(tf.test.TestCase):

  @mock.patch.object(publisher, 'Publisher')
  def testRun(self, mock_publisher):
    mock_publisher.return_value.publish_execution.return_value = {}

    test_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    connection_config = metadata_store_pb2.ConnectionConfig()
    connection_config.sqlite.SetInParent()

    pipeline_root = os.path.join(test_dir, 'Test')
    input_path = os.path.join(test_dir, 'input')
    tf.gfile.MakeDirs(os.path.dirname(input_path))
    file_io.write_string_to_file(input_path, 'test')

    input_artifact = types.Artifact(type_name='InputPath')
    input_artifact.uri = input_path

    component = _FakeComponent(
        name='FakeComponent',
        input_channel=channel_utils.as_channel([input_artifact]))

    pipeline_info = data_types.PipelineInfo(
        pipeline_name='Test', pipeline_root=pipeline_root, run_id='123')

    driver_args = data_types.DriverArgs(enable_cache=True)

    launcher = component_launcher.ComponentLauncher(
        component=component,
        pipeline_info=pipeline_info,
        driver_args=driver_args,
        metadata_connection_config=connection_config,
        additional_pipeline_args={})
    self.assertEqual(
        launcher._component_info.component_type,
        '.'.join([_FakeComponent.__module__, _FakeComponent.__name__]))
    launcher.launch()

    output_path = os.path.join(pipeline_root, 'output')
    self.assertTrue(tf.gfile.Exists(output_path))
    contents = file_io.read_file_to_string(output_path)
    self.assertEqual('test', contents)


if __name__ == '__main__':
  tf.test.main()
