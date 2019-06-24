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
"""Tests for tfx.orchestration.component_runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from typing import Any, Dict, List, Optional, Text
from tensorflow.python.lib.io import file_io  # pylint: disable=g-direct-tensorflow-import
from tfx.components.base import base_component
from tfx.components.base import base_executor
from tfx.orchestration import component_runner
from tfx.utils import channel
from tfx.utils import types


class _FakeExecutor(base_executor.BaseExecutor):

  def Do(self, input_dict: Dict[Text, List[types.TfxArtifact]],
         output_dict: Dict[Text, List[types.TfxArtifact]],
         exec_properties: Dict[Text, Any]) -> None:
    input_path = types.get_single_uri(input_dict['input'])
    output_path = types.get_single_uri(output_dict['output'])
    tf.gfile.Copy(input_path, output_path)


class _FakeComponentSpec(base_component.ComponentSpec):
  COMPONENT_NAME = 'fake_component'
  PARAMETERS = {}
  INPUTS = {'input': base_component.ChannelParameter(type_name='InputPath')}
  OUTPUTS = {'output': base_component.ChannelParameter(type_name='OutputPath')}


class _FakeComponent(base_component.BaseComponent):
  SPEC_CLASS = _FakeComponentSpec
  EXECUTOR_CLASS = _FakeExecutor

  def __init__(self,
               name: Text,
               input_channel: channel.Channel,
               output_channel: Optional[channel.Channel] = None):
    output_channel = output_channel or channel.Channel(
        type_name='OutputPath', artifacts=[types.TfxArtifact('OutputPath')])
    spec = _FakeComponentSpec(input=input_channel, output=output_channel)
    super(_FakeComponent, self).__init__(spec=spec, name=name)


class ComponentRunnerTest(tf.test.TestCase):

  def test_run(self):
    test_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    input_path = os.path.join(test_dir, 'input')
    output_path = os.path.join(test_dir, 'output')
    tf.gfile.MakeDirs(os.path.dirname(input_path))
    file_io.write_string_to_file(input_path, 'test')

    input_artifact = types.TfxArtifact(type_name='InputPath')
    input_artifact.uri = input_path

    component = _FakeComponent(
        name='FakeComponent',
        input_channel=channel.as_channel([input_artifact]))

    # TODO(jyzhao): remove after driver is supported.
    types.get_single_instance(
        component.outputs.get_all()['output'].get()).uri = output_path

    component_runner.ComponentRunner(
        component=component,
        pipeline_run_id=123,
        pipeline_name='Test',
        pipeline_root=test_dir).run()

    self.assertTrue(tf.gfile.Exists(output_path))
    contents = file_io.read_file_to_string(output_path)
    self.assertEqual('test', contents)


if __name__ == '__main__':
  tf.test.main()
