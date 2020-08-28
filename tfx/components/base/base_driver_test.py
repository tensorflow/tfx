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
"""Tests for tfx.components.base.base_driver."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import mock
import tensorflow as tf
from tfx import types
from tfx.components.base import base_driver
from tfx.orchestration import data_types
from tfx.types import channel_utils
from tfx.types import standard_artifacts
from ml_metadata.proto import metadata_store_pb2

# Mock value for string artifact.
_STRING_VALUE = u'This is a string'

# Mock byte value for string artifact.
_BYTE_VALUE = b'This is a string'


def fake_read(self):
  """Mock read method for ValueArtifact."""
  if not self._has_value:
    self._has_value = True
    self._value = self.decode(_BYTE_VALUE)
  return self._value


class _InputArtifact(types.Artifact):
  TYPE_NAME = 'InputArtifact'


class _OutputArtifact(types.Artifact):
  TYPE_NAME = 'OutputArtifact'


class BaseDriverTest(tf.test.TestCase):

  def setUp(self):
    super(BaseDriverTest, self).setUp()
    self._mock_metadata = tf.compat.v1.test.mock.Mock()
    self._input_dict = {
        'input_data':
            types.Channel(
                type=_InputArtifact,
                artifacts=[_InputArtifact()],
                producer_component_id='c',
                output_key='k'),
        'input_string':
            types.Channel(
                type=standard_artifacts.String,
                artifacts=[
                    standard_artifacts.String(),
                    standard_artifacts.String()
                ],
                producer_component_id='c2',
                output_key='k2'),
    }
    input_dir = os.path.join(
        os.environ.get('TEST_TMP_DIR', self.get_temp_dir()),
        self._testMethodName, 'input_dir')
    # valid input artifacts must have a uri pointing to an existing directory.
    for key, input_channel in self._input_dict.items():
      for index, artifact in enumerate(input_channel.get()):
        artifact.id = index + 1
        uri = os.path.join(input_dir, key, str(artifact.id))
        artifact.uri = uri
        tf.io.gfile.makedirs(uri)
    self._output_dict = {
        'output_data':
            types.Channel(type=_OutputArtifact, artifacts=[_OutputArtifact()]),
        'output_multi_data':
            types.Channel(
                type=_OutputArtifact, matching_channel_name='input_string')
    }
    self._input_artifacts = channel_utils.unwrap_channel_dict(self._input_dict)
    self._output_artifacts = channel_utils.unwrap_channel_dict(
        self._output_dict)
    self._exec_properties = {
        'key': 'value',
    }
    self._execution_id = 100
    self._execution = metadata_store_pb2.Execution()
    self._execution.id = self._execution_id
    self._context_id = 123
    self._driver_args = data_types.DriverArgs(enable_cache=True)
    self._pipeline_info = data_types.PipelineInfo(
        pipeline_name='my_pipeline_name',
        pipeline_root=os.environ.get('TEST_TMP_DIR', self.get_temp_dir()),
        run_id='my_run_id')
    self._component_info = data_types.ComponentInfo(
        component_type='a.b.c',
        component_id='my_component_id',
        pipeline_info=self._pipeline_info)

  @mock.patch(
      'tfx.components.base.base_driver.BaseDriver.verify_input_artifacts')
  @mock.patch.object(types.ValueArtifact, 'read', fake_read)
  def testPreExecutionNewExecution(self, mock_verify_input_artifacts_fn):
    self._mock_metadata.search_artifacts.return_value = list(
        self._input_dict['input_string'].get())
    self._mock_metadata.register_execution.side_effect = [self._execution]
    self._mock_metadata.get_cached_outputs.side_effect = [None]
    self._mock_metadata.register_run_context_if_not_exists.side_effect = [
        metadata_store_pb2.Context()
    ]

    driver = base_driver.BaseDriver(metadata_handler=self._mock_metadata)
    execution_decision = driver.pre_execution(
        input_dict=self._input_dict,
        output_dict=self._output_dict,
        exec_properties=self._exec_properties,
        driver_args=self._driver_args,
        pipeline_info=self._pipeline_info,
        component_info=self._component_info)
    self.assertFalse(execution_decision.use_cached_results)
    self.assertEqual(execution_decision.execution_id, self._execution_id)
    self.assertCountEqual(execution_decision.exec_properties,
                          self._exec_properties)
    self.assertEqual(
        execution_decision.output_dict['output_data'][0].uri,
        os.path.join(self._pipeline_info.pipeline_root,
                     self._component_info.component_id, 'output_data',
                     str(self._execution_id)))
    self.assertLen(execution_decision.output_dict['output_multi_data'], 2)
    for i in range(2):
      self.assertEqual(
          execution_decision.output_dict['output_multi_data'][i].uri,
          os.path.join(self._pipeline_info.pipeline_root,
                       self._component_info.component_id, 'output_multi_data',
                       str(self._execution_id), str(i)))
    self.assertEqual(execution_decision.input_dict['input_string'][0].value,
                     _STRING_VALUE)

  @mock.patch(
      'tfx.components.base.base_driver.BaseDriver.verify_input_artifacts')
  @mock.patch.object(types.ValueArtifact, 'read', fake_read)
  def testPreExecutionCached(self, mock_verify_input_artifacts_fn):
    self._mock_metadata.search_artifacts.return_value = list(
        self._input_dict['input_string'].get())
    self._mock_metadata.register_run_context_if_not_exists.side_effect = [
        metadata_store_pb2.Context()
    ]
    self._mock_metadata.register_execution.side_effect = [self._execution]
    self._mock_metadata.get_cached_outputs.side_effect = [
        self._output_artifacts
    ]

    driver = base_driver.BaseDriver(metadata_handler=self._mock_metadata)
    execution_decision = driver.pre_execution(
        input_dict=self._input_dict,
        output_dict=self._output_dict,
        exec_properties=self._exec_properties,
        driver_args=self._driver_args,
        pipeline_info=self._pipeline_info,
        component_info=self._component_info)
    self.assertTrue(execution_decision.use_cached_results)
    self.assertEqual(execution_decision.execution_id, self._execution_id)
    self.assertCountEqual(execution_decision.exec_properties,
                          self._exec_properties)
    self.assertCountEqual(execution_decision.output_dict,
                          self._output_artifacts)

  def testVerifyInputArtifactsOk(self):
    driver = base_driver.BaseDriver(metadata_handler=self._mock_metadata)
    driver.verify_input_artifacts(self._input_artifacts)

  def testVerifyInputArtifactsNotExists(self):
    driver = base_driver.BaseDriver(metadata_handler=self._mock_metadata)
    with self.assertRaises(RuntimeError):
      driver.verify_input_artifacts({'artifact': [_InputArtifact()]})


if __name__ == '__main__':
  tf.test.main()
