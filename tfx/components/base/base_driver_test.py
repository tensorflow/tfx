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
from ml_metadata.proto import metadata_store_pb2
from tfx import types
from tfx.components.base import base_driver
from tfx.orchestration import data_types
from tfx.types import channel
from tfx.types import channel_utils


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
                producer_info=channel.ChannelProducerInfo(
                    component_id='c', key='k'))
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
            types.Channel(type=_OutputArtifact, artifacts=[_OutputArtifact()])
    }
    self._input_artifacts = channel_utils.unwrap_channel_dict(self._input_dict)
    self._output_artifacts = channel_utils.unwrap_channel_dict(
        self._output_dict)
    self._exec_properties = {
        'key': 'value',
    }
    self._execution_id = 100
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
      'tfx.components.base.base_driver.BaseDriver.verify_input_artifacts'
  )
  def testPreExecutionNewExecution(self, mock_verify_input_artifacts_fn):
    self._mock_metadata.get_artifacts_by_info.side_effect = list(
        self._input_dict['input_data'].get())
    self._mock_metadata.register_execution.side_effect = [self._execution_id]
    self._mock_metadata.previous_execution.side_effect = [None]
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

  @mock.patch(
      'tfx.components.base.base_driver.BaseDriver.verify_input_artifacts'
  )
  def testPreExecutionCached(self, mock_verify_input_artifacts_fn):
    self._mock_metadata.get_artifacts_by_info.side_effect = list(
        self._input_dict['input_data'].get())
    self._mock_metadata.register_execution.side_effect = [self._execution_id]
    self._mock_metadata.previous_execution.side_effect = [2]
    self._mock_metadata.register_run_context_if_not_exists.side_effect = [
        metadata_store_pb2.Context()
    ]
    self._mock_metadata.fetch_previous_result_artifacts.side_effect = [
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
