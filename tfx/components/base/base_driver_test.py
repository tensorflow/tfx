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

import copy
import os
import mock
import tensorflow as tf
from tfx import types
from tfx.components.base import base_driver
from tfx.orchestration import data_types
from tfx.types import channel_utils


class BaseDriverTest(tf.test.TestCase):

  def setUp(self):
    super(BaseDriverTest, self).setUp()
    self._mock_metadata = tf.compat.v1.test.mock.Mock()
    self._input_dict = {
        'input_data':
            types.Channel(
                type_name='input_data',
                artifacts=[types.Artifact(type_name='input_data')])
    }
    input_dir = os.path.join(
        os.environ.get('TEST_TMP_DIR', self.get_temp_dir()),
        self._testMethodName, 'input_dir')
    # valid input artifacts must have a uri pointing to an existing directory.
    for key, input_channel in self._input_dict.items():
      for index, artifact in enumerate(input_channel.get()):
        artifact.id = index + 1
        uri = os.path.join(input_dir, key, str(artifact.id), '')
        artifact.uri = uri
        tf.io.gfile.makedirs(uri)
    self._output_dict = {
        'output_data':
            types.Channel(
                type_name='output_data',
                artifacts=[
                    types.Artifact(type_name='output_data', split='split')
                ])
    }
    self._input_artifacts = channel_utils.unwrap_channel_dict(self._input_dict)
    self._output_artifacts = {
        'output_data': [types.Artifact(type_name='OutputType')],
    }
    self._exec_properties = {
        'key': 'value',
    }
    self._execution_id = 100

  @mock.patch(
      'tfx.components.base.base_driver.BaseDriver.verify_input_artifacts'
  )
  def testPreExecutionNewExecution(self, mock_verify_input_artifacts_fn):
    input_dict = {
        'input_a':
            types.Channel(
                type_name='input_a',
                artifacts=[types.Artifact(type_name='input_a')])
    }
    output_dict = {
        'output_a':
            types.Channel(
                type_name='output_a',
                artifacts=[types.Artifact(type_name='output_a', split='split')])
    }
    execution_id = 1
    context_id = 123
    exec_properties = copy.deepcopy(self._exec_properties)
    driver_args = data_types.DriverArgs(enable_cache=True)
    pipeline_info = data_types.PipelineInfo(
        pipeline_name='my_pipeline_name',
        pipeline_root=os.environ.get('TEST_TMP_DIR', self.get_temp_dir()),
        run_id='my_run_id')
    component_info = data_types.ComponentInfo(
        component_type='a.b.c', component_id='my_component_id')
    self._mock_metadata.get_artifacts_by_info.side_effect = list(
        input_dict['input_a'].get())
    self._mock_metadata.register_execution.side_effect = [execution_id]
    self._mock_metadata.previous_execution.side_effect = [None]
    self._mock_metadata.register_run_context_if_not_exists.side_effect = [
        context_id
    ]

    driver = base_driver.BaseDriver(metadata_handler=self._mock_metadata)
    execution_decision = driver.pre_execution(
        input_dict=input_dict,
        output_dict=output_dict,
        exec_properties=exec_properties,
        driver_args=driver_args,
        pipeline_info=pipeline_info,
        component_info=component_info)
    self.assertFalse(execution_decision.use_cached_results)
    self.assertEqual(execution_decision.execution_id, 1)
    self.assertCountEqual(execution_decision.exec_properties, exec_properties)
    self.assertEqual(
        execution_decision.output_dict['output_a'][0].uri,
        os.path.join(pipeline_info.pipeline_root, component_info.component_id,
                     'output_a', str(execution_id), 'split', ''))

  @mock.patch(
      'tfx.components.base.base_driver.BaseDriver.verify_input_artifacts'
  )
  def testPreExecutionCached(self, mock_verify_input_artifacts_fn):
    input_dict = {
        'input_a':
            types.Channel(
                type_name='input_a',
                artifacts=[types.Artifact(type_name='input_a')])
    }
    output_dict = {
        'output_a':
            types.Channel(
                type_name='output_a',
                artifacts=[types.Artifact(type_name='output_a', split='split')])
    }
    execution_id = 1
    context_id = 123
    exec_properties = copy.deepcopy(self._exec_properties)
    driver_args = data_types.DriverArgs(enable_cache=True)
    pipeline_info = data_types.PipelineInfo(
        pipeline_name='my_pipeline_name',
        pipeline_root=os.environ.get('TEST_TMP_DIR', self.get_temp_dir()),
        run_id='my_run_id')
    component_info = data_types.ComponentInfo(
        component_type='a.b.c', component_id='my_component_id')
    self._mock_metadata.get_artifacts_by_info.side_effect = list(
        input_dict['input_a'].get())
    self._mock_metadata.register_execution.side_effect = [execution_id]
    self._mock_metadata.previous_execution.side_effect = [2]
    self._mock_metadata.register_run_context_if_not_exists.side_effect = [
        context_id
    ]
    self._mock_metadata.fetch_previous_result_artifacts.side_effect = [
        self._output_artifacts
    ]

    driver = base_driver.BaseDriver(metadata_handler=self._mock_metadata)
    execution_decision = driver.pre_execution(
        input_dict=input_dict,
        output_dict=output_dict,
        exec_properties=exec_properties,
        driver_args=driver_args,
        pipeline_info=pipeline_info,
        component_info=component_info)
    self.assertTrue(execution_decision.use_cached_results)
    self.assertEqual(execution_decision.execution_id, 1)
    self.assertCountEqual(execution_decision.exec_properties, exec_properties)
    self.assertCountEqual(execution_decision.output_dict,
                          self._output_artifacts)

  def testVerifyInputArtifactsOk(self):
    driver = base_driver.BaseDriver(metadata_handler=self._mock_metadata)
    driver.verify_input_artifacts(self._input_artifacts)

  def testVerifyInputArtifactsNotExists(self):
    driver = base_driver.BaseDriver(metadata_handler=self._mock_metadata)
    with self.assertRaises(RuntimeError):
      driver.verify_input_artifacts(
          {'artifact': [types.Artifact(type_name='input_data')]})


if __name__ == '__main__':
  tf.test.main()
