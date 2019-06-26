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
import tensorflow as tf
from tfx.components.base import base_driver
from tfx.orchestration import data_types
from tfx.utils import channel
from tfx.utils import types


class BaseDriverTest(tf.test.TestCase):

  def setUp(self):
    self._mock_metadata = tf.test.mock.Mock()
    self._input_dict = {
        'input_data': [types.TfxArtifact(type_name='InputType')],
    }
    input_dir = os.path.join(
        os.environ.get('TEST_TMP_DIR', self.get_temp_dir()),
        self._testMethodName, 'input_dir')
    # valid input artifacts must have a uri pointing to an existing directory.
    for key, input_list in self._input_dict.items():
      for index, artifact in enumerate(input_list):
        artifact.id = index + 1
        uri = os.path.join(input_dir, key, str(artifact.id), '')
        artifact.uri = uri
        tf.gfile.MakeDirs(uri)
    self._output_dict = {
        'output_data': [types.TfxArtifact(type_name='OutputType')],
    }
    self._exec_properties = {
        'key': 'value',
    }
    self._base_output_dir = os.path.join(
        os.environ.get('TEST_TMP_DIR', self.get_temp_dir()),
        self._testMethodName, 'base_output_dir')
    self._driver_args = data_types.DriverArgs(
        worker_name='worker_name',
        base_output_dir=self._base_output_dir,
        enable_cache=True)
    self._execution_id = 100

  def _check_output(self, execution_decision):
    output_dict = execution_decision.output_dict
    self.assertEqual(self._output_dict.keys(), output_dict.keys())
    for name, output_list in output_dict.items():
      for (original_output, output) in zip(self._output_dict[name],
                                           output_list):
        if execution_decision.execution_id:
          # Uncached results should have a newly created uri.
          self.assertEqual(
              os.path.join(self._base_output_dir, name,
                           str(execution_decision.execution_id), ''),
              output.uri)
        else:
          # Cached results have a different set of uri.
          self.assertEqual(
              os.path.join(self._base_output_dir, name, str(self._execution_id),
                           ''), output.uri)
        self.assertEqual(original_output.split, output.split)

  def test_prepare_execution(self):
    input_dict = copy.deepcopy(self._input_dict)
    output_dict = copy.deepcopy(self._output_dict)
    exec_properties = copy.deepcopy(self._exec_properties)

    self._mock_metadata.previous_run.return_value = None
    self._mock_metadata.prepare_execution.return_value = self._execution_id
    driver = base_driver.BaseDriver(metadata_handler=self._mock_metadata)
    execution_decision = driver.prepare_execution(input_dict, output_dict,
                                                  exec_properties,
                                                  self._driver_args)
    self.assertEqual(self._execution_id, execution_decision.execution_id)
    self._check_output(execution_decision)

  def test_cached_execution(self):
    input_dict = copy.deepcopy(self._input_dict)
    output_dict = copy.deepcopy(self._output_dict)
    exec_properties = copy.deepcopy(self._exec_properties)

    cached_output_dict = copy.deepcopy(self._output_dict)
    for key, artifact_list in cached_output_dict.items():
      for artifact in artifact_list:
        artifact.uri = os.path.join(self._base_output_dir, key,
                                    str(self._execution_id), '')
        # valid cached artifacts must have an existing uri.
        tf.gfile.MakeDirs(artifact.uri)
    self._mock_metadata.previous_run.return_value = self._execution_id
    self._mock_metadata.fetch_previous_result_artifacts.return_value = cached_output_dict
    driver = base_driver.BaseDriver(metadata_handler=self._mock_metadata)
    execution_decision = driver.prepare_execution(input_dict, output_dict,
                                                  exec_properties,
                                                  self._driver_args)
    self.assertIsNone(execution_decision.execution_id)
    self._check_output(execution_decision)

  def test_artifact_missing(self):
    input_dict = copy.deepcopy(self._input_dict)
    input_dict['input_data'][0].uri = 'should/not/exist'
    output_dict = copy.deepcopy(self._output_dict)
    exec_properties = copy.deepcopy(self._exec_properties)
    driver_options = copy.deepcopy(self._driver_args)
    driver_options.enable_cache = False

    cached_output_dict = copy.deepcopy(self._output_dict)
    for key, artifact_list in cached_output_dict.items():
      for artifact in artifact_list:
        artifact.uri = os.path.join(self._base_output_dir, key,
                                    str(self._execution_id), '')
        # valid cached artifacts must have an existing uri.
        tf.gfile.MakeDirs(artifact.uri)

    self._mock_metadata.previous_run.return_value = self._execution_id
    self._mock_metadata.fetch_previous_result_artifacts.return_value = cached_output_dict
    driver = base_driver.BaseDriver(self._mock_metadata)
    with self.assertRaises(RuntimeError):
      driver.prepare_execution(input_dict, output_dict, exec_properties,
                               driver_options)

  def test_no_cache_on_missing_uri(self):
    input_dict = copy.deepcopy(self._input_dict)
    output_dict = copy.deepcopy(self._output_dict)
    exec_properties = copy.deepcopy(self._exec_properties)

    cached_output_dict = copy.deepcopy(self._output_dict)
    for key, artifact_list in cached_output_dict.items():
      for artifact in artifact_list:
        artifact.uri = os.path.join(self._base_output_dir, key,
                                    str(self._execution_id), '')
        # Non existing output uri will force a cache miss.
        self.assertFalse(tf.gfile.Exists(artifact.uri))
    self._mock_metadata.previous_run.return_value = self._execution_id
    self._mock_metadata.fetch_previous_result_artifacts.return_value = cached_output_dict
    actual_execution_id = self._execution_id + 1
    self._mock_metadata.prepare_execution.return_value = actual_execution_id

    driver = base_driver.BaseDriver(metadata_handler=self._mock_metadata)
    execution_decision = driver.prepare_execution(input_dict, output_dict,
                                                  exec_properties,
                                                  self._driver_args)
    self.assertEqual(actual_execution_id, execution_decision.execution_id)
    self._check_output(execution_decision)

  def test_pre_execution_new_execution(self):
    input_dict = {
        'input_a':
            channel.Channel(
                type_name='input_a',
                artifacts=[types.TfxArtifact(type_name='input_a')])
    }
    output_dict = {
        'output_a':
            channel.Channel(
                type_name='output_a',
                artifacts=[
                    types.TfxArtifact(type_name='output_a', split='split')
                ])
    }
    execution_id = 1
    exec_properties = copy.deepcopy(self._exec_properties)
    driver_args = data_types.DriverArgs(
        worker_name='worker_name', base_output_dir='base', enable_cache=True)
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
    self.assertItemsEqual(execution_decision.exec_properties, exec_properties)
    self.assertEqual(
        execution_decision.output_dict['output_a'][0].uri,
        os.path.join(pipeline_info.pipeline_root, component_info.component_id,
                     'output_a', str(execution_id), 'split', ''))

  def test_pre_execution_cached(self):
    input_dict = {
        'input_a':
            channel.Channel(
                type_name='input_a',
                artifacts=[types.TfxArtifact(type_name='input_a')])
    }
    output_dict = {
        'output_a':
            channel.Channel(
                type_name='output_a',
                artifacts=[
                    types.TfxArtifact(type_name='output_a', split='split')
                ])
    }
    execution_id = 1
    exec_properties = copy.deepcopy(self._exec_properties)
    driver_args = data_types.DriverArgs(
        worker_name='worker_name', base_output_dir='base', enable_cache=True)
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
    self._mock_metadata.fetch_previous_result_artifacts.side_effect = [
        self._output_dict
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
    self.assertItemsEqual(execution_decision.exec_properties, exec_properties)
    self.assertItemsEqual(execution_decision.output_dict, self._output_dict)


if __name__ == '__main__':
  tf.test.main()
