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
"""Tests for tfx.extensions.google_cloud_ai_platform.trainer.executor."""

import copy
import os
from typing import Any, Dict
from unittest import mock


import tensorflow as tf
from tfx.components.trainer import executor as tfx_trainer_executor
from tfx.extensions.google_cloud_ai_platform.trainer import executor as ai_platform_trainer_executor
from tfx.types import standard_component_specs
from tfx.utils import json_utils


class ExecutorTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()

    self._output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    self._job_dir = os.path.join(self._output_data_dir, 'jobDir')
    self._project_id = '12345'
    self._inputs = {}
    self._outputs = {}
    # Dict format of exec_properties. custom_config needs to be serialized
    # before being passed into Do function.
    self._exec_properties = {
        standard_component_specs.CUSTOM_CONFIG_KEY: {
            ai_platform_trainer_executor.TRAINING_ARGS_KEY: {
                'project': self._project_id,
                'jobDir': self._job_dir,
            },
        },
    }
    self._executor_class_path = '%s.%s' % (
        tfx_trainer_executor.Executor.__module__,
        tfx_trainer_executor.Executor.__name__)
    self._generic_executor_class_path = '%s.%s' % (
        tfx_trainer_executor.GenericExecutor.__module__,
        tfx_trainer_executor.GenericExecutor.__name__)

    self.addCleanup(mock.patch.stopall)
    self.mock_runner = mock.patch(
        'tfx.extensions.google_cloud_ai_platform.trainer.executor.runner'
    ).start()

  def _serialize_custom_config_under_test(self) -> Dict[str, Any]:
    """Converts self._exec_properties['custom_config'] to string."""
    result = copy.deepcopy(self._exec_properties)
    result[standard_component_specs.CUSTOM_CONFIG_KEY] = json_utils.dumps(
        result[standard_component_specs.CUSTOM_CONFIG_KEY])
    return result

  def testDo(self):
    executor = ai_platform_trainer_executor.Executor()
    executor.Do(self._inputs, self._outputs,
                self._serialize_custom_config_under_test())
    self.mock_runner.start_cloud_training.assert_called_with(
        self._inputs, self._outputs, self._serialize_custom_config_under_test(),
        self._executor_class_path, {
            'project': self._project_id,
            'jobDir': self._job_dir,
        }, None, False, None)

  def testDoWithJobIdOverride(self):
    executor = ai_platform_trainer_executor.Executor()
    job_id = 'overridden_job_id'
    self._exec_properties[standard_component_specs.CUSTOM_CONFIG_KEY][
        ai_platform_trainer_executor.JOB_ID_KEY] = job_id
    executor.Do(self._inputs, self._outputs,
                self._serialize_custom_config_under_test())
    self.mock_runner.start_cloud_training.assert_called_with(
        self._inputs, self._outputs, self._serialize_custom_config_under_test(),
        self._executor_class_path, {
            'project': self._project_id,
            'jobDir': self._job_dir,
        }, job_id, False, None)

  def testDoWithGenericExecutorClass(self):
    executor = ai_platform_trainer_executor.GenericExecutor()
    executor.Do(self._inputs, self._outputs,
                self._serialize_custom_config_under_test())
    self.mock_runner.start_cloud_training.assert_called_with(
        self._inputs, self._outputs, self._serialize_custom_config_under_test(),
        self._generic_executor_class_path, {
            'project': self._project_id,
            'jobDir': self._job_dir,
        }, None, False, None)

  def testDoWithEnableVertexOverride(self):
    executor = ai_platform_trainer_executor.Executor()
    enable_vertex = True
    vertex_region = 'us-central2'
    self._exec_properties[standard_component_specs.CUSTOM_CONFIG_KEY][
        ai_platform_trainer_executor.ENABLE_VERTEX_KEY] = enable_vertex
    self._exec_properties[standard_component_specs.CUSTOM_CONFIG_KEY][
        ai_platform_trainer_executor.VERTEX_REGION_KEY] = vertex_region
    executor.Do(self._inputs, self._outputs,
                self._serialize_custom_config_under_test())
    self.mock_runner.start_cloud_training.assert_called_with(
        self._inputs, self._outputs, self._serialize_custom_config_under_test(),
        self._executor_class_path, {
            'project': self._project_id,
            'jobDir': self._job_dir,
        }, None, enable_vertex, vertex_region)


if __name__ == '__main__':
  tf.test.main()
