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
"""Tests for tfx.extensions.google_cloud_ai_platform.trainer.executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
from typing import Any, Dict, Text

# Standard Imports

import mock
import tensorflow as tf

from tfx.components.trainer import executor as tfx_trainer_executor
from tfx.extensions.google_cloud_ai_platform.trainer import executor as ai_platform_trainer_executor
from tfx.utils import json_utils


class ExecutorTest(tf.test.TestCase):

  def setUp(self):
    super(ExecutorTest, self).setUp()

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
        'custom_config': {
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

  def _serialize_custom_config_under_test(self) -> Dict[Text, Any]:
    """Converts self._exec_properties['custom_config'] to string."""
    result = copy.deepcopy(self._exec_properties)
    result['custom_config'] = json_utils.dumps(result['custom_config'])
    return result

  def testDo(self):
    executor = ai_platform_trainer_executor.Executor()
    executor.Do(self._inputs, self._outputs,
                self._serialize_custom_config_under_test())
    self.mock_runner.start_aip_training.assert_called_with(
        self._inputs, self._outputs, self._serialize_custom_config_under_test(),
        self._executor_class_path, {
            'project': self._project_id,
            'jobDir': self._job_dir,
        }, None)

  def testDoWithJobIdOverride(self):
    executor = ai_platform_trainer_executor.Executor()
    job_id = 'overridden_job_id'
    self._exec_properties['custom_config'][
        ai_platform_trainer_executor.JOB_ID_KEY] = job_id
    executor.Do(self._inputs, self._outputs,
                self._serialize_custom_config_under_test())
    self.mock_runner.start_aip_training.assert_called_with(
        self._inputs, self._outputs, self._serialize_custom_config_under_test(),
        self._executor_class_path, {
            'project': self._project_id,
            'jobDir': self._job_dir,
        }, job_id)

  def testDoWithGenericExecutorClass(self):
    executor = ai_platform_trainer_executor.GenericExecutor()
    executor.Do(self._inputs, self._outputs,
                self._serialize_custom_config_under_test())
    self.mock_runner.start_aip_training.assert_called_with(
        self._inputs, self._outputs, self._serialize_custom_config_under_test(),
        self._generic_executor_class_path, {
            'project': self._project_id,
            'jobDir': self._job_dir,
        }, None)


if __name__ == '__main__':
  tf.test.main()
