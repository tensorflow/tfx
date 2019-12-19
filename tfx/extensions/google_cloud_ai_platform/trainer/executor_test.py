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

import os
# Standard Imports
import mock
import tensorflow as tf

from tfx.components.trainer import executor as tfx_trainer_executor
from tfx.extensions.google_cloud_ai_platform.trainer.executor import Executor


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
    self._exec_properties = {
        'custom_config': {
            'ai_platform_training_args': {
                'project': self._project_id,
                'jobDir': self._job_dir,
            },
        },
    }
    self._executor_class_path = '%s.%s' % (
        tfx_trainer_executor.Executor.__module__,
        tfx_trainer_executor.Executor.__name__)

  @mock.patch(
      'tfx.extensions.google_cloud_ai_platform.trainer.executor.runner'
  )
  def testDo(self, mock_runner):
    executor = Executor()
    executor.Do(self._inputs, self._outputs, self._exec_properties)
    mock_runner.start_aip_training.assert_called_with(
        self._inputs, self._outputs, self._exec_properties,
        self._executor_class_path, {
            'project': self._project_id,
            'jobDir': self._job_dir,
        }, None)

  @mock.patch(
      'tfx.extensions.google_cloud_ai_platform.trainer.executor.runner'
  )
  def testDoWithJobIdOverride(self, mock_runner):
    executor = Executor()
    job_id = 'overridden_job_id'
    self._exec_properties['custom_config'][
        'ai_platform_training_job_id'] = job_id
    executor.Do(self._inputs, self._outputs, self._exec_properties)
    mock_runner.start_aip_training.assert_called_with(
        self._inputs, self._outputs, self._exec_properties,
        self._executor_class_path, {
            'project': self._project_id,
            'jobDir': self._job_dir,
        }, job_id)


if __name__ == '__main__':
  tf.test.main()
