# Copyright 2020 Google LLC. All Rights Reserved.
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
"""Tests for tfx.extensions.google_cloud_kubernetes.trainer.executor."""


import copy
import os
from typing import Any, Dict, Text

import mock
import tensorflow as tf

from tfx.components.trainer import executor as tfx_trainer_executor
from tfx.extensions.google_cloud_kubernetes.trainer import executor as gke_trainer_executor
from tfx.utils import json_utils


class ExecutorTest(tf.test.TestCase):

  def setUp(self):
    super(ExecutorTest, self).setUp()

    self._output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    self._job_dir = os.path.join(self._output_data_dir, 'jobDir')
    self._num_workers = 2
    self._num_gpus_per_worker = 1
    self._inputs = {}
    self._outputs = {}
    self._unique_id = 'UNIQUE_ID'
    # Dict format of exec_properties. custom_config needs to be serialized
    # before being passed into Do function.
    self._exec_properties = {
        'custom_config': {
            gke_trainer_executor.TRAINING_ARGS_KEY: {
                'num_workers': self._num_workers,
                'num_gpus_per_worker': self._num_gpus_per_worker,
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
        'tfx.extensions.google_cloud_kubernetes.trainer.executor.runner'
    ).start()

  def _serialize_custom_config_under_test(self) -> Dict[Text, Any]:
    """Converts self._exec_properties['custom_config'] to string."""
    result = copy.deepcopy(self._exec_properties)
    result['custom_config'] = json_utils.dumps(result['custom_config'])
    return result

  def testDo(self):
    executor = gke_trainer_executor.Executor(
        gke_trainer_executor.Executor.Context(unique_id=self._unique_id)
    )
    executor.Do(self._inputs, self._outputs,
                self._serialize_custom_config_under_test())
    self.mock_runner.start_gke_training.assert_called_with(
        self._inputs, self._outputs, self._serialize_custom_config_under_test(),
        self._executor_class_path, {
            'num_gpus_per_worker': self._num_gpus_per_worker,
            'num_workers': self._num_workers,
        }, self._unique_id)

  def testDoWithGenericExecutorClass(self):
    executor = gke_trainer_executor.GenericExecutor(
        tfx_trainer_executor.GenericExecutor.Context(unique_id=self._unique_id)
    )
    executor.Do(self._inputs, self._outputs,
                self._serialize_custom_config_under_test())
    self.mock_runner.start_gke_training.assert_called_with(
        self._inputs, self._outputs, self._serialize_custom_config_under_test(),
        self._generic_executor_class_path, {
            'num_gpus_per_worker': self._num_gpus_per_worker,
            'num_workers': self._num_workers,
        }, self._unique_id)


if __name__ == '__main__':
  tf.test.main()
