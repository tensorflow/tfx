# Lint as: python3
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
"""Tests for Cloud AI Platform Tuner Executor."""

import copy
import os

from typing import Any, Dict, Text
import mock
import tensorflow as tf

from tfx.extensions.google_cloud_ai_platform.trainer import executor as ai_platform_trainer_executor
from tfx.extensions.google_cloud_ai_platform.tuner import executor as ai_platform_tuner_executor
from tfx.proto import tuner_pb2
from tfx.utils import json_utils
from google.protobuf import json_format


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
        ai_platform_tuner_executor._WorkerExecutor.__module__,
        ai_platform_tuner_executor._WorkerExecutor.__name__)

    self.addCleanup(mock.patch.stopall)
    self.mock_runner = mock.patch(
        'tfx.extensions.google_cloud_ai_platform.tuner.executor.runner').start(
        )

  def _serialize_custom_config_under_test(self) -> Dict[Text, Any]:
    """Converts self._exec_properties['custom_config'] to string."""
    result = copy.deepcopy(self._exec_properties)
    result['custom_config'] = json_utils.dumps(result['custom_config'])
    return result

  def testDo(self):
    executor = ai_platform_tuner_executor.Executor()
    executor.Do(self._inputs, self._outputs,
                self._serialize_custom_config_under_test())
    self.mock_runner.start_aip_training.assert_called_with(
        self._inputs, self._outputs, self._serialize_custom_config_under_test(),
        self._executor_class_path, {
            'project': self._project_id,
            'jobDir': self._job_dir,
        }, mock.ANY)

  def testDoWithTuneArgs(self):
    executor = ai_platform_tuner_executor.Executor()
    self._exec_properties['tune_args'] = json_format.MessageToJson(
        message=tuner_pb2.TuneArgs(num_parallel_trials=3),
        preserving_proto_field_name=True)

    executor.Do(self._inputs, self._outputs,
                self._serialize_custom_config_under_test())

    self.mock_runner.start_aip_training.assert_called_with(
        self._inputs, self._outputs, self._serialize_custom_config_under_test(),
        self._executor_class_path, {
            'project': self._project_id,
            'jobDir': self._job_dir,
            'scaleTier': 'CUSTOM',
            'masterType': 'standard',
            'workerType': 'standard',
            'workerCount': 2,
        }, mock.ANY)

  def testDoWithTuneArgsAndTrainingInputOverride(self):
    executor = ai_platform_tuner_executor.Executor()
    self._exec_properties['tune_args'] = json_format.MessageToJson(
        message=tuner_pb2.TuneArgs(num_parallel_trials=6),
        preserving_proto_field_name=True)

    self._exec_properties['custom_config'][
        ai_platform_trainer_executor.TRAINING_ARGS_KEY].update({
            'scaleTier': 'CUSTOM',
            'masterType': 'n1-highmem-16',
            'workerType': 'n1-highmem-16',
            'workerCount': 2,
        })

    executor.Do(self._inputs, self._outputs,
                self._serialize_custom_config_under_test())

    self.mock_runner.start_aip_training.assert_called_with(
        self._inputs,
        self._outputs,
        self._serialize_custom_config_under_test(),
        self._executor_class_path,
        {
            'project': self._project_id,
            'jobDir': self._job_dir,
            # Confirm scale tier and machine types are not overritten.
            'scaleTier': 'CUSTOM',
            'masterType': 'n1-highmem-16',
            'workerType': 'n1-highmem-16',
            # Confirm workerCount has been adjusted to num_parallel_trials.
            'workerCount': 5,
        },
        mock.ANY)


if __name__ == '__main__':
  tf.test.main()
