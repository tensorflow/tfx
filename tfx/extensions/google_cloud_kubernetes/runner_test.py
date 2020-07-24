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
"""Tests for tfx.extensions.google_cloud_kubernetes.runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
import sys
from typing import Any, Dict, Text

import mock
import tensorflow as tf

from tfx import version
from tfx.extensions.google_cloud_ai_platform import runner
from tfx.extensions.google_cloud_ai_platform.trainer import executor
from tfx.utils import json_utils
from tfx.utils import telemetry_utils


class RunnerTest(tf.test.TestCase):

  def setUp(self):
    super(RunnerTest, self).setUp()
    self._output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    self._mock_k8s_client = mock.Mock()
    self._inputs = {}
    self._outputs = {}
    self._training_inputs = {
        '': self._project_id,
    }
    self._job_id = 'my_jobid'
    # Dict format of exec_properties. custom_config needs to be serialized
    # before being passed into start_aip_training function.
    self._exec_properties = {
        'custom_config': {
            executor.TRAINING_ARGS_KEY: self._training_inputs,
        },
    }
    self._model_name = 'model_name'
    self._ai_platform_serving_args = {
        'model_name': self._model_name,
        'project_id': self._project_id,
    }
    self._executor_class_path = 'my.executor.Executor'

  def _setUpTrainingMocks(self):
    # self._mock_create = mock.Mock()
    # self._mock_api_client.projects().jobs().create = self._mock_create
    # self._mock_get = mock.Mock()
    # self._mock_api_client.projects().jobs().get.return_value = self._mock_get
    # self._mock_get.execute.return_value = {
    #     'state': 'SUCCEEDED',
    # }

  def _serialize_custom_config_under_test(self) -> Dict[Text, Any]:
    """Converts self._exec_properties['custom_config'] to string."""
    result = copy.deepcopy(self._exec_properties)
    result['custom_config'] = json_utils.dumps(result['custom_config'])
    return result