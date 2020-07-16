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
"""Tests for tfx.extensions.google_cloud_ai_platform.pusher.executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
from typing import Any, Dict, Text
# Standard Imports
import mock
import tensorflow as tf

from tfx.components.pusher import executor as tfx_pusher_executor
from tfx.extensions.google_cloud_kubernetes.pusher import executor
from tfx.types import standard_artifacts
from tfx.utils import json_utils


_TEST_MODEL_NAME = 'TEST_MODEL_NAME'
_TEST_NUM_REPLICAS = 2
_TEST_MODEL_EXPORT_URI = 'TEST_UNDECLARED_OUTPUTS_DIR'


class ExecutorTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._model_export = standard_artifacts.Model()
    self._model_export.uri = _TEST_MODEL_EXPORT_URI
    self._model_blessing = standard_artifacts.ModelBlessing()
    self._input_dict = {
        tfx_pusher_executor.MODEL_KEY: [self._model_export],
        tfx_pusher_executor.MODEL_BLESSING_KEY: [self._model_blessing],
    }
    self._model_push = standard_artifacts.PushedModel()
    self._output_dict = {
        tfx_pusher_executor.PUSHED_MODEL_KEY: [self._model_push],
    }
    self._executor = executor.Executor()
    self._exec_properties = self._MakeExecProperties()

  def _MakeExecProperties(self):
    return {
        'custom_config': {
            executor.TF_SERVING_ARGS_KEY: {
                'model_name': _TEST_MODEL_NAME,
                'num_replicas': _TEST_NUM_REPLICAS,
            },
        },
    }

  def _SerializeCustomConfigUnderTest(self) -> Dict[Text, Any]:
    """Converts self._exec_properties['custom_config'] to string."""
    result = copy.deepcopy(self._exec_properties)
    result['custom_config'] = json_utils.dumps(result['custom_config'])
    return result

  def assertPushed(self):
    self.assertEqual(1, self._model_push.get_int_custom_property('pushed'))

  def assertNotPushed(self):
    self.assertEqual(0, self._model_push.get_int_custom_property('pushed'))

  def testDoBlessed(self):
    with mock.patch.object(
        executor.Executor,
        'DeployTFServingService',
        ) as mock_service, mock.patch.object(
            executor.Executor,
            'DeployTFServingDeployment',
        ) as mock_deployment:
      self._model_blessing.set_int_custom_property('blessed', 1)
      self._executor.Do(self._input_dict, self._output_dict,
                        self._SerializeCustomConfigUnderTest())
      self.assertPushed()
      mock_deployment.assert_called_with(
          model_name=_TEST_MODEL_NAME,
          model_uri=os.path.join(_TEST_MODEL_EXPORT_URI, 'serving_model_dir'),
          num_replicas=_TEST_NUM_REPLICAS,
      )
      mock_service.assert_called()

  def testDoNotBlessed(self):
    with mock.patch.object(
        executor.Executor,
        'DeployTFServingService',
        ) as mock_service, mock.patch.object(
            executor.Executor,
            'DeployTFServingDeployment',
        ) as mock_deployment:
      self._model_blessing.set_int_custom_property('blessed', 0)
      self._executor.Do(self._input_dict, self._output_dict,
                        self._SerializeCustomConfigUnderTest())
      self.assertNotPushed()
      mock_deployment.assert_not_called()
      mock_service.assert_not_called()

  def testDo_NoBlessing(self):
    # Input without any blessing.
    input_dict = {tfx_pusher_executor.MODEL_KEY: [self._model_export]}
    with mock.patch.object(
        executor.Executor,
        'DeployTFServingService',
        ) as mock_service, mock.patch.object(
            executor.Executor,
            'DeployTFServingDeployment',
        ) as mock_deployment:
      # Run executor
      self._executor.Do(input_dict,
                        self._output_dict,
                        self._SerializeCustomConfigUnderTest())

      # Check model is pushed.
      self.assertPushed()
      mock_deployment.assert_called_with(
          model_name=_TEST_MODEL_NAME,
          model_uri=os.path.join(_TEST_MODEL_EXPORT_URI, 'serving_model_dir'),
          num_replicas=_TEST_NUM_REPLICAS,
      )
      mock_service.assert_called()


if __name__ == '__main__':
  tf.test.main()
