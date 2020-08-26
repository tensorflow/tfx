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
"""Tests for tfx.extensions.google_cloud_kubernetes.pusher.executor."""


import copy
import json
import mock
import os
import tensorflow as tf
from typing import Any, Dict, Text

from tfx.components.pusher import executor
from tfx.extensions.google_cloud_kubernetes.pusher import executor as k8s_pusher_executor
from tfx.types import standard_artifacts
from tfx.utils import json_utils


_TEST_MODEL_NAME = 'TEST_MODEL_NAME'
_TEST_MODEL_VERSION = 1598000000
_TEST_MODEL_EXPORT_URI = 'TEST_UNDECLARED_OUTPUTS_DIR' #os.path.join('TEST_UNDECLARED_OUTPUTS_DIR',
    #_TEST_MODEL_NAME, _TEST_MODEL_VERSION)
_TEST_NUM_REPLICAS = 2


class ExecutorTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')
    self._output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    tf.io.gfile.makedirs(self._output_data_dir)
    self._model_export = standard_artifacts.Model()
    self._model_export.uri = os.path.join(self._source_data_dir,
                                          'trainer/current')
    self._model_blessing = standard_artifacts.ModelBlessing()
    self._input_dict = {
        executor.MODEL_KEY: [self._model_export],
        executor.MODEL_BLESSING_KEY: [self._model_blessing],
    }

    self._model_push = standard_artifacts.PushedModel()
    self._model_push.uri = os.path.join(self._output_data_dir, 'model_push')
    tf.io.gfile.makedirs(self._model_push.uri)
    self._output_dict = {
        executor.PUSHED_MODEL_KEY: [self._model_push],
    }
    self._serving_model_dir = os.path.join(self._output_data_dir,
                                           'serving_model_dir',
                                           _TEST_MODEL_NAME)
    tf.io.gfile.makedirs(self._serving_model_dir)
    self._executor = k8s_pusher_executor.Executor()
    self._exec_properties = self._MakeExecProperties()

  def _MakeExecProperties(self, versioning='AUTO'):
    return {
        'push_destination': json.dumps({
            'filesystem': {
                'base_directory': self._serving_model_dir,
                'versioning': versioning
            }
        }),
        'custom_config': {
            k8s_pusher_executor.TF_SERVING_ARGS_KEY: {
                'num_replicas': _TEST_NUM_REPLICAS,
            },
        },
    }

  def _SerializeCustomConfigUnderTest(self) -> Dict[Text, Any]:
    """Converts self._exec_properties['custom_config'] to string."""
    result = copy.deepcopy(self._exec_properties)
    result['custom_config'] = json_utils.dumps(result['custom_config'])
    return result

  @mock.patch.object(executor, 'time')
  def testDoBlessed(self, mock_time):
    mock_time.time = mock.Mock(return_value=_TEST_MODEL_VERSION)
    with mock.patch.object(
        k8s_pusher_executor.Executor,
        'DeployTFServingService',
        ) as mock_service, mock.patch.object(
            k8s_pusher_executor.Executor,
            'DeployTFServingDeployment',
        ) as mock_deployment:
      self._model_blessing.set_int_custom_property('blessed', 1)
      self._executor.Do(self._input_dict, self._output_dict,
                        self._SerializeCustomConfigUnderTest())
      mock_deployment.assert_called_with(
          model_name=_TEST_MODEL_NAME,
          model_uri=self._serving_model_dir,
          model_version=_TEST_MODEL_VERSION,
          num_replicas=_TEST_NUM_REPLICAS,
      )
      mock_service.assert_called()

  def testDoNotBlessed(self):
    with mock.patch.object(
        k8s_pusher_executor.Executor,
        'DeployTFServingService',
        ) as mock_service, mock.patch.object(
            k8s_pusher_executor.Executor,
            'DeployTFServingDeployment',
        ) as mock_deployment:
      self._model_blessing.set_int_custom_property('blessed', 0)
      self._executor.Do(self._input_dict, self._output_dict,
                        self._SerializeCustomConfigUnderTest())
      mock_deployment.assert_not_called()
      mock_service.assert_not_called()

  @mock.patch.object(executor, 'time')
  def testDo_NoBlessing(self, mock_time):
    mock_time.time = mock.Mock(return_value=_TEST_MODEL_VERSION)
    # Input without any blessing.
    input_dict = {executor.MODEL_KEY: [self._model_export]}
    with mock.patch.object(
        k8s_pusher_executor.Executor,
        'DeployTFServingService',
        ) as mock_service, mock.patch.object(
            k8s_pusher_executor.Executor,
            'DeployTFServingDeployment',
        ) as mock_deployment:
      # Run executor.
      self._executor.Do(input_dict,
                        self._output_dict,
                        self._SerializeCustomConfigUnderTest())

      # Check model is pushed.
      mock_deployment.assert_called_with(
          model_name=_TEST_MODEL_NAME,
          model_uri=self._serving_model_dir,
          model_version=_TEST_MODEL_VERSION,
          num_replicas=_TEST_NUM_REPLICAS,
      )
      mock_service.assert_called()


if __name__ == '__main__':
  tf.test.main()
