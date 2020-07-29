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

import copy
import os
from typing import Any, Dict, Text, List

import mock
import tensorflow as tf

from tfx.extensions.google_cloud_kubernetes import runner
from tfx.extensions.google_cloud_kubernetes.trainer import executor
from tfx.utils import json_utils


def mock_build_service_names(num_workers: int, unique_id: Text) -> List[Text]:
  return ['TEST-SERVICE-{}-{}'.format(unique_id, i) for i in range(num_workers)]


class RunnerTest(tf.test.TestCase):

  def setUp(self):
    super(RunnerTest, self).setUp()
    self._output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    self._mock_api_client = mock.Mock()
    self._mock_pod = mock.Mock()
    self._mock_service = mock.Mock()
    self._inputs = {}
    self._outputs = {}
    self._unique_id = "UNIQUE_ID"
    self._num_workers = 5
    self._num_gpus_per_worker = 2
    self._training_inputs = {
        'num_workers': self._num_workers,
        'num_gpus_per_worker': self._num_gpus_per_worker
    }
    # Dict format of exec_properties. custom_config needs to be serialized
    # before being passed into start_aip_training function.
    self._exec_properties = {
        'custom_config': {
            executor.TRAINING_ARGS_KEY: self._training_inputs,
        },
    }
    self._model_name = 'model_name'
    self._executor_class_path = 'my.executor.Executor'

  def _set_up_training_mocks(self):
    self._mock_create_pod = mock.Mock()
    self._mock_api_client.create_namespaced_pod = self._mock_create_pod
    self._mock_create_service = mock.Mock()
    self._mock_api_client.create_namespaced_service = self._mock_create_service
    self._mock_delete_service = mock.Mock()
    self._mock_api_client.create_delete_service = self._mock_delete_service

  def _serialize_custom_config_under_test(self) -> Dict[Text, Any]:
    """Converts self._exec_properties['custom_config'] to string."""
    result = copy.deepcopy(self._exec_properties)
    result['custom_config'] = json_utils.dumps(result['custom_config'])
    return result

  @mock.patch.object(runner, '_build_service_names', mock_build_service_names)
  @mock.patch('tfx.extensions.google_cloud_kubernetes.runner.client')
  @mock.patch('tfx.extensions.google_cloud_kubernetes.runner.kube_utils')
  def testStartKubernetesTraining(self, mock_kube_utils, mock_client):
    mock_client.V1Pod.return_value = self._mock_pod
    mock_client.V1Service.return_value = self._mock_service
    mock_kube_utils.make_core_v1_api.return_value = self._mock_api_client
    mock_kube_utils.wait_pod.return_value = mock.Mock()
    self._set_up_training_mocks()

    runner.start_gke_training(self._inputs, self._outputs,
                              self._serialize_custom_config_under_test(),
                              self._executor_class_path,
                              self._training_inputs, self._unique_id)

    self._mock_api_client.create_namespaced_service.assert_called_with(
        namespace='default',
        body=self._mock_service,)

    self._mock_api_client.create_namespaced_pod.assert_called_with(
        namespace='default',
        body=self._mock_pod,)

    expected_service_names = mock_build_service_names(self._num_workers,
                                                      self._unique_id)
    expected_calls = [mock.call(namespace='default', name=expected_service_name)
                      for expected_service_name in expected_service_names]
    self.assertEqual(expected_calls,
                     self._mock_api_client.delete_namespaced_service.mock_calls)


if __name__ == '__main__':
  tf.test.main()
