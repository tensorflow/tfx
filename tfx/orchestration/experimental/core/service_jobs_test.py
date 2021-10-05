# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Tests for tfx.orchestration.experimental.core.service_jobs."""

from absl.testing.absltest import mock
import tensorflow as tf
from tfx.orchestration.experimental.core import service_jobs
from tfx.orchestration.experimental.core import test_utils


class ExceptionHandlingServiceJobManagerWrapperTest(test_utils.TfxTest):

  def setUp(self):
    super().setUp()
    self._mock_service_job_manager = mock.create_autospec(
        service_jobs.ServiceJobManager, instance=True)
    self._mock_service_job_manager.ensure_node_services.return_value = (
        service_jobs.ServiceStatus.SUCCESS)
    self._mock_service_job_manager.stop_node_services.return_value = True
    self._mock_service_job_manager.is_pure_service_node.return_value = True
    self._mock_service_job_manager.is_mixed_service_node.return_value = False
    self._wrapper = service_jobs.ExceptionHandlingServiceJobManagerWrapper(
        self._mock_service_job_manager)

  def test_calls_forwarded_to_underlying_instance(self):
    self.assertEqual(service_jobs.ServiceStatus.SUCCESS,
                     self._wrapper.ensure_node_services(mock.Mock(), 'node1'))
    self.assertTrue(self._wrapper.stop_node_services(mock.Mock(), 'node2'))
    self.assertTrue(self._wrapper.is_pure_service_node(mock.Mock(), 'node3'))
    self.assertFalse(self._wrapper.is_mixed_service_node(mock.Mock(), 'node4'))
    self._mock_service_job_manager.ensure_node_services.assert_called_once_with(
        mock.ANY, 'node1')
    self._mock_service_job_manager.stop_node_services.assert_called_once_with(
        mock.ANY, 'node2')
    self._mock_service_job_manager.is_pure_service_node.assert_called_once_with(
        mock.ANY, 'node3')
    self._mock_service_job_manager.is_mixed_service_node.assert_called_once_with(
        mock.ANY, 'node4')

  def test_ensure_node_services_exception_handling(self):
    self._mock_service_job_manager.ensure_node_services.side_effect = RuntimeError(
        'test error')
    self.assertEqual(service_jobs.ServiceStatus.FAILED,
                     self._wrapper.ensure_node_services(mock.Mock(), 'node1'))
    self._mock_service_job_manager.ensure_node_services.assert_called_once_with(
        mock.ANY, 'node1')

  def test_stop_node_services_exception_handling(self):
    self._mock_service_job_manager.stop_node_services.side_effect = RuntimeError(
        'test error')
    self.assertFalse(self._wrapper.stop_node_services(mock.Mock(), 'node2'))
    self._mock_service_job_manager.stop_node_services.assert_called_once_with(
        mock.ANY, 'node2')


if __name__ == '__main__':
  tf.test.main()
