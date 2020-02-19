# Lint as: python2, python3
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
"""Tests for tfx.components.infra_validator.model_server_runners.kubernetes_runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Any, Dict, Text

from kubernetes.client import rest
import mock
import tensorflow as tf

from google.protobuf import json_format
from tfx.components.infra_validator import error_types
from tfx.components.infra_validator import serving_binary_lib
from tfx.components.infra_validator.model_server_runners import kubernetes_runner
from tfx.proto import infra_validator_pb2
from tfx.types import standard_artifacts
from tfx.utils import kube_utils
from tfx.utils import time_utils


def _create_serving_spec(payload: Dict[Text, Any]):
  result = infra_validator_pb2.ServingSpec()
  json_format.ParseDict(payload, result)
  return result


class KubernetesRunnerTest(tf.test.TestCase):

  def setUp(self):
    super(KubernetesRunnerTest, self).setUp()

    base_dir = os.path.join(
        os.path.dirname(  # components/
            os.path.dirname(  # infra_validator/
                os.path.dirname(__file__))),  # model_server_runners/
        'testdata'
    )
    self._model = standard_artifacts.Model()
    self._model.uri = os.path.join(base_dir, 'trainer', 'current')
    self._model_name = 'chicago-taxi'

    patcher = mock.patch.object(kube_utils, 'make_core_v1_api')
    self._core_v1_api = patcher.start().return_value
    self.addCleanup(patcher.stop)

  def _CreateKubernetesRunner(self, k8s_config_dict=None):
    self._serving_spec = infra_validator_pb2.ServingSpec()
    json_format.ParseDict(self._serving_spec, {
        'tensorflow_serving': {
            'model_name': self._model_name,
            'tags': ['1.15.0']},
        'kubernetes': k8s_config_dict or {}
    })
    self._serving_binary = serving_binary_lib.parse_serving_binaries(
        self._serving_spec)[0]
    patcher = mock.patch.object(self._serving_binary, 'MakeClient')
    self._model_server_client = patcher.start().return_value
    self.addCleanup(patcher.stop)

    return kubernetes_runner.KubernetesRunner(
        model=self._model,
        serving_binary=self._serving_binary,
        serving_spec=self._serving_spec)

  def _AssumeInsideKfp(
      self,
      namespace='my-namespace',
      pod_name='my-pod-name',
      pod_uid='my-pod-uid',
      pod_service_account_name='my-service-account-name'):
    patcher = mock.patch.object(kube_utils, 'is_inside_kfp')
    patcher.start().return_value = True
    self.addCleanup(patcher.stop)

    patcher = mock.patch.object(kube_utils, 'get_current_kfp_pod')
    pod = patcher.start().return_value
    pod.api_version = 'v1'
    pod.kind = 'Pod'
    pod.metadata.name = pod_name
    pod.metadata.uid = pod_uid
    pod.spec.service_account_name = pod_service_account_name
    self.addCleanup(patcher.stop)

    patcher = mock.patch.object(kube_utils, 'get_kfp_namespace')
    patcher.start().return_value = namespace
    self.addCleanup(patcher.stop)

  def _AssumeOutsideKfp(self):
    patcher = mock.patch.object(kube_utils, 'is_inside_kfp')
    patcher.start().return_value = False
    self.addCleanup(patcher.stop)

  def testStart_InsideKfp(self):
    # Prepare mocks and variables.
    self._AssumeInsideKfp(
        namespace='strawberry-latte',
        pod_name='green-tea-latte',
        pod_uid='chocolate-latte',
        pod_service_account_name='vanilla-latte')
    runner = self._CreateKubernetesRunner()

    # Act.
    runner.Start()

    # Check calls.
    self._core_v1_api.create_namespaced_pod.assert_called()
    _, kwargs = self._core_v1_api.create_namespaced_pod.call_args
    self.assertEqual(kwargs['namespace'], 'strawberry-latte')
    pod_manifest = kwargs['body']
    self.assertTrue(
        pod_manifest.metadata.name.startswith('infra-validator-model-server-'))
    self.assertEqual(pod_manifest.metadata.labels, {
        'app': 'infra-validator-model-server'
    })
    owner_ref = pod_manifest.ownerReferences[0]
    self.assertEqual(owner_ref.name, 'green-tea-latte')
    self.assertEqual(owner_ref.uid, 'chocolate-latte')
    self.assertEqual(pod_manifest.spec.service_account_name, 'vanilla-latte')
    self.assertEqual(pod_manifest.spec.restart_policy, 'Never')
    container = pod_manifest.spec.containers[0]
    self.assertEqual(container.name, 'model-server')
    self.assertEqual(container.image, 'tensorflow/serving:1.15.0')
    self.assertEqual(container.env, {
        'MODEL_NAME': self._model_name,
        'MODEL_BASE_PATH': '/model'
    })

  def testStart_InsideKfp_OverrideConfig(self):
    # Prepare mocks and variables.
    self._AssumeInsideKfp()
    runner = self._CreateKubernetesRunner(k8s_config_dict={
        'service_account_name': 'chocolate-latte',
        'image_pull_secrets': ['vanilla', 'latte'],
        'resources': {
            'limits': {
                'cpu': '1',
                'memory': '500Mi',
            },
            'requests': {
                'cpu': '0.5',
                'memory': '250Mi',
            },
        }
    })

    # Act.
    runner.Start()

    # Check calls.
    self._core_v1_api.create_namespaced_pod.assert_called()
    _, kwargs = self._core_v1_api.create_namespaced_pod.call_args
    pod_manifest = kwargs['body']
    self.assertEqual(pod_manifest.spec.service_account_name, 'chocolate-latte')
    self.assertEqual(pod_manifest.spec.image_pull_requests[0].name, 'vanilla')
    self.assertEqual(pod_manifest.spec.image_pull_requests[1].name, 'latte')
    container = pod_manifest.spec.containers[0]
    self.assertEqual(container.resources.limits['cpu'], '1')
    self.assertEqual(container.resources.limits['memory'], '500Mi')
    self.assertEqual(container.resources.requests['cpu'], '0.5')
    self.assertEqual(container.resources.requests['memory'], '250Mi')

  def testStart_FailsIfOutsideKfp(self):
    # Prepare mocks and variables.
    self._AssumeOutsideKfp()
    runner = self._CreateKubernetesRunner()

    # Act.
    with self.assertRaises(error_types.IllegalState):
      runner.Start()

  def testStart_FailsIfStartedTwice(self):
    # Prepare mocks and variables.
    self._AssumeInsideKfp()
    runner = self._CreateKubernetesRunner()

    # Act.
    runner.Start()
    with self.assertRaises(error_types.IllegalState):
      runner.Start()

  @mock.patch.object(time_utils, 'utc_timestamp')
  def testWaitUntilRunning(self, timestamp_mock):
    # Prepare mocks and variables.
    self._AssumeInsideKfp()
    runner = self._CreateKubernetesRunner()
    timestamp_mock.side_effect = list(range(20))
    pending_pod = mock.Mock()
    pending_pod.status.phase = 'Pending'
    running_pod = mock.Mock()
    running_pod.status.phase = 'Running'
    self._core_v1_api.read_namespaced_pod.side_effect = [
        rest.ApiException('meh'),  # Error is tolerable.
        pending_pod,
        pending_pod,
        running_pod
    ]

    # Act.
    runner.Start()
    try:
      runner.WaitUntilRunning(deadline=10)
    except Exception as e:  # pylint: disable=broad-except
      self.fail(e)

    # Check calls.
    self.assertEqual(self._core_v1_api.read_namespaced_pod.call_count, 4)

  def testWaitUntilRunning_FailsIfNotStarted(self):
    # Prepare mocks and variables.
    self._AssumeInsideKfp()
    runner = self._CreateKubernetesRunner()

    # Act.
    with self.assertRaises(error_types.IllegalState):
      runner.WaitUntilRunning(deadline=10)

  @mock.patch.object(time_utils, 'utc_timestamp')
  def testWaitUntilRunning_FailsIfJobAborted(self, timestamp_mock):
    # Prepare mocks and variables.
    self._AssumeInsideKfp()
    runner = self._CreateKubernetesRunner()
    timestamp_mock.side_effect = list(range(20))
    terminated_pod = mock.Mock()
    terminated_pod.status.phase = 'Succeeded'
    self._core_v1_api.read_namespaced_pod.return_value = terminated_pod

    # Act.
    runner.Start()
    with self.assertRaises(error_types.JobAborted):
      runner.WaitUntilRunning(deadline=10)

  @mock.patch.object(time_utils, 'utc_timestamp')
  def testWaitUntilRunning_FailsIfDeadlineExceeded(self, timestamp_mock):
    # Prepare mocks and variables.
    self._AssumeInsideKfp()
    runner = self._CreateKubernetesRunner()
    timestamp_mock.side_effect = list(range(20))
    pending_pod = mock.Mock()
    pending_pod.status.phase = 'Pending'
    self._core_v1_api.read_namespaced_pod.return_value = pending_pod

    # Act.
    runner.Start()
    with self.assertRaises(error_types.DeadlineExceeded):
      runner.WaitUntilRunning(deadline=10)

  def testStop(self):
    # Prepare mocks and variables.
    self._AssumeInsideKfp()
    runner = self._CreateKubernetesRunner()

    # Act.
    try:
      runner.Start()
      runner.Stop()
    except Exception as e:  # pylint: disable=broad-except
      self.fail(e)

    # Check calls.
    self._core_v1_api.delete_namespaced_pod.assert_called_once()

  def testStop_OkEvenIfApiException(self):
    # Prepare mocks and variables.
    self._AssumeInsideKfp()
    runner = self._CreateKubernetesRunner()
    self._core_v1_api.delete_namespaced_pod.side_effect = rest.ApiException

    # Act.
    try:
      runner.Start()
      runner.Stop()
    except Exception as e:  # pylint: disable=broad-except
      self.fail(e)

    # Check calls.
    self._core_v1_api.delete_namespaced_pod.assert_called_once()
