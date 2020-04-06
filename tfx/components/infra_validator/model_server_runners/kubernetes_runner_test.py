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

from kubernetes import client as k8s_client
from kubernetes.client import rest
import mock
import tensorflow as tf

from google.protobuf import json_format
from tfx.components.infra_validator import error_types
from tfx.components.infra_validator import serving_bins
from tfx.components.infra_validator.model_server_runners import kubernetes_runner
from tfx.proto import infra_validator_pb2
from tfx.types import standard_artifacts
from tfx.utils import kube_utils
from tfx.utils import path_utils


def _create_serving_spec(payload: Dict[Text, Any]):
  result = infra_validator_pb2.ServingSpec()
  json_format.ParseDict(payload, result)
  return result


class KubernetesRunnerTest(tf.test.TestCase):

  def setUp(self):
    super(KubernetesRunnerTest, self).setUp()
    self.addCleanup(mock.patch.stopall)

    self._base_dir = os.path.join(
        os.path.dirname(  # components/
            os.path.dirname(  # infra_validator/
                os.path.dirname(__file__))),  # model_server_runners/
        'testdata'
    )
    self._model = standard_artifacts.Model()
    self._model.uri = os.path.join(self._base_dir, 'trainer', 'current')
    self._model_name = 'chicago-taxi'

    # Prepare mocks
    self._mock_sleep = mock.patch('time.sleep').start()
    self._mock_core_v1_api = mock.patch.object(
        kube_utils, 'make_core_v1_api').start().return_value

  def _CreateKubernetesRunner(self, k8s_config_dict=None):
    self._serving_spec = infra_validator_pb2.ServingSpec()
    json_format.ParseDict({
        'tensorflow_serving': {
            'tags': ['1.15.0']},
        'kubernetes': k8s_config_dict or {},
        'model_name': self._model_name,
    }, self._serving_spec)
    serving_binary = serving_bins.parse_serving_binaries(self._serving_spec)[0]

    return kubernetes_runner.KubernetesRunner(
        model_path=path_utils.serving_model_path(self._model.uri),
        serving_binary=serving_binary,
        serving_spec=self._serving_spec)

  def _AssumeInsideKfp(
      self,
      namespace='my-namespace',
      pod_name='my-pod-name',
      pod_uid='my-pod-uid',
      pod_service_account_name='my-service-account-name',
      with_pvc=False):
    pod = k8s_client.V1Pod(
        api_version='v1',
        kind='Pod',
        metadata=k8s_client.V1ObjectMeta(
            name=pod_name,
            uid=pod_uid,
        ),
        spec=k8s_client.V1PodSpec(
            containers=[
                k8s_client.V1Container(
                    name='main',
                    volume_mounts=[]),
            ],
            volumes=[]))

    if with_pvc:
      pod.spec.volumes.append(
          k8s_client.V1Volume(
              name='my-volume',
              persistent_volume_claim=k8s_client
              .V1PersistentVolumeClaimVolumeSource(
                  claim_name='my-pvc')))
      pod.spec.containers[0].volume_mounts.append(
          k8s_client.V1VolumeMount(
              name='my-volume',
              mount_path=self._base_dir))

    mock.patch.object(kube_utils, 'is_inside_kfp', return_value=True).start()
    pod.spec.service_account_name = pod_service_account_name
    mock.patch.object(kube_utils, 'get_current_kfp_pod',
                      return_value=pod).start()
    mock.patch.object(kube_utils, 'get_kfp_namespace',
                      return_value=namespace).start()
    if with_pvc:
      (self._mock_core_v1_api.read_namespaced_persistent_volume_claim
       .return_value) = k8s_client.V1PersistentVolumeClaim(
           metadata=k8s_client.V1ObjectMeta(
               name='my-pvc'),
           spec=k8s_client.V1PersistentVolumeClaimSpec(
               access_modes=['ReadWriteMany']))

  def _AssumeOutsideKfp(self):
    mock.patch.object(kube_utils, 'is_inside_kfp', return_value=False).start()

  def testStart_InsideKfp(self):
    # Prepare mocks and variables.
    self._AssumeInsideKfp(namespace='vanilla-latte')
    runner = self._CreateKubernetesRunner()

    # Act.
    runner.Start()

    # Check states.
    self._mock_core_v1_api.create_namespaced_pod.assert_called()
    _, kwargs = self._mock_core_v1_api.create_namespaced_pod.call_args
    self.assertEqual(kwargs['namespace'], 'vanilla-latte')
    self.assertTrue(runner._pod_name)

  def testBuildPodManifest_InsideKfp(self):
    # Prepare mocks and variables.
    self._AssumeInsideKfp(
        namespace='strawberry-latte',
        pod_name='green-tea-latte',
        pod_uid='chocolate-latte',
        pod_service_account_name='vanilla-latte')
    runner = self._CreateKubernetesRunner()

    # Act.
    pod_manifest = runner._BuildPodManifest()

    # Check result.
    self.assertEqual(
        pod_manifest.metadata.generate_name, 'tfx-infraval-modelserver-')
    self.assertEqual(pod_manifest.metadata.labels, {
        'app': 'tfx-infraval-modelserver'
    })
    owner_ref = pod_manifest.metadata.owner_references[0]
    self.assertEqual(owner_ref.name, 'green-tea-latte')
    self.assertEqual(owner_ref.uid, 'chocolate-latte')
    self.assertEqual(pod_manifest.spec.service_account_name, 'vanilla-latte')
    self.assertEqual(pod_manifest.spec.restart_policy, 'Never')
    container = pod_manifest.spec.containers[0]
    self.assertEqual(container.name, 'model-server')
    self.assertEqual(container.image, 'tensorflow/serving:1.15.0')
    container_envs = {env.name for env in container.env}
    self.assertIn('MODEL_NAME', container_envs)
    self.assertIn('MODEL_BASE_PATH', container_envs)

  def testBuildPodManifest_InsideKfp_WithPvc(self):
    # Prepare mocks and variables.
    self._AssumeInsideKfp(with_pvc=True)
    runner = self._CreateKubernetesRunner()

    # Act.
    pod_manifest = runner._BuildPodManifest()

    # Check Volume.
    volume = pod_manifest.spec.volumes[0]
    self.assertEqual(volume.name, 'model-volume')
    self.assertEqual(volume.persistent_volume_claim.claim_name, 'my-pvc')

    # Check VolumeMount.
    container = pod_manifest.spec.containers[0]
    volume_mount = container.volume_mounts[0]
    self.assertEqual(volume_mount.name, 'model-volume')
    self.assertEqual(volume_mount.mount_path, self._base_dir)

  def testBuildPodManifest_InsideKfp_OverrideConfig(self):
    # Prepare mocks and variables.
    self._AssumeInsideKfp()
    runner = self._CreateKubernetesRunner(k8s_config_dict={
        'service_account_name': 'chocolate-latte',
        'active_deadline_seconds': 123,
    })

    # Act.
    pod_manifest = runner._BuildPodManifest()

    # Check result.
    self.assertEqual(pod_manifest.spec.service_account_name, 'chocolate-latte')
    self.assertEqual(pod_manifest.spec.active_deadline_seconds, 123)

  def testStart_FailsIfOutsideKfp(self):
    # Prepare mocks and variables.
    self._AssumeOutsideKfp()

    # Act.
    with self.assertRaises(NotImplementedError):
      self._CreateKubernetesRunner()

  def testStart_FailsIfStartedTwice(self):
    # Prepare mocks and variables.
    self._AssumeInsideKfp()
    runner = self._CreateKubernetesRunner()

    # Act.
    runner.Start()
    with self.assertRaises(AssertionError):
      runner.Start()

  @mock.patch('time.time')
  def testWaitUntilRunning(self, mock_time):
    # Prepare mocks and variables.
    self._AssumeInsideKfp()
    runner = self._CreateKubernetesRunner()
    mock_time.side_effect = list(range(20))
    pending_pod = mock.Mock()
    pending_pod.status.phase = 'Pending'
    running_pod = mock.Mock()
    running_pod.status.phase = 'Running'
    self._mock_core_v1_api.read_namespaced_pod.side_effect = [
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
    self.assertEqual(self._mock_core_v1_api.read_namespaced_pod.call_count, 4)

  def testWaitUntilRunning_FailsIfNotStarted(self):
    # Prepare mocks and variables.
    self._AssumeInsideKfp()
    runner = self._CreateKubernetesRunner()

    # Act.
    with self.assertRaises(AssertionError):
      runner.WaitUntilRunning(deadline=10)

  @mock.patch('time.time')
  def testWaitUntilRunning_FailsIfJobAborted(self, mock_time):
    # Prepare mocks and variables.
    self._AssumeInsideKfp()
    runner = self._CreateKubernetesRunner()
    mock_time.side_effect = list(range(20))
    terminated_pod = mock.Mock()
    terminated_pod.status.phase = 'Succeeded'
    self._mock_core_v1_api.read_namespaced_pod.return_value = terminated_pod

    # Act.
    runner.Start()
    with self.assertRaises(error_types.JobAborted):
      runner.WaitUntilRunning(deadline=10)

  @mock.patch('time.time')
  def testWaitUntilRunning_FailsIfDeadlineExceeded(self, mock_time):
    # Prepare mocks and variables.
    self._AssumeInsideKfp()
    runner = self._CreateKubernetesRunner()
    mock_time.side_effect = list(range(20))
    pending_pod = mock.Mock()
    pending_pod.status.phase = 'Pending'
    self._mock_core_v1_api.read_namespaced_pod.return_value = pending_pod

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
    self._mock_core_v1_api.delete_namespaced_pod.assert_called_once()

  def testStop_RetryIfApiException(self):
    # Prepare mocks and variables.
    self._AssumeInsideKfp()
    runner = self._CreateKubernetesRunner()
    self._mock_core_v1_api.delete_namespaced_pod.side_effect = rest.ApiException

    # Act.
    try:
      runner.Start()
      runner.Stop()
    except Exception as e:  # pylint: disable=broad-except
      self.fail(e)

    # Check calls.
    self.assertEqual(self._mock_sleep.call_count, 4)
    self.assertEqual(self._mock_core_v1_api.delete_namespaced_pod.call_count, 5)


if __name__ == '__main__':
  tf.test.main()
