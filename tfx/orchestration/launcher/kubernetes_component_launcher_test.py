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
"""Tests for tfx.orchestration.launcher.kubernetes_component_launcher."""

import os
from unittest import mock

from kubernetes import client
from kubernetes import config
import tensorflow as tf
from tfx.dsl.components.base import base_executor
from tfx.dsl.components.base import executor_spec
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.orchestration import publisher
from tfx.orchestration.config import kubernetes_component_config
from tfx.orchestration.launcher import kubernetes_component_launcher
from tfx.orchestration.launcher import test_utils
from tfx.types import channel_utils
from tfx.utils import kube_utils

from ml_metadata.proto import metadata_store_pb2

_KFP_NAMESPACE = 'ns-1'
_KFP_PODNAME = 'pod-1'


def _mock_current_kfp_pod(cli: client.CoreV1Api) -> client.V1Pod:
  """Mock method to get KFP pod manifest."""
  return cli.read_namespaced_pod(name=_KFP_PODNAME, namespace=_KFP_NAMESPACE)


class KubernetesComponentLauncherTest(tf.test.TestCase):

  def testCanLaunch(self):
    self.assertTrue(
        kubernetes_component_launcher.KubernetesComponentLauncher.can_launch(
            executor_spec.ExecutorContainerSpec(image='test')))
    self.assertFalse(
        kubernetes_component_launcher.KubernetesComponentLauncher.can_launch(
            executor_spec.ExecutorClassSpec(base_executor.BaseExecutor)))

  @mock.patch.object(kube_utils, 'get_current_kfp_pod', _mock_current_kfp_pod)
  @mock.patch.object(
      kube_utils,
      'get_kfp_namespace',
      autospec=True,
      return_value=_KFP_NAMESPACE)
  @mock.patch.object(
      kube_utils, 'is_inside_kfp', autospec=True, return_value=True)
  @mock.patch.object(publisher, 'Publisher', autospec=True)
  @mock.patch.object(config, 'load_incluster_config', autospec=True)
  @mock.patch.object(client, 'CoreV1Api', autospec=True)
  def testLaunch_loadInClusterSucceed(self, mock_core_api_cls,
                                      mock_incluster_config, mock_publisher,
                                      mock_is_inside_kfp, mock_kfp_namespace):
    mock_publisher.return_value.publish_execution.return_value = {}
    core_api = mock_core_api_cls.return_value
    core_api.read_namespaced_pod.side_effect = [
        self._mock_launcher_pod(),
        client.rest.ApiException(status=404),  # Mock no existing pod state.
        self._mock_executor_pod(
            'Pending'),  # Mock pending state after creation.
        self._mock_executor_pod('Running'),  # Mock running state after pending.
        self._mock_executor_pod('Succeeded'),  # Mock Succeeded state.
    ]
    # Mock successful pod creation.
    core_api.create_namespaced_pod.return_value = client.V1Pod()
    core_api.read_namespaced_pod_log.return_value.stream.return_value = [
        b'log-1'
    ]
    context = self._create_launcher_context()

    context['launcher'].launch()

    self.assertEqual(5, core_api.read_namespaced_pod.call_count)
    core_api.create_namespaced_pod.assert_called_once()
    core_api.read_namespaced_pod_log.assert_called_once()
    _, mock_kwargs = core_api.create_namespaced_pod.call_args
    self.assertEqual(_KFP_NAMESPACE, mock_kwargs['namespace'])
    pod_manifest = mock_kwargs['body']
    self.assertDictEqual(
        {
            'apiVersion': 'v1',
            'kind': 'Pod',
            'metadata': {
                'name':
                    'test-123-fakecomponent-fakecomponent-123',
                'ownerReferences': [{
                    'apiVersion': 'argoproj.io/v1alpha1',
                    'kind': 'Workflow',
                    'name': 'wf-1',
                    'uid': 'wf-uid-1'
                }]
            },
            'spec': {
                'restartPolicy': 'Never',
                'containers': [{
                    'name': 'main',
                    'image': 'gcr://test',
                    'command': None,
                    'args': [context['input_artifact'].uri],
                }],
                'serviceAccount': 'sa-1',
                'serviceAccountName': None
            }
        }, pod_manifest)

  @mock.patch.object(publisher, 'Publisher', autospec=True)
  @mock.patch.object(config, 'load_incluster_config', autospec=True)
  @mock.patch.object(config, 'load_kube_config', autospec=True)
  @mock.patch.object(client, 'CoreV1Api', autospec=True)
  def testLaunch_loadKubeConfigSucceed(self, mock_core_api_cls,
                                       mock_kube_config, mock_incluster_config,
                                       mock_publisher):
    mock_publisher.return_value.publish_execution.return_value = {}
    mock_incluster_config.side_effect = config.config_exception.ConfigException(
    )
    core_api = mock_core_api_cls.return_value
    core_api.read_namespaced_pod.side_effect = [
        client.rest.ApiException(status=404),  # Mock no existing pod state.
        self._mock_executor_pod(
            'Pending'),  # Mock pending state after creation.
        self._mock_executor_pod('Running'),  # Mock running state after pending.
        self._mock_executor_pod('Succeeded'),  # Mock Succeeded state.
    ]
    # Mock successful pod creation.
    core_api.create_namespaced_pod.return_value = client.V1Pod()
    core_api.read_namespaced_pod_log.return_value.stream.return_value = [
        b'log-1'
    ]
    context = self._create_launcher_context()

    context['launcher'].launch()

    self.assertEqual(4, core_api.read_namespaced_pod.call_count)
    core_api.create_namespaced_pod.assert_called_once()
    core_api.read_namespaced_pod_log.assert_called_once()
    _, mock_kwargs = core_api.create_namespaced_pod.call_args
    self.assertEqual('kubeflow', mock_kwargs['namespace'])
    pod_manifest = mock_kwargs['body']
    self.assertDictEqual(
        {
            'apiVersion': 'v1',
            'kind': 'Pod',
            'metadata': {
                'name': 'test-123-fakecomponent-fakecomponent-123',
            },
            'spec': {
                'restartPolicy':
                    'Never',
                'serviceAccount':
                    'tfx-service-account',
                'serviceAccountName':
                    'tfx-service-account',
                'containers': [{
                    'name': 'main',
                    'image': 'gcr://test',
                    'command': None,
                    'args': [context['input_artifact'].uri],
                }],
            }
        }, pod_manifest)

  @mock.patch.object(kube_utils, 'get_current_kfp_pod', _mock_current_kfp_pod)
  @mock.patch.object(
      kube_utils,
      'get_kfp_namespace',
      autospec=True,
      return_value=_KFP_NAMESPACE)
  @mock.patch.object(
      kube_utils, 'is_inside_kfp', autospec=True, return_value=True)
  @mock.patch.object(publisher, 'Publisher', autospec=True)
  @mock.patch.object(config, 'load_incluster_config', autospec=True)
  @mock.patch.object(client, 'CoreV1Api', autospec=True)
  def testLaunch_withComponentConfig(self, mock_core_api_cls,
                                     mock_incluster_config, mock_publisher,
                                     mock_is_inside_kfp, mock_kfp_namespace):
    mock_publisher.return_value.publish_execution.return_value = {}
    core_api = mock_core_api_cls.return_value
    core_api.read_namespaced_pod.side_effect = [
        self._mock_launcher_pod(),
        client.rest.ApiException(status=404),  # Mock no existing pod state.
        self._mock_executor_pod(
            'Pending'),  # Mock pending state after creation.
        self._mock_executor_pod('Running'),  # Mock running state after pending.
        self._mock_executor_pod('Succeeded'),  # Mock Succeeded state.
    ]
    # Mock successful pod creation.
    core_api.create_namespaced_pod.return_value = client.V1Pod()
    core_api.read_namespaced_pod_log.return_value.stream.return_value = [
        b'log-1'
    ]
    component_config = kubernetes_component_config.KubernetesComponentConfig(
        client.V1Pod(
            spec=client.V1PodSpec(containers=[
                client.V1Container(
                    name='main', resources={'limits': {
                        'memory': '200mi'
                    }})
            ])))
    context = self._create_launcher_context(component_config)

    context['launcher'].launch()

    self.assertEqual(5, core_api.read_namespaced_pod.call_count)
    core_api.create_namespaced_pod.assert_called_once()
    core_api.read_namespaced_pod_log.assert_called_once()
    _, mock_kwargs = core_api.create_namespaced_pod.call_args
    self.assertEqual(_KFP_NAMESPACE, mock_kwargs['namespace'])
    pod_manifest = mock_kwargs['body']
    print(pod_manifest)
    self.assertDictEqual(
        {
            'apiVersion': 'v1',
            'kind': 'Pod',
            'metadata': {
                'name':
                    'test-123-fakecomponent-fakecomponent-123',
                'ownerReferences': [{
                    'apiVersion': 'argoproj.io/v1alpha1',
                    'kind': 'Workflow',
                    'name': 'wf-1',
                    'uid': 'wf-uid-1'
                }]
            },
            'spec': {
                'restartPolicy': 'Never',
                'containers': [{
                    'name': 'main',
                    'image': 'gcr://test',
                    'command': None,
                    'args': [context['input_artifact'].uri],
                    'resources': {
                        'limits': {
                            'memory': '200mi'
                        }
                    }
                }],
                'serviceAccount': 'sa-1',
                'serviceAccountName': None
            }
        }, pod_manifest)

  def _create_launcher_context(self, component_config=None):
    test_dir = self.get_temp_dir()

    connection_config = metadata_store_pb2.ConnectionConfig()
    connection_config.sqlite.SetInParent()
    metadata_connection = metadata.Metadata(connection_config)

    pipeline_root = os.path.join(test_dir, 'Test')

    input_artifact = test_utils._InputArtifact()
    input_artifact.uri = os.path.join(test_dir, 'input')

    component = test_utils._FakeComponent(
        name='FakeComponent',
        input_channel=channel_utils.as_channel([input_artifact]),
        custom_executor_spec=executor_spec.ExecutorContainerSpec(
            image='gcr://test', args=['{{input_dict["input"][0].uri}}']))

    pipeline_info = data_types.PipelineInfo(
        pipeline_name='Test', pipeline_root=pipeline_root, run_id='123')

    driver_args = data_types.DriverArgs(enable_cache=True)

    launcher = kubernetes_component_launcher.KubernetesComponentLauncher.create(
        component=component,
        pipeline_info=pipeline_info,
        driver_args=driver_args,
        metadata_connection=metadata_connection,
        beam_pipeline_args=[],
        additional_pipeline_args={},
        component_config=component_config)

    return {'launcher': launcher, 'input_artifact': input_artifact}

  def _mock_launcher_pod(self):
    return client.V1Pod(
        metadata=client.V1ObjectMeta(owner_references=[
            client.V1OwnerReference(
                api_version='argoproj.io/v1alpha1',
                kind='Workflow',
                name='wf-1',
                uid='wf-uid-1')
        ]),
        spec=client.V1PodSpec(containers=[], service_account='sa-1'))

  def _mock_executor_pod(self, phase):
    return client.V1Pod(status=client.V1PodStatus(phase=phase))


if __name__ == '__main__':
  tf.test.main()
