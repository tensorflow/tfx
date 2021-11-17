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
"""Docker component launcher which launches a container in docker environment ."""

from typing import Any, Dict, List, Optional, cast

from absl import logging
from kubernetes import client
from tfx import types
from tfx.dsl.component.experimental import executor_specs
from tfx.dsl.components.base import executor_spec
from tfx.orchestration.config import base_component_config
from tfx.orchestration.config import kubernetes_component_config
from tfx.orchestration.launcher import base_component_launcher
from tfx.orchestration.launcher import container_common
from tfx.utils import kube_utils


class KubernetesComponentLauncher(base_component_launcher.BaseComponentLauncher
                                 ):
  """Responsible for launching a container executor on Kubernetes."""

  # TODO(hongyes): add container spec into exec_properties for driver to check.
  @classmethod
  def can_launch(
      cls,
      component_executor_spec: executor_spec.ExecutorSpec,
      component_config: Optional[
          base_component_config.BaseComponentConfig] = None
  ) -> bool:
    """Checks if the launcher can launch the executor spec."""
    if component_config and not isinstance(
        component_config,
        kubernetes_component_config.KubernetesComponentConfig):
      return False

    return isinstance(component_executor_spec,
                      (executor_spec.ExecutorContainerSpec,
                       executor_specs.TemplatedExecutorContainerSpec))

  def _run_executor(self, execution_id: int,
                    input_dict: Dict[str, List[types.Artifact]],
                    output_dict: Dict[str, List[types.Artifact]],
                    exec_properties: Dict[str, Any]) -> None:
    """Execute underlying component implementation.

    Runs executor container in a Kubernetes Pod and wait until it goes into
    `Succeeded` or `Failed` state.

    Args:
      execution_id: The ID of the execution.
      input_dict: Input dict from input key to a list of Artifacts. These are
        often outputs of another component in the pipeline and passed to the
        component by the orchestration system.
      output_dict: Output dict from output key to a list of Artifacts. These are
        often consumed by a dependent component.
      exec_properties: A dict of execution properties. These are inputs to
        pipeline with primitive types (int, string, float) and fully
        materialized when a pipeline is constructed. No dependency to other
        component or later injection from orchestration systems is necessary or
        possible on these values.

    Raises:
      RuntimeError: when the pod is in `Failed` state or unexpected failure from
      Kubernetes API.

    """

    container_spec = cast(executor_spec.ExecutorContainerSpec,
                          self._component_executor_spec)

    # Replace container spec with jinja2 template.
    container_spec = container_common.resolve_container_template(
        container_spec, input_dict, output_dict, exec_properties)
    pod_name = self._build_pod_name(execution_id)
    # TODO(hongyes): replace the default value from component config.
    try:
      namespace = kube_utils.get_kfp_namespace()
    except RuntimeError:
      namespace = 'kubeflow'

    pod_manifest = self._build_pod_manifest(pod_name, container_spec)
    core_api = kube_utils.make_core_v1_api()

    if kube_utils.is_inside_kfp():
      launcher_pod = kube_utils.get_current_kfp_pod(core_api)
      pod_manifest['spec']['serviceAccount'] = launcher_pod.spec.service_account
      pod_manifest['spec'][
          'serviceAccountName'] = launcher_pod.spec.service_account_name
      pod_manifest['metadata'][
          'ownerReferences'] = container_common.to_swagger_dict(
              launcher_pod.metadata.owner_references)
    else:
      pod_manifest['spec']['serviceAccount'] = kube_utils.TFX_SERVICE_ACCOUNT
      pod_manifest['spec'][
          'serviceAccountName'] = kube_utils.TFX_SERVICE_ACCOUNT

    logging.info('Looking for pod "%s:%s".', namespace, pod_name)
    resp = kube_utils.get_pod(core_api, pod_name, namespace)
    if not resp:
      logging.info('Pod "%s:%s" does not exist. Creating it...',
                   namespace, pod_name)
      logging.info('Pod manifest: %s', pod_manifest)
      try:
        resp = core_api.create_namespaced_pod(
            namespace=namespace, body=pod_manifest)
      except client.rest.ApiException as e:
        raise RuntimeError(
            'Failed to created container executor pod!\nReason: %s\nBody: %s' %
            (e.reason, e.body))

    # Wait up to 300 seconds for the pod to move from pending to another status.
    logging.info('Waiting for pod "%s:%s" to start.', namespace, pod_name)
    kube_utils.wait_pod(
        core_api,
        pod_name,
        namespace,
        exit_condition_lambda=kube_utils.pod_is_not_pending,
        condition_description='non-pending status',
        timeout_sec=300)

    logging.info('Start log streaming for pod "%s:%s".', namespace, pod_name)
    try:
      logs = core_api.read_namespaced_pod_log(
          name=pod_name,
          namespace=namespace,
          container=kube_utils.ARGO_MAIN_CONTAINER_NAME,
          follow=True,
          _preload_content=False).stream()
    except client.rest.ApiException as e:
      raise RuntimeError(
          'Failed to stream the logs from the pod!\nReason: %s\nBody: %s' %
          (e.reason, e.body))

    for log in logs:
      logging.info(log.decode().rstrip('\n'))

    # Wait indefinitely for the pod to complete.
    resp = kube_utils.wait_pod(
        core_api,
        pod_name,
        namespace,
        exit_condition_lambda=kube_utils.pod_is_done,
        condition_description='done state')

    if resp.status.phase == kube_utils.PodPhase.FAILED.value:
      raise RuntimeError('Pod "%s:%s" failed with status "%s".' %
                         (namespace, pod_name, resp.status))

    logging.info('Pod "%s:%s" is done.', namespace, pod_name)

  def _build_pod_manifest(
      self, pod_name: str,
      container_spec: executor_spec.ExecutorContainerSpec) -> Dict[str, Any]:
    """Build a pod spec.

    The function builds a pod spec by patching executor container spec into
    the pod spec from component config.

    Args:
      pod_name: The name of the pod.
      container_spec: The resolved executor container spec.

    Returns:
      The pod manifest in dictionary format.
    """
    if self._component_config:
      kubernetes_config = cast(
          kubernetes_component_config.KubernetesComponentConfig,
          self._component_config)
      pod_manifest = container_common.to_swagger_dict(kubernetes_config.pod)
    else:
      pod_manifest = {}

    pod_manifest.update({
        'apiVersion': 'v1',
        'kind': 'Pod',
    })
    # TODO(hongyes): figure out a better way to figure out type hints for nested
    # dict.
    metadata = pod_manifest.setdefault('metadata', {})  # type: Dict[str, Any]  # pytype: disable=annotation-type-mismatch
    metadata.update({'name': pod_name})
    spec = pod_manifest.setdefault('spec', {})  # type: Dict[str, Any]  # pytype: disable=annotation-type-mismatch
    spec.update({'restartPolicy': 'Never'})
    containers = spec.setdefault('containers', [])  # type: List[Dict[str, Any]]
    container = None  # type: Optional[Dict[str, Any]]
    for c in containers:
      if c['name'] == kube_utils.ARGO_MAIN_CONTAINER_NAME:
        container = c
        break
    if not container:
      container = {'name': kube_utils.ARGO_MAIN_CONTAINER_NAME}
      containers.append(container)
    container.update({
        'image': container_spec.image,
        'command': container_spec.command,
        'args': container_spec.args,
    })
    return pod_manifest

  def _build_pod_name(self, execution_id: int) -> str:
    if self._pipeline_info.run_id:
      pipeline_name = (
          self._pipeline_info.pipeline_name[:50] + '-' +
          self._pipeline_info.run_id[:50])
    else:
      pipeline_name = self._pipeline_info.pipeline_name[:100]

    pod_name = '%s-%s-%s' % (
        pipeline_name, self._component_info.component_id[:50], execution_id)
    return kube_utils.sanitize_pod_name(pod_name)
