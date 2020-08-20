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
"""Docker component launcher which launches a container in docker environment ."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import re
import time
from typing import Any, Callable, Dict, List, Optional, Text, cast

from absl import logging
from kubernetes import client

from tfx import types
from tfx.components.base import executor_spec
from tfx.dsl.component.experimental import executor_specs
from tfx.orchestration.config import base_component_config
from tfx.orchestration.config import kubernetes_component_config
from tfx.orchestration.launcher import base_component_launcher
from tfx.orchestration.launcher import container_common
from tfx.utils import kube_utils


def _pod_is_not_pending(resp: client.V1Pod):
  return resp.status.phase != kube_utils.PodPhase.PENDING.value


def _pod_is_done(resp: client.V1Pod):
  return kube_utils.PodPhase(resp.status.phase).is_done


def _sanitize_pod_name(pod_name: Text) -> Text:
  pod_name = re.sub(r'[^a-z0-9-]', '-', pod_name.lower())
  pod_name = re.sub(r'^[-]+', '', pod_name)
  return re.sub(r'[-]+', '-', pod_name)


class KubernetesComponentLauncher(base_component_launcher.BaseComponentLauncher
                                 ):
  """Responsible for launching a container executor on Kubernetes."""

  # TODO(hongyes): add container spec into exec_properties for driver to check.
  @classmethod
  def can_launch(
      cls,
      component_executor_spec: executor_spec.ExecutorSpec,
      component_config: base_component_config.BaseComponentConfig = None
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
                    input_dict: Dict[Text, List[types.Artifact]],
                    output_dict: Dict[Text, List[types.Artifact]],
                    exec_properties: Dict[Text, Any]) -> None:
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

    logging.info('Looking for pod "%s:%s".', namespace, pod_name)
    resp = self._get_pod(core_api, pod_name, namespace)
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

    logging.info('Waiting for pod "%s:%s" to start.', namespace, pod_name)
    self._wait_pod(
        core_api,
        pod_name,
        namespace,
        exit_condition_lambda=_pod_is_not_pending,
        condition_description='non-pending status')

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

    resp = self._wait_pod(
        core_api,
        pod_name,
        namespace,
        exit_condition_lambda=_pod_is_done,
        condition_description='done state')

    if resp.status.phase == kube_utils.PodPhase.FAILED.value:
      raise RuntimeError('Pod "%s:%s" failed with status "%s".' %
                         (namespace, pod_name, resp.status))

    logging.info('Pod "%s:%s" is done.', namespace, pod_name)

  def _build_pod_manifest(
      self, pod_name: Text,
      container_spec: executor_spec.ExecutorContainerSpec) -> Dict[Text, Any]:
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
    metadata = pod_manifest.setdefault('metadata', {})  # type: Dict[Text, Any]
    metadata.update({'name': pod_name})
    spec = pod_manifest.setdefault('spec', {})  # type: Dict[Text, Any]
    spec.update({'restartPolicy': 'Never'})
    containers = spec.setdefault('containers',
                                 [])  # type: List[Dict[Text, Any]]
    container = None  # type: Optional[Dict[Text, Any]]
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

  def _get_pod(self, core_api: client.CoreV1Api, pod_name: Text,
               namespace: Text) -> Optional[client.V1Pod]:
    """Get a pod from Kubernetes metadata API.

    Args:
      core_api: Client of Core V1 API of Kubernetes API.
      pod_name: The name of the POD.
      namespace: The namespace of the POD.

    Returns:
      The found POD object. None if it's not found.

    Raises:
      RuntimeError: When it sees unexpected errors from Kubernetes API.
    """
    try:
      return core_api.read_namespaced_pod(name=pod_name, namespace=namespace)
    except client.rest.ApiException as e:
      if e.status != 404:
        raise RuntimeError('Unknown error! \nReason: %s\nBody: %s' %
                           (e.reason, e.body))
      return None

  def _wait_pod(self,
                core_api: client.CoreV1Api,
                pod_name: Text,
                namespace: Text,
                exit_condition_lambda: Callable[[client.V1Pod], bool],
                condition_description: Text,
                timeout_sec: int = 300) -> client.V1Pod:
    """Wait for a POD to meet an exit condition.

    Args:
      core_api: Client of Core V1 API of Kubernetes API.
      pod_name: The name of the POD.
      namespace: The namespace of the POD.
      exit_condition_lambda: A lambda which will be called intervally to wait
        for a POD to exit. The function returns True to exit.
      condition_description: The description of the exit condition which will be
        set in the error message if the wait times out.
      timeout_sec: The seconds for the function to wait. Defaults to 300s.

    Returns:
      The POD object which meets the exit condition.

    Raises:
      RuntimeError: when the function times out.
    """
    start_time = datetime.datetime.utcnow()
    while True:
      resp = self._get_pod(core_api, pod_name, namespace)
      logging.info(resp.status.phase)
      if exit_condition_lambda(resp):
        return resp
      elapse_time = datetime.datetime.utcnow() - start_time
      if elapse_time.seconds >= timeout_sec:
        raise RuntimeError(
            'Pod "%s:%s" does not reach "%s" within %s seconds.' %
            (namespace, pod_name, condition_description, timeout_sec))
      # TODO(hongyes): add exponential backoff here.
      time.sleep(1)

  def _build_pod_name(self, execution_id: int) -> Text:
    if self._pipeline_info.run_id:
      pipeline_name = (
          self._pipeline_info.pipeline_name[:50] + '-' +
          self._pipeline_info.run_id[:50])
    else:
      pipeline_name = self._pipeline_info.pipeline_name[:100]

    pod_name = '%s-%s-%s' % (
        pipeline_name, self._component_info.component_id[:50], execution_id)
    return _sanitize_pod_name(pod_name)
