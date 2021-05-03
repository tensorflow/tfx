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
"""Model server runner for kubernetes runtime."""

import datetime
import os
import time
from typing import Optional

from absl import logging
from apache_beam.utils import retry
from kubernetes import client as k8s_client
from kubernetes.client import rest

from tfx.components.infra_validator import error_types
from tfx.components.infra_validator import serving_bins
from tfx.components.infra_validator.model_server_runners import base_runner
from tfx.proto import infra_validator_pb2
from tfx.utils import kube_utils

_DEFAULT_POLLING_INTERVAL_SEC = 5
_DEFAULT_ACTIVE_DEADLINE_SEC = int(datetime.timedelta(hours=24).total_seconds())

_NUM_RETRIES = 4  # Total 5 attempts
# Total delay is smaller than 2 + 4 + 8 + 16 = 30 seconds which is the default
# kubernetes graceful shutdown period.
_INITIAL_BACKOFF_DELAY_SEC = 2.0

# Alias enums.
_PodPhase = kube_utils.PodPhase
_RestartPolicy = kube_utils.RestartPolicy
_AccessMode = kube_utils.PersistentVolumeAccessMode

# Kubernetes resource metadata values
_APP_KEY = 'app'
_MODEL_SERVER_POD_NAME_PREFIX = 'tfx-infraval-modelserver-'
_MODEL_SERVER_APP_LABEL = 'tfx-infraval-modelserver'
_MODEL_SERVER_CONTAINER_NAME = 'model-server'
_MODEL_SERVER_MODEL_VOLUME_NAME = 'model-volume'


# TODO(b/149534564): Use pathlib.
def _is_subdirectory(maybe_parent: str, maybe_child: str) -> bool:
  paren = os.path.realpath(maybe_parent).split(os.path.sep)
  child = os.path.realpath(maybe_child).split(os.path.sep)
  return len(paren) <= len(child) and all(a == b for a, b in zip(paren, child))


def _get_container_or_error(
    pod: k8s_client.V1Pod, container_name: str) -> k8s_client.V1Container:
  for container in pod.spec.containers:
    if container.name == container_name:
      return container
  raise ValueError(
      'Unable to find {} container from the pod (found {}).'.format(
          container_name, [c.name for c in pod.spec.containers]))


def _api_exception_retry_filter(exception: Exception):
  return isinstance(exception, rest.ApiException)


def _convert_to_kube_env(
    env: infra_validator_pb2.EnvVar) -> k8s_client.V1EnvVar:
  """Convert infra_validator_pb2.EnvVar to kubernetes.V1EnvVar."""
  if not env.name:
    raise ValueError('EnvVar.name must be specified.')
  if env.HasField('value_from'):
    if env.value_from.HasField('secret_key_ref'):
      value_source = k8s_client.V1EnvVarSource(
          secret_key_ref=k8s_client.V1SecretKeySelector(
              name=env.value_from.secret_key_ref.name,
              key=env.value_from.secret_key_ref.key))
      return k8s_client.V1EnvVar(name=env.name, value_from=value_source)
    else:
      raise ValueError(f'Bad EnvVar: {env}')
  else:
    # Note that env.value can be empty.
    return k8s_client.V1EnvVar(name=env.name, value=env.value)


class KubernetesRunner(base_runner.BaseModelServerRunner):
  """A model server runner that launches model server in kubernetes cluster."""

  def __init__(
      self,
      model_path: str,
      serving_binary: serving_bins.ServingBinary,
      serving_spec: infra_validator_pb2.ServingSpec):
    """Create a kubernetes model server runner.

    Args:
      model_path: An IV-flavored model path. (See model_path_utils.py)
      serving_binary: A ServingBinary to run.
      serving_spec: A ServingSpec instance.
    """
    assert serving_spec.WhichOneof('serving_platform') == 'kubernetes', (
        'ServingSpec configuration mismatch.')
    self._config = serving_spec.kubernetes

    self._model_path = model_path
    self._serving_binary = serving_binary
    self._serving_spec = serving_spec
    self._k8s_core_api = kube_utils.make_core_v1_api()
    if not kube_utils.is_inside_kfp():
      raise NotImplementedError(
          'KubernetesRunner should be running inside KFP.')
    self._executor_pod = kube_utils.get_current_kfp_pod(self._k8s_core_api)
    self._executor_container = _get_container_or_error(
        self._executor_pod,
        container_name=kube_utils.ARGO_MAIN_CONTAINER_NAME)
    self._namespace = kube_utils.get_kfp_namespace()
    self._label_dict = {
        _APP_KEY: _MODEL_SERVER_APP_LABEL,
    }
    # Pod name would be populated once creation request sent.
    self._pod_name = None
    # Endpoint would be populated once the Pod is running.
    self._endpoint = None

  def __repr__(self):
    return 'KubernetesRunner(image: {image}, pod_name: {pod_name})'.format(
        image=self._serving_binary.image,
        pod_name=self._pod_name)

  def GetEndpoint(self) -> str:
    assert self._endpoint is not None, (
        'self._endpoint is not ready. You should call Start() and '
        'WaitUntilRunning() first.')
    return self._endpoint

  def Start(self) -> None:
    assert not self._pod_name, (
        'You cannot start model server multiple times.')

    # We're creating a Pod rather than a Deployment as we're relying on
    # executor's retry mechanism for failure recovery, and the death of the Pod
    # should be regarded as a validation failure.
    pod = self._k8s_core_api.create_namespaced_pod(
        namespace=self._namespace,
        body=self._BuildPodManifest())
    self._pod_name = pod.metadata.name
    logging.info('Created Pod:\n%s', pod)

  def WaitUntilRunning(self, deadline: float) -> None:
    assert self._pod_name, (
        'Pod has not been created yet. You should call Start() first.')

    while time.time() < deadline:
      try:
        pod = self._k8s_core_api.read_namespaced_pod(
            name=self._pod_name,
            namespace=self._namespace)
      except rest.ApiException as e:
        logging.info('Continue polling after getting ApiException(%s)', e)
        time.sleep(_DEFAULT_POLLING_INTERVAL_SEC)
        continue
      # Pod phase is one of Pending, Running, Succeeded, Failed, or Unknown.
      # Succeeded and Failed indicates the pod lifecycle has reached its end,
      # while we expect the job to be running and hanging. Phase is Unknown if
      # the state of the pod could not be obtained, thus we can wait until we
      # confirm the phase.
      pod_phase = _PodPhase(pod.status.phase)
      if pod_phase == _PodPhase.RUNNING and pod.status.pod_ip:
        self._endpoint = '{}:{}'.format(pod.status.pod_ip,
                                        self._serving_binary.container_port)
        return
      if pod_phase.is_done:
        raise error_types.JobAborted(
            'Job has been aborted. (phase={})'.format(pod_phase))
      logging.info('Waiting for the pod to be running. (phase=%s)', pod_phase)
      time.sleep(_DEFAULT_POLLING_INTERVAL_SEC)

    raise error_types.DeadlineExceeded(
        'Deadline exceeded while waiting for pod to be running.')

  def Stop(self) -> None:
    try:
      self._DeleteModelServerPod()
    except:  # pylint: disable=broad-except, bare-except
      logging.warning('Error occurred while deleting the Pod. Please run the '
                      'following command to manually clean it up:\n\n'
                      'kubectl delete pod --namespace %s %s',
                      self._namespace, self._pod_name, exc_info=True)

  @retry.with_exponential_backoff(
      num_retries=_NUM_RETRIES,
      initial_delay_secs=_INITIAL_BACKOFF_DELAY_SEC,
      logger=logging.warning,
      retry_filter=_api_exception_retry_filter)
  def _DeleteModelServerPod(self):
    if self._pod_name is None:
      # No model server Pod has been created yet.
      logging.info('Server pod has not been created.')
      return
    try:
      logging.info('Deleting Pod (name=%s)', self._pod_name)
      self._k8s_core_api.delete_namespaced_pod(
          name=self._pod_name,
          namespace=self._namespace)
    except rest.ApiException as e:
      if e.status == 404:  # Pod is already deleted.
        logging.info('Pod (name=%s) does not exist.', self._pod_name)
        return
      else:
        raise

  def _BuildPodManifest(self) -> k8s_client.V1Pod:
    annotations = {}
    env_vars = []

    if isinstance(self._serving_binary, serving_bins.TensorFlowServing):
      env_vars_dict = self._serving_binary.MakeEnvVars(
          model_path=self._model_path)
      env_vars.extend(
          k8s_client.V1EnvVar(name=key, value=value)
          for key, value in env_vars_dict.items())

    if self._config.serving_pod_overrides:
      overrides = self._config.serving_pod_overrides
      if overrides.annotations:
        annotations.update(overrides.annotations)
      if overrides.env:
        env_vars.extend(_convert_to_kube_env(env) for env in overrides.env)

    service_account_name = (self._config.service_account_name or
                            self._executor_pod.spec.service_account_name)
    active_deadline_seconds = (self._config.active_deadline_seconds or
                               _DEFAULT_ACTIVE_DEADLINE_SEC)
    if active_deadline_seconds < 0:
      raise ValueError(
          'active_deadline_seconds should be > 0, but got '
          f'{active_deadline_seconds}.')

    result = k8s_client.V1Pod(
        metadata=k8s_client.V1ObjectMeta(
            generate_name=_MODEL_SERVER_POD_NAME_PREFIX,
            annotations=annotations,
            labels=self._label_dict,
            # Resources with ownerReferences are automatically deleted once all
            # its owners are deleted.
            owner_references=[
                k8s_client.V1OwnerReference(
                    api_version=self._executor_pod.api_version,
                    kind=self._executor_pod.kind,
                    name=self._executor_pod.metadata.name,
                    uid=self._executor_pod.metadata.uid,
                ),
            ],
        ),
        spec=k8s_client.V1PodSpec(
            containers=[
                k8s_client.V1Container(
                    name=_MODEL_SERVER_CONTAINER_NAME,
                    image=self._serving_binary.image,
                    env=env_vars,
                    volume_mounts=[],
                ),
            ],
            service_account_name=service_account_name,
            # No retry in case model server container failed. Retry will happen
            # at the outermost loop (executor.py).
            restart_policy=_RestartPolicy.NEVER.value,
            # This is a hard deadline for the model server container to ensure
            # the Pod is properly cleaned up even with an unexpected termination
            # of an infra validator. After the deadline, container will be
            # removed but Pod resource won't. This makes the Pod log visible
            # after the termination.
            active_deadline_seconds=active_deadline_seconds,
            volumes=[],
            # TODO(b/152002076): Add TTL controller once it graduates Beta.
            # ttl_seconds_after_finished=,
        )
    )

    self._SetupModelVolumeIfNeeded(result)

    return result

  def _FindVolumeMountForPath(self, path) -> Optional[k8s_client.V1VolumeMount]:
    if not os.path.exists(path):
      return None
    for mount in self._executor_container.volume_mounts:
      if _is_subdirectory(mount.mount_path, self._model_path):
        return mount
    return None

  def _SetupModelVolumeIfNeeded(self, pod_manifest: k8s_client.V1Pod):
    mount = self._FindVolumeMountForPath(self._model_path)
    if not mount:
      return
    [volume] = [v for v in self._executor_pod.spec.volumes
                if v.name == mount.name]
    if volume.persistent_volume_claim is None:
      raise NotImplementedError('Only PersistentVolumeClaim is allowed.')
    claim_name = volume.persistent_volume_claim.claim_name
    pvc = self._k8s_core_api.read_namespaced_persistent_volume_claim(
        name=claim_name,
        namespace=self._namespace)

    # PersistentVolumeClaim for pipeline root SHOULD have ReadWriteMany access
    # mode. Although it is allowed to mount ReadWriteOnce volume if Pods share
    # the Node, there's no guarantee the model server Pod will be launched in
    # the same Node.
    if all(access_mode != _AccessMode.READ_WRITE_MANY.value
           for access_mode in pvc.spec.access_modes):
      raise RuntimeError('Access mode should be ReadWriteMany.')

    logging.info('PersistentVolumeClaim %s will be mounted to %s.',
                 pvc, mount.mount_path)

    pod_manifest.spec.volumes.append(
        k8s_client.V1Volume(
            name=_MODEL_SERVER_MODEL_VOLUME_NAME,
            persistent_volume_claim=k8s_client
            .V1PersistentVolumeClaimVolumeSource(
                claim_name=claim_name,
                read_only=True)))
    container_manifest = _get_container_or_error(
        pod_manifest, container_name=_MODEL_SERVER_CONTAINER_NAME)
    container_manifest.volume_mounts.append(
        k8s_client.V1VolumeMount(
            name=_MODEL_SERVER_MODEL_VOLUME_NAME,
            mount_path=mount.mount_path,
            read_only=True,
        )
    )
