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
"""Model server runner for kubernetes runtime."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import string
import time
from typing import Text

from absl import logging
from kubernetes import client as k8s_client
from kubernetes.client import rest

from tfx.components.infra_validator import error_types
from tfx.components.infra_validator import serving_binary_lib
from tfx.components.infra_validator.model_server_runners import base_runner
from tfx.proto import infra_validator_pb2
from tfx.utils import kube_utils
from tfx.utils import path_utils
from tfx.utils import time_utils

_POLLING_INTERVAL_SEC = 5

# Length of the alphanumeric token. (62^10 ~= 8.39e17)
_TOKEN_LENGTH = 10
_APP_KEY = 'app'
_MODEL_SERVER_POD_NAME_FORMAT = 'infra-validator-model-server-{}'
_MODEL_SERVER_APP_LABEL = 'infra-validator-model-server'
_MODEL_SERVER_CONTAINER_NAME = 'model-server'

# Phases of the pod as described in
# https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#pod-phase.
_POD_PHASE_RUNNING = 'Running'
_POD_PHASE_SUCCEEDED = 'Succeeded'
_POD_PHASE_FAILED = 'Failed'

# PodSpec container restart policy as described in
# https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#restart-policy
_POD_CONTAINER_RESTART_POLICY_NEVER = 'Never'


def _random_alphanumeric_sequence(length: int) -> Text:
  alphanumeric = string.ascii_letters + string.digits
  return ''.join(random.choice(alphanumeric) for _ in range(length))


class KubernetesRunner(base_runner.BaseModelServerRunner):
  """A model server runner that launches model server in kubernetes cluster."""

  def __init__(
      self,
      standard_model_path: path_utils.StandardModelPath,
      serving_binary: serving_binary_lib.ServingBinary,
      serving_spec: infra_validator_pb2.ServingSpec):
    """Create a kubernetes model server runner.

    Args:
      standard_model_path: A StandardModelPath instance.
      serving_binary: A ServingBinary to run.
      serving_spec: A ServingSpec instance.
    """
    if serving_spec.WhichOneof('serving_platform') != 'kubernetes':
      raise error_types.IllegalState('ServingSpec configuration mismatch.')
    self._config = serving_spec.kubernetes

    self._standard_model_path = standard_model_path
    self._serving_binary = serving_binary
    self._serving_spec = serving_spec
    self._k8s_core_api = kube_utils.make_core_v1_api()
    if not kube_utils.is_inside_kfp():
      raise error_types.IllegalState(
          'KubernetesRunner should be running inside KFP.')
    self._executor_pod = kube_utils.get_current_kfp_pod(self._k8s_core_api)
    self._namespace = kube_utils.get_kfp_namespace()
    # Token is a unique identifier to distinguish this model server runner. In a
    # single kubernetes namespace there could be multiple pipelines running in
    # parallel. Each pipeline can have multiple infra validator instances, and
    # finally each infra validator can possibly run multiple model servers.
    self._token = _random_alphanumeric_sequence(_TOKEN_LENGTH)
    self._pod_name = _MODEL_SERVER_POD_NAME_FORMAT.format(self._token)
    self._label_dict = {
        _APP_KEY: _MODEL_SERVER_APP_LABEL,
    }
    self._deployment = None
    self._pod_created = False
    self._endpoint = None

  def __repr__(self):
    return 'KubernetesRunner(image: {image}, token: {token})'.format(
        image=self._serving_binary.image,
        token=self._token)

  def GetEndpoint(self) -> Text:
    if not self._endpoint:
      raise error_types.IllegalState(
          'self._endpoint is not ready. You should call Start() and '
          'WaitUntilRunning()first.')
    return self._endpoint

  def Start(self) -> None:
    if self._pod_created is not None:
      raise error_types.IllegalState(
          'You cannot start model server multiple times.')

    # We're creating a Pod rather than a Deployment as we're relying on
    # executor's retry mechanism for failure recovery, and the death of the Pod
    # should be regarded as a validation failure.
    manifest = self._BuildPodManifest()
    pod = self._k8s_core_api.create_namespaced_pod(
        namespace=self._namespace,
        body=manifest)
    logging.info('Created pod:\n%s', pod)

  def WaitUntilRunning(self, deadline: float) -> None:
    if not self._pod_created:
      raise error_types.IllegalState(
          'Pod is not yet created. You should call Start() first.')

    while time_utils.utc_timestamp() < deadline:
      try:
        pod = self._k8s_core_api.read_namespaced_pod(
            name=self._pod_name,
            namespace=self._namespace)
      except rest.ApiException as e:
        logging.info('Continue polling after getting ApiException(%s)', e)
        time.sleep(_POLLING_INTERVAL_SEC)
        continue
      # Pod phase is one of Pending, Running, Succeeded, Failed, or Unknown.
      # Succeeded and Failed indicates the pod lifecycle has reached its end,
      # while we expect the job to be running and hanging. Phase is Unknown if
      # the state of the pod could not be obtained, thus we can wait until we
      # confirm the phase.
      pod_phase = pod.status.phase
      if pod_phase == _POD_PHASE_RUNNING and pod.status.pod_ip:
        self._endpoint = '{}:{}'.format(pod.status.pod_ip,
                                        self._serving_binary.container_port)
        return
      if pod_phase in (_POD_PHASE_SUCCEEDED, _POD_PHASE_FAILED):
        raise error_types.JobAborted(
            'Job has been aborted. (phase={})'.format(pod_phase))
      logging.info('Waiting for the pod to be running. (phase=%s)', pod_phase)
      time.sleep(_POLLING_INTERVAL_SEC)

    raise error_types.DeadlineExceeded(
        'Deadline exceeded while waiting for pod to be running.')

  def Stop(self) -> None:
    try:
      logging.info('Deleting pod (name=%s)', self._pod_name)
      self._k8s_core_api.delete_namespaced_pod(
          name=self._pod_name,
          namespace=self._namespace)
    except rest.ApiException as e:
      if e.status == 404:
        logging.info('Pod (name=%s) does not exist.', self._pod_name)
      else:
        logging.error(e)
        logging.warning('Could not delete pod (name=%s).', self._pod_name)

  def _BuildPodManifest(self) -> k8s_client.V1Pod:
    if isinstance(self._serving_binary, serving_binary_lib.TensorFlowServing):
      env_vars = self._serving_binary.MakeEnvVars(
          model_base_path=self._standard_model_path.base_path)
      env_vars = [k8s_client.V1EnvVar(name=key, value=value)
                  for key, value in env_vars.items()]
    else:
      raise NotImplementedError('Unsupported serving binary {}'.format(
          type(self._serving_binary).__name__))

    service_account_name = self._executor_pod.spec.service_account_name
    if self._config.service_account_name:
      service_account_name = self._config.service_account_name

    image_pull_secrets = None
    if self._config.image_pull_secrets:
      image_pull_secrets = [
          k8s_client.V1LocalObjectReference(name=name)
          for name in self._config.image_pull_secrets]

    resources = None
    if self._config.resources:
      resources = k8s_client.V1ResourceRequirements()
      if self._config.resources.limits:
        resources.limits = dict(self._config.resources.limits)
      if self._config.resources.requests:
        resources.requests = dict(self._config.resources.requests)

    return k8s_client.V1Pod(
        metadata=k8s_client.V1ObjectMeta(
            name=self._pod_name,
            labels=self._label_dict,
            # Resource with ownerReferences are automatically deleted once all
            # its owners are deleted.
            owner_references=[
                k8s_client.V1OwnerReference(
                    api_version=self._executor_pod.api_version,
                    kind=self._executor_pod.kind,
                    name=self._executor_pod.metadata.name,
                    uid=self._executor_pod.metadata.uid)]),
        spec=k8s_client.V1PodSpec(
            containers=[
                k8s_client.V1Container(
                    name=_MODEL_SERVER_CONTAINER_NAME,
                    image=self._serving_binary.image,
                    resources=resources,
                    env=env_vars)],
            image_pull_secrets=image_pull_secrets,
            service_account_name=service_account_name,
            restart_policy=_POD_CONTAINER_RESTART_POLICY_NEVER))
