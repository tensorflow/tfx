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
"""Utilities for the kubernetes related functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import enum
import os
from typing import Text

from kubernetes import client as k8s_client
from kubernetes import config as k8s_config

# Name of the main container that Argo workflow launches. As KFP internally uses
# Argo, container name for the KFP Pod is also the same.
# https://github.com/argoproj/argo/blob/master/workflow/common/common.go#L14
ARGO_MAIN_CONTAINER_NAME = 'main'

# Set of environment variables that are set in the KubeFlow Pipelines pods.
KFP_POD_NAME = 'KFP_POD_NAME'
KFP_NAMESPACE = 'KFP_NAMESPACE'


class PodPhase(enum.Enum):
  """Phase of the Kubernetes Pod.

  Pod phases are defined in
  https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#pod-phase.
  """

  PENDING = 'Pending'
  RUNNING = 'Running'
  SUCCEEDED = 'Succeeded'
  FAILED = 'Failed'
  UNKNOWN = 'Unknown'

  @property
  def is_done(self):
    return self == self.SUCCEEDED or self == self.FAILED


class RestartPolicy(enum.Enum):
  """Restart policy of the Kubernetes Pod container.

  Restart policies are defined in
  https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#restart-policy
  """

  ALWAYS = 'Always'
  ON_FAILURE = 'OnFailure'
  NEVER = 'Never'


class PersistentVolumeAccessMode(enum.Enum):
  """Access mode of the Kubernetes Persistent Volume.

  Access modes are defined in
  https://kubernetes.io/docs/concepts/storage/persistent-volumes/#access-modes
  """

  READ_WRITE_ONCE = 'ReadWriteOnce'
  READ_ONLY_MANY = 'ReadOnlyMany'
  READ_WRITE_MANY = 'ReadWriteMany'


class _KubernetesClientFactory(object):
  """Factory class for creating kubernetes API client."""

  def __init__(self):
    self._config_loaded = False
    self._inside_cluster = False

  @property
  def inside_cluster(self):
    """Whether current environment is inside the kubernetes cluster."""
    if not self._config_loaded:
      self._LoadConfig()
    return self._inside_cluster

  def _LoadConfig(self) -> None:  # pylint: disable=invalid-name
    """Load the kubernetes client config.

    Depending on the environment (whether it is inside the running kubernetes
    cluster or remote host), different location will be searched for the config
    file. The loaded config will be used as a default value for the clients this
    factory is creating.

    If config is already loaded, it is a no-op.

    Raises:
      kubernetes.config.ConfigException: If fails to locate configuration in
          current environment.
    """
    try:
      # If this code is running inside Kubernetes Pod, service account admission
      # controller [1] sets volume mounts in which the service account tokens
      # and certificates exists, and it can be loaded using
      # `load_incluster_config()`.
      #
      # [1]
      # https://kubernetes.io/docs/reference/access-authn-authz/service-accounts-admin/#service-account-admission-controller
      self._inside_cluster = True
      k8s_config.load_incluster_config()
    except k8s_config.ConfigException:
      # If loading incluster config fails, it means we're not running the code
      # inside Kubernetes cluster. We try to load ~/.kube/config file, or the
      # filename from the KUBECONFIG environment variable.
      # It will raise kubernetes.config.ConfigException if no kube config file
      # is found.
      self._inside_cluster = False
      k8s_config.load_kube_config()

    self._config_loaded = True

  def MakeCoreV1Api(self) -> k8s_client.CoreV1Api:  # pylint: disable=invalid-name
    """Make a kubernetes CoreV1Api client."""
    if not self._config_loaded:
      self._LoadConfig()
    return k8s_client.CoreV1Api()

_factory = _KubernetesClientFactory()


def make_core_v1_api() -> k8s_client.CoreV1Api:
  """Make a kubernetes CoreV1Api client."""
  return _factory.MakeCoreV1Api()


def is_inside_cluster() -> bool:
  """Whether current running environment is inside the kubernetes cluster."""
  return _factory.inside_cluster


def is_inside_kfp() -> bool:
  """Whether current running environment is inside the KFP runtime."""
  return (
      is_inside_cluster()
      and KFP_POD_NAME in os.environ
      and KFP_NAMESPACE in os.environ
  )


def get_kfp_namespace() -> Text:
  """Get kubernetes namespace for the KFP.

  Raises:
    RuntimeError: If KFP pod cannot be determined from the environment, i.e.
        this program is not running inside the KFP.
  Returns:
    The namespace of the KFP app, to which the pod this program is running on
    belongs.
  """
  try:
    return os.environ[KFP_NAMESPACE]
  except KeyError:
    raise RuntimeError('Cannot determine KFP namespace from the environment.')


def get_current_kfp_pod(client: k8s_client.CoreV1Api) -> k8s_client.V1Pod:
  """Get manifest of the KFP pod in which this program is running.

  Args:
    client: A kubernetes CoreV1Api client.
  Raises:
    RuntimeError: If KFP pod cannot be determined from the environment, i.e.
        this program is not running inside the KFP.
  Returns:
    The manifest of the pod this program is running on.
  """
  try:
    namespace = os.environ[KFP_NAMESPACE]
    pod_name = os.environ[KFP_POD_NAME]
    return client.read_namespaced_pod(name=pod_name, namespace=namespace)
  except KeyError:
    raise RuntimeError('Cannot determine KFP pod from the environment.')
