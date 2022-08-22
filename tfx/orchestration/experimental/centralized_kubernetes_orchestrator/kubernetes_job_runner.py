# Copyright 2022 Google LLC. All Rights Reserved.
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
"""Kubernetes job runner for orchestrator.

Runner which executes given pipeline components as a Kubernetes job.
"""
import abc
import datetime
import random
import string
import time

from absl import logging
from kubernetes import client as k8s_client
from tfx.orchestration.experimental.core import task_scheduler
from tfx.orchestration.python_execution_binary import python_execution_binary_utils
from tfx.utils import kube_utils
from tfx.utils import status as status_lib

_COMMAND = [
    'python',
    '-m',
    'tfx.orchestration.experimental.centralized_kubernetes_orchestrator.entrypoint',
]

_DEFAULT_POLLING_INTERVAL_SEC = 2
_JOB_CREATION_TIMEOUT = 300


def _generate_component_name_suffix() -> str:
  letters = string.ascii_lowercase
  return '-' + ''.join(random.choice(letters) for i in range(10))


class JobExceptionError(Exception):
  """Exception error class to handle exceptions while running Kubernetes job."""

  def __init__(self, message: str):
    super().__init__(message)
    self.msg = message


class KubernetesJobRunner(abc.ABC):
  """A Kubernetes job runner that launches and executes pipeline components in kubernetes cluster."""

  def __init__(self,
               tfx_image,
               job_prefix,
               container_name,
               name_space='default',
               stream_logs=False):
    """Create a kubernetes model server runner.

    Args:
      tfx_image: container image for tfx.
      job_prefix: prefix for the job. Unique hash will follow as suffix.
      container_name: name of the container.
      name_space: namespace of the run.
      stream_logs: whether to stream logs from the pod.
    """
    self._image = tfx_image
    self._k8s_core_api = kube_utils.make_core_v1_api()
    self._namespace = name_space
    self._container_name = container_name
    self._job_name = kube_utils.sanitize_pod_name(
        job_prefix + _generate_component_name_suffix())
    # Time to delete the job after completion.
    self.ttl_seconds = 5
    # Pod name would be populated once creation request sent.
    self._pod_name = None
    self._stream_pod_logs = stream_logs

  def run(self, execution_info,
          executable_spec) -> task_scheduler.TaskSchedulerResult:
    """Execute component in the pod."""

    try:
      self._create_job(execution_info, executable_spec)
      self._wait_until_pod_is_runnable()
      if self._stream_pod_logs:
        self._stream_logs()
      self._wait_until_completion()
      return task_scheduler.TaskSchedulerResult(
          status=status_lib.Status(code=status_lib.Code.OK),
          output=task_scheduler.ExecutorNodeOutput())
    except k8s_client.rest.ApiException as e:
      # TODO(b/240237394): Error type specification.
      msg = 'Unable to run job. \nReason: %s\nBody: %s' % (
          e.reason if not None else '', e.body if not None else '')
      logging.info(msg)
      return task_scheduler.TaskSchedulerResult(
          status=status_lib.Status(code=status_lib.Code.CANCELLED, message=msg))
    except JobExceptionError as e:
      logging.info(e.msg)
      return task_scheduler.TaskSchedulerResult(
          status=status_lib.Status(
              code=status_lib.Code.CANCELLED, message=e.msg))

  def _create_job(self, execution_info, executable_spec) -> None:
    """Create a job and wait for the pod to be runnable."""

    assert not self._pod_name, ('You cannot start a job multiple times.')
    serialized_execution_info = python_execution_binary_utils.serialize_execution_info(
        execution_info)
    serialized_executable_spec = python_execution_binary_utils.serialize_executable_spec(
        executable_spec)

    run_arguments = [
        '--tfx_execution_info_b64',
        serialized_execution_info,
        '--tfx_python_class_executable_spec_b64',
        serialized_executable_spec,
    ]
    orchestrator_commands = _COMMAND + run_arguments

    batch_api = kube_utils.make_batch_v1_api()
    job = kube_utils.make_job_object(
        name=self._job_name,
        container_image=self._image,
        command=orchestrator_commands,
        container_name=self._container_name,
        pod_labels={
            'job-name': self._job_name,
        },
        ttl_seconds_after_finished=self.ttl_seconds,
    )
    batch_api.create_namespaced_job(self._namespace, job, pretty=True)
    logging.info('Job %s created!', self._job_name)

  def _wait_until_pod_is_runnable(self) -> None:
    """Wait for the pod to be created and runnable."""

    assert self._job_name, ('You should first create a job to run.')
    orchestrator_pods = []
    start_time = datetime.datetime.utcnow()
    # Wait for the kubernetes job to launch a pod.
    while (datetime.datetime.utcnow() -
           start_time).seconds < _JOB_CREATION_TIMEOUT:
      orchestrator_pods = self._k8s_core_api.list_namespaced_pod(
          namespace='default',
          label_selector='job-name={}'.format(self._job_name)).items
      try:
        orchestrator_pods = self._k8s_core_api.list_namespaced_pod(
            namespace='default',
            label_selector='job-name={}'.format(self._job_name)).items
      except k8s_client.rest.ApiException as e:
        if e.status != 404:
          raise e
        time.sleep(_DEFAULT_POLLING_INTERVAL_SEC)
      if len(orchestrator_pods) != 1:
        continue
      pod = orchestrator_pods.pop()
      pod_phase = kube_utils.PodPhase(pod.status.phase)
      if pod_phase == kube_utils.PodPhase.RUNNING and pod.status.pod_ip:
        self._pod_name = pod.metadata.name
        logging.info('Pod created with name %s', self._pod_name)
        return
      if pod_phase.is_done:
        raise JobExceptionError(
            message='Job has been aborted. Please restart for execution.')
      time.sleep(_DEFAULT_POLLING_INTERVAL_SEC)
    raise JobExceptionError(
        message='Deadline exceeded while waiting for pod to be running.')

  def _stream_logs(self) -> None:
    """Stream logs from orchestrator pod."""
    logging.info('Start log streaming for pod %s:%s.', self._namespace,
                 self._pod_name)
    logs = self._k8s_core_api.read_namespaced_pod_log(
        name=self._pod_name,
        namespace='default',
        container=self._container_name,
        follow=True,
        _preload_content=False).stream()
    for log in logs:
      logging.info(log.decode().rstrip('\n'))

  def _wait_until_completion(self) -> None:
    """Wait until the processs is completed."""
    pod = kube_utils.wait_pod(
        self._k8s_core_api,
        self._pod_name,
        self._namespace,
        exit_condition_lambda=kube_utils.pod_is_done,
        condition_description='done state',
        exponential_backoff=True)
    pod_phase = kube_utils.PodPhase(pod.status.phase)
    if pod_phase == kube_utils.PodPhase.FAILED:
      raise JobExceptionError(message='Pod "%s" failed with status "%s".' %
                              (self._pod_name, pod.status))
    if pod_phase.is_done:
      logging.info('Job completed! Ending log streaming for pod %s:%s.',
                   self._namespace, self._pod_name)

    if self.ttl_seconds:
      logging.info('Job %s will be deleted after %d seconds.', self._job_name,
                   self.ttl_seconds)
    else:
      logging.info(
          'To delete the job, please run the following command:\n\n'
          'kubectl delete jobs/%s', self._job_name)
