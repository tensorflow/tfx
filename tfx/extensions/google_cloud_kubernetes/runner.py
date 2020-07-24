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
"""Helper class to start TFX multi-worker training jobs on GKE."""

import datetime
import json
import sys
import time
from typing import Any, Dict, List, Optional, Text

from absl import logging
import tensorflow as tf

from tfx import types
from tfx import version
from tfx.components.trainer import constants
from tfx.types import artifact_utils
from tfx.utils import telemetry_utils
from tfx.utils import kube_utils
from tfx.orchestration.launcher import kubernetes_component_launcher

import kubernetes.client as client
from kubernetes.client.rest import ApiException

#TODO: change
_TFX_IMAGE = "gcr.io/tfx-eric/gpu-tfx"

_COMMAND = "python /tfx-src/tfx/scripts/run_executor.py"

def _get_pod_name(name: Text="keras", job: Text="worker", index:int=0):
  return name + '-' + job + '-' + str(index)

def create_worker_pods(job_args:List[Text], training_inputs: Dict[Text,
                       Any], name: Text="keras", job: Text="worker"):
  num_workers = training_inputs.get("num_workers", 1)
  num_gpus_per_worker = training_inputs.get("num_gpus_per_worker", 0)
  api_instance = kube_utils.make_core_v1_api()
  worker_hosts = ["{}:5000".format(_get_pod_name(name, job, i)) for i in range(num_workers)]
  for i in range(num_workers):
    pod = client.V1Pod(
      metadata=client.V1ObjectMeta(
        name=_get_pod_name(name, job, i),
        labels={
          'name': name,
          'job': job,
          'task': str(i),
        },
      ),
      spec=client.V1PodSpec(
        containers=[
          client.V1Container(
            name='worker-pod',
            image=_TFX_IMAGE,
            # replace with file download
            command=_COMMAND,
            # add other args
            args=job_args,
            ports=[
              client.V1ContainerPort(
                container_port=5000,
              ),
            ],
            resources=client.V1ResourceRequirements(
              limits={
                'nvidia.com/gpu': num_gpus_per_worker,
              } if num_gpus_per_worker > 0 else {},
            ),
          ),
        ],
        restart_policy=kube_utils.RestartPolicy.NEVER.value,
      ),
    )
    try:
      api_response = api_instance.create_namespaced_pod(namespace='default', body=pod)
    except ApiException as e:
      print("Exception when calling CoreV1Api->create_namespaced_pod: %s\n" % e)
  print("created {} worker pods".format(num_workers))


def create_worker_services(training_inputs: Dict[Text,
                       Any], name="keras", job="worker"):
  num_workers = training_inputs.get("num_workers", 1)
  api_instance = kube_utils.make_core_v1_api()
  for i in range(num_workers):
    service = client.V1Service(
      metadata=client.V1ObjectMeta(
        name=_get_pod_name(name, job, i),
      ),
      spec=client.V1ServiceSpec(
        selector={
          'name': name,
          'job': job,
          'task': str(i),
        },
        ports=[
          client.V1ServicePort(
            port=5000,
          ),
        ],
      ),
    )
    try:
      api_response = api_instance.create_namespaced_service(namespace='default', body=service)
    except ApiException as e:
      # TODO(ericlege): use absl
      print("Exception when calling CoreV1Api->create_namespaced_service: %s\n" % e)
  print("created {} worker services".format(num_workers))


def delete_worker_services(training_inputs: Dict[Text,
                       Any], name="keras", job="worker"):
  num_workers = training_inputs.get("num_workers", 1)
  api_instance = kube_utils.make_core_v1_api()
  for i in range(num_workers):
    service_name = name + '-' + job + '-' + str(i),
    try:
      api_response = api_instance.delete_namespaced_service(namespace='default', name=service_name)
    except ApiException as e:
      # TODO(ericlege): use absl
      print("Exception when calling CoreV1Api->delete_namespaced_service: %s\n" % e)
  print("Deleted {} worker services".format(num_workers))

def start_gke_training(input_dict: Dict[Text, List[types.Artifact]],
                       output_dict: Dict[Text, List[types.Artifact]],
                       exec_properties: Dict[Text,
                                             Any], executor_class_path: Text,
                       training_inputs: Dict[Text,
                                             Any]):
  """Start a trainer job on Google Kubernetes Engine (GKE).

  This is done by forwarding the inputs/outputs/exec_properties to the
  tfx.scripts.run_executor module on a AI Platform training job interpreter.

  Args:
    input_dict: Passthrough input dict for tfx.components.Trainer.executor.
    output_dict: Passthrough input dict for tfx.components.Trainer.executor.
    exec_properties: Passthrough input dict for tfx.components.Trainer.executor.
    executor_class_path: class path for TFX core default trainer.
    training_inputs: Training input argument for GKE.
      'pythonModule', 'pythonVersion' and 'runtimeVersion' will be inferred. For
      the full set of parameters, refer to
      https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#TrainingInput

  Returns:
    None
  Raises:
    RuntimeError: if the Google Cloud AI Platform training job failed/cancelled.
  """
  training_inputs = training_inputs.copy()

  json_inputs = artifact_utils.jsonify_artifact_dict(input_dict)
  logging.info('json_inputs=\'%s\'.', json_inputs)
  json_outputs = artifact_utils.jsonify_artifact_dict(output_dict)
  logging.info('json_outputs=\'%s\'.', json_outputs)
  json_exec_properties = json.dumps(exec_properties, sort_keys=True)
  logging.info('json_exec_properties=\'%s\'.', json_exec_properties)


  # We use custom containers to launch training on GKE, which invokes
  # the specified image using the container's entrypoint. The default
  # entrypoint for TFX containers is to call scripts/run_executor.py. The
  # arguments below are passed to this run_executor entry to run the executor
  # specified in `executor_class_path`.
  job_args = [
      '--executor_class_path', executor_class_path, '--inputs', json_inputs,
      '--outputs', json_outputs, '--exec-properties', json_exec_properties
  ]

  # launch the services
  create_worker_services(training_inputs=training_inputs)

  # launch the worker pods
  create_worker_pods(job_args=job_args, training_inputs=training_inputs)

  # wait for finish. TODO: not use protected members
  exit_condition = kubernetes_component_launcher._pod_is_done
  kubernetes_component_launcher.KubernetesComponentLauncher()._wait_pod(self,
                core_api=kube_utils.make_core_v1_api(),
                pod_name=_get_pod_name(), # chief
                namespace="default",
                exit_condition_lambda=exit_condition,
                condition_description="Chief finished",
                timeout_sec=1200) # wait for autoscaler
  
  # clean up
  delete_worker_services(training_inputs=training_inputs)

  # GKE training complete
  logging.info('Job \'%s\' successful.', job_name)

