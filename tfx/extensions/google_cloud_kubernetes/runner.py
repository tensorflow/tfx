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

import json
from typing import Any, Dict, List, Text

from absl import logging

from tfx import types
from tfx.types import artifact_utils
from tfx.utils import kube_utils
from kubernetes.client.rest import ApiException
import kubernetes.client as client

# For maintenance, see:
# https://gist.github.com/Eric-Le-Ge/a7ef6c5ae66d4af9cc886536d6724175
_TFX_IMAGE = "gcr.io/tfx-eric/gpu-tfx"

_COMMAND = ["python", "-m", "tfx.scripts.run_executor"]


def _build_pod_names(num_workers: int, unique_id: Text) -> List[Text]:
  return ['training-worker-{}-{}'.format(unique_id,
                                         i) for i in range(num_workers)]


def _build_service_names(num_workers: int, unique_id: Text) -> List[Text]:
  return ['training-service-{}-{}'.format(unique_id,
                                          i) for i in range(num_workers)]


def _pod_is_done(resp: client.V1Pod):
  return kube_utils.PodPhase(resp.status.phase).is_done


def create_worker_pods(job_args: List[Text],
                       training_inputs: Dict[Text, Any],
                       unique_id: Text):
  """Create worker pods for multi-worker training."""
  num_workers = training_inputs.get('num_workers', 1)
  num_gpus_per_worker = training_inputs.get('num_gpus_per_worker', 0)

  api_instance = kube_utils.make_core_v1_api()
  service_names = _build_service_names(num_workers=num_workers,
                                       unique_id=unique_id)
  pod_names = _build_pod_names(num_workers=num_workers, unique_id=unique_id)
  worker_hosts = ['{}:5000'.format(svc_name) for svc_name in service_names]

  # TODO(ericlege): consider using a jinja2 template instead
  for i in range(num_workers):
    tf_config = json.dumps({
        'cluster': {
            'worker': worker_hosts
        },
        'task': {'type': 'worker', 'index': i}
    })
    pod = client.V1Pod(
        metadata=client.V1ObjectMeta(
            name=pod_names[i],
            labels={
                'name': 'training',
                'id': unique_id,
                'task': str(i),
            },
        ),
        spec=client.V1PodSpec(
            containers=[
                client.V1Container(
                    name='worker-pod',
                    image=_TFX_IMAGE,
                    command=_COMMAND,
                    args=job_args,
                    security_context=client.V1SecurityContext(
                        privileged=True,
                    ),
                    env=[
                        client.V1EnvVar(
                            name='TF_CONFIG',
                            value=tf_config,
                        ),
                    ],
                    ports=[
                        client.V1ContainerPort(
                            container_port=5000,
                        ),
                    ],
                    resources=client.V1ResourceRequirements(
                        limits={
                            'nvidia.com/gpu': num_gpus_per_worker,
                        },
                    ) if num_gpus_per_worker > 0 else None,
                ),
            ],
            restart_policy=kube_utils.RestartPolicy.NEVER.value,
        ),
    )
    try:
      api_instance.create_namespaced_pod(namespace='default', body=pod)
    except ApiException as e:
      logging.error(
          'Exception when calling CoreV1Api->create_namespaced_pod: %s' % e)

  logging.info('created {} worker pods'.format(num_workers))


def create_worker_services(training_inputs: Dict[Text, Any],
                           unique_id: Text):
  """Create worker services for multi-worker training."""
  num_workers = training_inputs.get('num_workers', 1)
  service_names = _build_service_names(num_workers=num_workers,
                                       unique_id=unique_id)
  api_instance = kube_utils.make_core_v1_api()

  # TODO(ericlege): consider using a jinja2 template instead
  for i in range(num_workers):
    service = client.V1Service(
        metadata=client.V1ObjectMeta(
            name=service_names[i],
        ),
        spec=client.V1ServiceSpec(
            selector={
                'name': 'training',
                'id': unique_id,
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
      api_instance.create_namespaced_service(namespace='default', body=service)
    except ApiException as e:
      logging.error(
          'Exception when calling CoreV1Api->create_namespaced_service: %s' % e)
  logging.info('created {} worker services'.format(num_workers))


def delete_worker_services(training_inputs: Dict[Text, Any],
                           unique_id: Text):
  """Clean up worker services deployed to the kubernetes cluster."""
  num_workers = training_inputs.get('num_workers', 1)
  service_names = _build_service_names(num_workers=num_workers,
                                       unique_id=unique_id)
  api_instance = kube_utils.make_core_v1_api()
  for service_name in service_names:
    try:
      api_instance.delete_namespaced_service(namespace='default',
                                             name=service_name)
    except ApiException as e:
      logging.error(
          'Exception when calling CoreV1Api->delete_namespaced_service: %s' % e)
  logging.info('Deleted {} worker services'.format(num_workers))


def start_gke_training(input_dict: Dict[Text, List[types.Artifact]],
                       output_dict: Dict[Text, List[types.Artifact]],
                       exec_properties: Dict[Text,
                                             Any], executor_class_path: Text,
                       training_inputs: Dict[Text,
                                             Any], unique_id: Text):
  """Start a trainer job on Google Kubernetes Engine (GKE).

  This is done by forwarding the inputs/outputs/exec_properties to the
  tfx.scripts.run_executor module on a AI Platform training job interpreter.

  Args:
    input_dict: Passthrough input dict for tfx.components.Trainer.executor.
    output_dict: Passthrough input dict for tfx.components.Trainer.executor.
    exec_properties: Passthrough input dict for tfx.components.Trainer.executor.
    executor_class_path: class path for TFX core default trainer.
    training_inputs: Training input argument for GKE.
      'num_workers', and 'num_gpus_per_worker' will be consumed.

  Returns:
    None
  Raises:
    RuntimeError: if the Google Kubernetes Engine training job failed/cancelled.
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
  create_worker_services(training_inputs=training_inputs, unique_id=unique_id)

  # launch the worker pods
  create_worker_pods(job_args=job_args,
                     training_inputs=training_inputs,
                     unique_id=unique_id)

  # wait for finish.
  num_workers = training_inputs.get('num_workers', 1)
  pod_names = _build_pod_names(unique_id=unique_id,
                               num_workers=num_workers)
  resp = kube_utils.wait_pod(core_api=kube_utils.make_core_v1_api(),
                             pod_name=pod_names[0], # chief
                             namespace='default',
                             exit_condition_lambda=_pod_is_done,
                             condition_description='Chief finished',
                             timeout_sec=1200, # wait for autoscaler
                             exponential_backoff=True,)
  if resp.status.phase == kube_utils.PodPhase.FAILED.value:
    raise RuntimeError('Pod "%s:%s" failed with status "%s".' %
                       ('default', pod_names[0], resp.status))

  # clean up
  delete_worker_services(training_inputs=training_inputs, unique_id=unique_id)

  # GKE training complete
  #logging.info('Job \'%s\' successful.', job_name)
  logging.info('Job successful')
