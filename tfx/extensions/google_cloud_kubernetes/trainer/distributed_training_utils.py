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
"""Utilies for distributed training with GKE."""


from tfx.utils import kube_utils
from typing import Text
import kubernetes.client as client
from kubernetes.client.rest import ApiException
import json

_TFX_IMAGE = "gcr.io/tfx-eric/gpu-keras"

def create_worker_pods(name: Text="keras", job: Text="worker", num_workers: int=1, num_gpus_per_worker: int=1):
  api_instance = kube_utils.make_core_v1_api()
  worker_hosts = ["{}-{}-{}:5000".format(name, job, i) for i in range(num_workers)]
  for i in range(num_workers):
    pod = client.V1Pod(
      metadata=client.V1ObjectMeta(
        name=name + '-' + job + '-' + str(i),
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
            command=["python", "train.py"],
            # add other args
            args=[
              "--worker_index=" + str(i),
              "--worker_hosts=" + json.dumps(worker_hosts)
            ],
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

def create_worker_services(name="keras", job="worker", num_workers=1):
  api_instance = kube_utils.make_core_v1_api()
  for i in range(num_workers):
    service = client.V1Service(
      metadata=client.V1ObjectMeta(
        name=name + '-' + job + '-' + str(i),
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


if __name__ == '__main__':
  create_worker_pods(num_workers=8)
  create_worker_services(num_workers=8)
  # TODO(ericlege): clean up services