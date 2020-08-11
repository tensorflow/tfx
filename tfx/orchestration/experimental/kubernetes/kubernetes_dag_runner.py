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
"""Definition of kubernetes TFX runner."""

import absl
import datetime
import time
from typing import Optional, Text, Type

from ml_metadata.proto import metadata_store_pb2
from tfx.dsl.component.experimental import container_component
from tfx.components.base import base_node
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.orchestration import pipeline as tfx_pipeline
from tfx.orchestration import tfx_runner
from tfx.orchestration.config import base_component_config
from tfx.orchestration.config import config_utils
from tfx.orchestration.config import pipeline_config
from tfx.orchestration.kubeflow import node_wrapper
from tfx.orchestration.kubeflow import utils
from tfx.orchestration.launcher import base_component_launcher
from tfx.orchestration.launcher import in_process_component_launcher
from tfx.orchestration.launcher import kubernetes_component_launcher
from tfx.utils import json_utils, kube_utils
from google.protobuf import json_format
from kubernetes import client
import json

_CONTAINER_COMMAND = [
    'python', '/tfx-src/tfx/orchestration/experimental/kubernetes/container_entrypoint.py'
]

_DRIVER_COMMAND = [
    'python', '/tfx-src/tfx/orchestration/experimental/kubernetes/orchestrator_container_entrypoint.py'
]

# Suffix added to the component id to avoid MLMD conflict when
# registering this component.
_WRAPPER_SUFFIX = 'Wrapper'

_TFX_IMAGE = 'gcr.io/tfx-eric/tfx-dev'


def get_default_kubernetes_metadata_config(
) -> metadata_store_pb2.ConnectionConfig:
  """Returns the default metadata connection config for a kubernetes cluster.

  Returns:
    A config proto that will be serialized as JSON and passed to the running
    container so the TFX component driver is able to communicate with MLMD in
    a kubernetes cluster.
  """
  connection_config = metadata_store_pb2.ConnectionConfig()
  connection_config.mysql.host = 'mysql'
  connection_config.mysql.port = 3306
  connection_config.mysql.database = 'mysql'
  connection_config.mysql.user = 'root'
  connection_config.mysql.password = ''
  return connection_config


def _wrap_container_component(
    component: base_node.BaseNode,
    component_launcher_class:
        Type[base_component_launcher.BaseComponentLauncher],
    component_config: Optional[base_component_config.BaseComponentConfig],
    pipeline: tfx_pipeline.Pipeline,
) -> base_node.BaseNode:
  """Wrapper for container component.

  Args:
  component: Component to be executed.
  component_launcher_class: The class of the launcher to launch the
    component.
  component_config: component config to launch the component.
  pipeline: Logical pipeline that contains pipeline related information.
  """

  component_launcher_class_path = '.'.join([
      component_launcher_class.__module__, component_launcher_class.__name__
  ])

  serialized_component = utils.replace_placeholder(
      json_utils.dumps(node_wrapper.NodeWrapper(component)))

  arguments = [
      '--pipeline_name',
      pipeline.pipeline_info.pipeline_name,
      '--pipeline_root',
      pipeline.pipeline_info.pipeline_root,
      '--run_id',
      pipeline.pipeline_info.run_id,
      '--metadata_config',
      json_format.MessageToJson(
          message=get_default_kubernetes_metadata_config(),
          preserving_proto_field_name=True),
      '--beam_pipeline_args',
      json.dumps(pipeline.beam_pipeline_args),
      '--additional_pipeline_args',
      json.dumps(pipeline.additional_pipeline_args),
      '--component_launcher_class_path',
      component_launcher_class_path,
      '--serialized_component',
      serialized_component,
      '--component_config',
      json_utils.dumps(component_config),
  ]

  # Outputs/Parameters fields are not used as they are contained in
  # the serialized component.
  return container_component.create_container_component(
      name=component.id + _WRAPPER_SUFFIX,
      outputs={},
      parameters={},
      image=_TFX_IMAGE,
      command=_CONTAINER_COMMAND + arguments
  )()


def launch_container_component(
    component: base_node.BaseNode,
    component_launcher_class:
        Type[base_component_launcher.BaseComponentLauncher],
    component_config: base_component_config.BaseComponentConfig,
    pipeline: tfx_pipeline.Pipeline):
  """Use the kubernetes component launcher to launch the component.

  Args:
    component: Container component to be executed.
    component_config: component config to launch the component.
    component_launcher_class: The class of the launcher to launch the
      component.
    pipeline: Logical pipeline that contains pipeline related information.
  """
  driver_args = data_types.DriverArgs(enable_cache=pipeline.enable_cache)
  metadata_connection = metadata.Metadata(
      pipeline.metadata_connection_config)

  component_launcher = component_launcher_class.create(
      component=component,
      pipeline_info=pipeline.pipeline_info,
      driver_args=driver_args,
      metadata_connection=metadata_connection,
      beam_pipeline_args=pipeline.beam_pipeline_args,
      additional_pipeline_args=pipeline.additional_pipeline_args,
      component_config=component_config)
  absl.logging.info('Component %s is running.', component.id)
  component_launcher.launch()
  absl.logging.info('Component %s is finished.', component.id)


class KubernetesDagRunner(tfx_runner.TfxRunner):
  """Tfx runner on Kubernetes."""

  def __init__(self,
               config: Optional[pipeline_config.PipelineConfig] = None):
    """Initializes KubernetesDagRunner as a TFX orchestrator.

    Args:
      config: Optional pipeline config for customizing the launching of each
        component. Defaults to pipeline config that supports
        InProcessComponentLauncher and KubernetesComponentLauncher.
    """
    if config is None:
      config = pipeline_config.PipelineConfig(
          supported_launcher_classes=[
              in_process_component_launcher.InProcessComponentLauncher,
              kubernetes_component_launcher.KubernetesComponentLauncher,
          ],
      )
    super(KubernetesDagRunner, self).__init__(config)

  def run(self, pipeline: tfx_pipeline.Pipeline) -> None:
    """
    Args:
      pipeline: Logical pipeline containing pipeline args and components.
    """
    if not pipeline.pipeline_info.run_id:
      pipeline.pipeline_info.run_id = datetime.datetime.now().isoformat()

    if not kube_utils.is_inside_cluster():
      self._run_as_kubernetes_job(pipeline)
      return

    # TODO(ericlege): Support running components in parallel.
    ran_components = set()

    # Runs component in topological order.
    for component in pipeline.components:
      # Verify that components are in topological order.
      if hasattr(component, 'upstream_nodes') and component.upstream_nodes:
        for upstream_node in component.upstream_nodes:
          assert upstream_node in ran_components, ('Components is not in '
                                                   'topological order')

      (component_launcher_class,
       component_config) = config_utils.find_component_launch_info(
           self._config, component)

      # Check if the component is launchable as a container component.
      if kubernetes_component_launcher.KubernetesComponentLauncher.can_launch(
          component.executor_spec, component_config):
        launch_container_component(component,
                                   component_launcher_class,
                                   component_config,
                                   pipeline)
      # Otherwise, the component should be launchable with the in process
      # component launcher. wrap the component to a container component.
      elif in_process_component_launcher.InProcessComponentLauncher.can_launch(
          component.executor_spec, component_config):
        wrapped_component = _wrap_container_component(
            component=component,
            component_launcher_class=component_launcher_class,
            component_config=component_config,
            pipeline=pipeline
        )

        # Component launch info is updated by wrapping the component into a
        # container component. Therefore, these properties need to be reloaded.
        (wrapped_component_launcher_class,
         wrapped_component_config) = config_utils.find_component_launch_info(
             self._config, wrapped_component)

        launch_container_component(wrapped_component,
                                   wrapped_component_launcher_class,
                                   wrapped_component_config,
                                   pipeline)
      else:
        raise ValueError("Can not find suitable launcher for component.")

      ran_components.add(component)


  def _run_as_kubernetes_job(self, pipeline: tfx_pipeline.Pipeline) -> None:
    """Submits and runs a tfx pipeline from outside the cluster.

    Args:
      pipeline: Logical pipeline containing pipeline args and components.
    """
    serialized_pipeline = self._serialize_pipeline(pipeline)
    arguments = [
        '--serialized_pipeline',
        serialized_pipeline,
    ]
    batch_api = kube_utils.make_batch_v1_api()
    job_name = 'Job_' + pipeline.pipeline_info.run_id
    pod_label = kube_utils.sanitize_pod_name(job_name)
    container_name = 'pipeline-orchestrator'
    job = kube_utils.make_job_object(
        name=job_name,
        container_image=_TFX_IMAGE,
        command=_DRIVER_COMMAND + arguments,
        container_name=container_name,
        pod_labels={
            'job-name': pod_label,
        },
    )
    try:
      batch_api.create_namespaced_job(
          "default", job, pretty=True)
    except client.rest.ApiException as e:
      raise RuntimeError('Failed to submit job! \nReason: %s\nBody: %s' %
                         (e.reason, e.body))

    # Wait for pod to start
    orchestrator_pods = []
    core_api = kube_utils.make_core_v1_api()
    start_time = datetime.datetime.utcnow()
    while not orchestrator_pods and (datetime.datetime.utcnow() - start_time
                                     ).seconds < 300:
      try:
        orchestrator_pods = core_api.list_namespaced_pod(
            namespace='default',
            label_selector='job-name={}'.format(pod_label)
        ).items
      except client.rest.ApiException as e:
        if e.status != 404:
          raise RuntimeError('Unknown error! \nReason: %s\nBody: %s' %
                             (e.reason, e.body))
      time.sleep(1)

    # transient orchestrator should only have 1 pod
    orchestrator_pod = orchestrator_pods.pop()
    pod_name = orchestrator_pod.metadata.name

    #TODO(https://github.com/tensorflow/tfx/pull/2248): refactor protected members
    cond = kubernetes_component_launcher._pod_is_not_pending # pylint: disable=W0212
    absl.logging.info('Waiting for pod "default:%s" to start.', pod_name)
    kube_utils.wait_pod(
        core_api,
        pod_name,
        'default',
        exit_condition_lambda=cond,
        condition_description='non-pending status')

    # stream logs from pod
    absl.logging.info('Start log streaming for pod "default:%s".', pod_name)
    try:
      logs = core_api.read_namespaced_pod_log(
          name=pod_name,
          namespace='default',
          container=container_name,
          follow=True,
          _preload_content=False).stream()
    except client.rest.ApiException as e:
      raise RuntimeError(
          'Failed to stream the logs from the pod!\nReason: %s\nBody: %s' %
          (e.reason, e.body))

    for log in logs:
      absl.logging.info(log.decode().rstrip('\n'))

    cond = kubernetes_component_launcher._pod_is_done # pylint: disable=W0212
    resp = kube_utils.wait_pod(
        core_api,
        pod_name,
        'default',
        exit_condition_lambda=cond,
        condition_description='done state',
        exponential_backoff=True)

    if resp.status.phase == kube_utils.PodPhase.FAILED.value:
      raise RuntimeError('Pod "default:%s" failed with status "%s".' %
                         (pod_name, resp.status))

  def _serialize_pipeline(self, pipeline: tfx_pipeline.Pipeline) -> Text:
    """Serializes a TFX pipeline.

    To be replaced with the "portable core" of the unified TFX orchestrator:
    https://go/tfx-dsl-ir

    Args:
      pipeline: Logical pipeline containing pipeline args and components.

    Returns:
      Pipeline serialized as JSON string.
    """
    serialized_components = []
    for component in pipeline.components:
      serialized_components.append(utils.replace_placeholder(
          json_utils.dumps(node_wrapper.NodeWrapper(component))))
    return json.dumps({
        'pipeline_name': pipeline.pipeline_info.pipeline_name,
        'pipeline_root': pipeline.pipeline_info.pipeline_root,
        'enable_cache': pipeline.enable_cache,
        'components': serialized_components,
        'metadata_connection_config': json_format.MessageToJson(
            message=pipeline.metadata_connection_config,
            preserving_proto_field_name=True,
        ),
        'beam_pipeline_args': pipeline.beam_pipeline_args,
    })
