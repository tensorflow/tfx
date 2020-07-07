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
"""Definition of Beam TFX runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
from typing import Any, Iterable, List, Optional, Text, Type

import absl
import apache_beam as beam

from ml_metadata.proto import metadata_store_pb2
from tfx.dsl.component.experimental import container_component
from tfx.dsl.component.experimental import placeholders
from tfx.types import standard_artifacts
from tfx.components.base import base_node
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration import tfx_runner
from tfx.orchestration.config import base_component_config
from tfx.orchestration.config import config_utils
from tfx.orchestration.config import pipeline_config
from tfx.orchestration.kubeflow import node_wrapper
from tfx.orchestration.kubeflow import utils
from tfx.orchestration.launcher import base_component_launcher
from tfx.orchestration.launcher import docker_component_launcher
from tfx.orchestration.launcher import in_process_component_launcher
from tfx.orchestration.launcher import kubernetes_component_launcher
from tfx.utils import telemetry_utils, json_utils, kube_utils
from google.protobuf import json_format
import json

_CONTAINER_COMMAND = [
    'python', '/tfx-src/tfx/orchestration/experimental/kubernetes/container_entrypoint.py'
]

_DRIVER_COMMAND = [
    'python', '/tfx-src/tfx/orchestration/experimental/kubernetes/driver_container_entrypoint.py'
]

_TFX_IMAGE = "gcr.io/tfx-eric/tfx-dev"

def _getDefaultConnectionConfig() -> metadata_store_pb2.ConnectionConfig:
  connection_config = metadata_store_pb2.ConnectionConfig()
  connection_config.mysql.host = 'mysql'
  connection_config.mysql.port = 3306
  connection_config.mysql.database = 'mysql'
  connection_config.mysql.user = 'root'
  connection_config.mysql.password = ''
  return connection_config

def _wrapContainerComponent(
    component: base_node.BaseNode,
    component_launcher_class: Type[base_component_launcher.BaseComponentLauncher],
    component_config: Optional[base_component_config.BaseComponentConfig],
    tfx_pipeline: pipeline.Pipeline
  ) -> base_node.BaseNode:
  """wrapper for container component"""

  # Reference: tfx.orchestration.kubeflow.base_component
  component_launcher_class_path = '.'.join([
      component_launcher_class.__module__, component_launcher_class.__name__
  ])

  serialized_component = utils.replace_placeholder(
      json_utils.dumps(node_wrapper.NodeWrapper(component)))

  arguments = [
    '--pipeline_name',
    tfx_pipeline.pipeline_info.pipeline_name,
    '--pipeline_root',
    tfx_pipeline.pipeline_info.pipeline_root,
    '--run_id',
    tfx_pipeline.pipeline_info.run_id,
    '--metadata_config',
    json_format.MessageToJson(
        message=_getDefaultConnectionConfig(), preserving_proto_field_name=True),
    '--beam_pipeline_args',
    json.dumps(tfx_pipeline.beam_pipeline_args),
    '--additional_pipeline_args',
    json.dumps(tfx_pipeline.additional_pipeline_args),
    '--component_launcher_class_path',
    component_launcher_class_path,
    '--serialized_component',
    serialized_component,
    '--component_config',
    json_utils.dumps(component_config),
  ]

  # outputs/parameters fields are ignored as they will come with the serialized component
  return container_component.create_container_component(
    name=component.get_id(component._instance_name),
    outputs={},
    parameters={},
    image=_TFX_IMAGE,
    command=_CONTAINER_COMMAND + arguments
  )()

class _LaunchAsContainerComponent():
  """wrapper for kubernetes_component_launcher"""
  def __init__(self, component: base_node.BaseNode,
               component_launcher_class: Type[
                   base_component_launcher.BaseComponentLauncher],
               component_config: base_component_config.BaseComponentConfig,
               tfx_pipeline: pipeline.Pipeline):
    """Initialize the _LaunchAsContainerComponent.

    Args:
      component: Component that to be executed.
      component_launcher_class: The class of the launcher to launch the
        component.
      component_config: component config to launch the component.
      tfx_pipeline: Logical pipeline that contains pipeline related information.
    """
    driver_args = data_types.DriverArgs(enable_cache=tfx_pipeline.enable_cache)
    metadata_connection = metadata.Metadata(
        tfx_pipeline.metadata_connection_config)

    """def create(
      cls,
      component: base_node.BaseNode,
      pipeline_info: data_types.PipelineInfo,
      driver_args: data_types.DriverArgs,
      metadata_connection: metadata.Metadata,
      beam_pipeline_args: List[Text],
      additional_pipeline_args: Dict[Text, Any],
      component_config: Optional[
          base_component_config.BaseComponentConfig] = None,
  ) -> 'BaseComponentLauncher':"""

    self._component_launcher = component_launcher_class.create(
        component=component,
        pipeline_info=tfx_pipeline.pipeline_info,
        driver_args=driver_args,
        metadata_connection=metadata_connection,
        beam_pipeline_args=tfx_pipeline.beam_pipeline_args,
        additional_pipeline_args=tfx_pipeline.additional_pipeline_args,
        component_config=component_config)
    self._component_id = component.id

  def _run_component(self) -> None:
    absl.logging.info('Component %s is running.', self._component_id)
    self._component_launcher.launch()
    absl.logging.info('Component %s is finished.', self._component_id)



class KubernetesDagRunner(tfx_runner.TfxRunner):
  """Tfx runner on Beam."""

  def __init__(self,
               beam_orchestrator_args: Optional[List[Text]] = None,
               config: Optional[pipeline_config.PipelineConfig] = None):
    """Initializes BeamDagRunner as a TFX orchestrator.

    Args:
      beam_orchestrator_args: beam args for the beam orchestrator. Note that
        this is different from the beam_pipeline_args within
        additional_pipeline_args, which is for beam pipelines in components.
      config: Optional pipeline config for customizing the launching of each
        component. Defaults to pipeline config that supports
        InProcessComponentLauncher and DockerComponentLauncher.
    """
    if config is None:
      config = pipeline_config.PipelineConfig(
          supported_launcher_classes=[
              in_process_component_launcher.InProcessComponentLauncher,
              #docker_component_launcher.DockerComponentLauncher,
              kubernetes_component_launcher.KubernetesComponentLauncher,
          ],
      )
    super(KubernetesDagRunner, self).__init__(config)
    self._beam_orchestrator_args = beam_orchestrator_args

  def run(self, tfx_pipeline: pipeline.Pipeline) -> None:
    """
    Args:
      tfx_pipeline: Logical pipeline containing pipeline args and components.
    """
    if not tfx_pipeline.pipeline_info.run_id:
      tfx_pipeline.pipeline_info.run_id = datetime.datetime.now().isoformat()

    if not kube_utils.is_inside_cluster():
      return self._run_as_kubernetes_job(tfx_pipeline)

    # TODO(ericlege) kubernetesComponentLauncher wait pod is blocking, so can use multithread to workaround for callback
    ran_components = set()

    # a serial implementation for running components in topological order
    for component in tfx_pipeline.components:
      component_id = component.id

    # verify that components are in topological order, Pipeline constructor should have automatically sorted them 
      if component.upstream_nodes:
        for upstream_node in component.upstream_nodes:
          assert upstream_node in ran_components, ('Components is not in '
                                                    'topological order')

      (component_launcher_class,
        component_config) = config_utils.find_component_launch_info(
            self._config, component)

      # check if the component is launchable as a containerComponent. If not, perform an on the fly conversion step. Should we forward I/O with a pre traversal?
      if not kubernetes_component_launcher.KubernetesComponentLauncher.can_launch(component.executor_spec, component_config):
        wrapped_component = _wrapContainerComponent(
          component=component,
          component_launcher_class=component_launcher_class,
          component_config=component_config,
          tfx_pipeline=tfx_pipeline
        )
        # reload properties
        (wrapped_component_launcher_class,
        wrapped_component_config) = config_utils.find_component_launch_info(
            self._config, wrapped_component)
        # launch wrapped component
        _LaunchAsContainerComponent(wrapped_component, wrapped_component_launcher_class,
                                    wrapped_component_config, tfx_pipeline)._run_component()
      
      else:
        # launch component
        _LaunchAsContainerComponent(component, component_launcher_class,
                                    component_config, tfx_pipeline)._run_component()

      # record the ran component. Always record the unwrapped component
      ran_components.add(component)


  def _run_as_kubernetes_job(self, tfx_pipeline: pipeline.Pipeline) -> None:
    """Submits and runs a tfx pipeline from outside the cluster

    Args:
      tfx_pipeline: Logical pipeline containing pipeline args and components.
    """
    _serialized_pipeline = self._serialize_pipeline(tfx_pipeline)
    arguments = [
        '--pipeline_name',
        _serialized_pipeline,
    ]
    _api_instance = kube_utils.make_batch_v1_api()
    _job = kube_utils.make_job_object(
        name='Job_' + tfx_pipeline.pipeline_info.run_id,
        container_image=_TFX_IMAGE,
        command=_DRIVER_COMMAND + arguments,
      )
    _api_response = _api_instance.create_namespaced_job("default", _job, pretty=True)
    absl.logging.info('Submitted Job to Kubernetes: %s', _api_response)


  def _serialize_pipeline(self, tfx_pipeline: pipeline.Pipeline) -> Text:
    """Serializes a TFX pipeline.
    
    To be replaced with the "portable core" of the unified TFX orchestrator:
    https://go/tfx-dsl-ir
    TODO(ericlege): support this for pipelines with container components

    Args:
      tfx_pipeline: Logical pipeline containing pipeline args and components.
    
    Returns:
      Serialized pipeline
    """
    serialzed_components = []
    for component in tfx_pipeline._components:
      serialzed_components.append(utils.replace_placeholder(
        json_utils.dumps(node_wrapper.NodeWrapper(component))))
    return json.dumps({
      'pipeline_name': tfx_pipeline.pipeline_info.pipeline_name,
      'pipeline_root': tfx_pipeline.pipeline_info.pipeline_root,
      'enable_cache':tfx_pipeline.enable_cache,
      'components': serialzed_components,
      'metadata_connection_config':json_format.MessageToJson(
          message=tfx_pipeline.metadata_connection_config,
          preserving_proto_field_name=True,
      ),
      'beam_pipeline_args': tfx_pipeline.beam_pipeline_args,
    })
