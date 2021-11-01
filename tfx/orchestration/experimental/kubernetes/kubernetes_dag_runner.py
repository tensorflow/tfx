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
"""Definition of Kubernetes TFX runner."""

import datetime
import json
from typing import List, Optional, Type

import absl
from tfx.dsl.component.experimental import container_component
from tfx.dsl.components.base import base_node
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.orchestration import pipeline as tfx_pipeline
from tfx.orchestration import tfx_runner
from tfx.orchestration.config import base_component_config
from tfx.orchestration.config import config_utils
from tfx.orchestration.config import pipeline_config
from tfx.orchestration.experimental.kubernetes import kubernetes_remote_runner
from tfx.orchestration.kubeflow import node_wrapper
from tfx.orchestration.launcher import base_component_launcher
from tfx.orchestration.launcher import in_process_component_launcher
from tfx.orchestration.launcher import kubernetes_component_launcher
from tfx.utils import json_utils
from tfx.utils import kube_utils

from google.protobuf import json_format
from ml_metadata.proto import metadata_store_pb2

_CONTAINER_COMMAND = [
    'python', '-m',
    'tfx.orchestration.experimental.kubernetes.container_entrypoint'
]

# Suffix added to the component id to avoid MLMD conflict when
# registering this component.
_WRAPPER_SUFFIX = '.Wrapper'

_TFX_IMAGE = 'tensorflow/tfx'


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


def launch_container_component(
    component: base_node.BaseNode,
    component_launcher_class: Type[
        base_component_launcher.BaseComponentLauncher],
    component_config: base_component_config.BaseComponentConfig,
    pipeline: tfx_pipeline.Pipeline):
  """Use the kubernetes component launcher to launch the component.

  Args:
    component: Container component to be executed.
    component_launcher_class: The class of the launcher to launch the component.
    component_config: component config to launch the component.
    pipeline: Logical pipeline that contains pipeline related information.
  """
  driver_args = data_types.DriverArgs(enable_cache=pipeline.enable_cache)
  metadata_connection = metadata.Metadata(pipeline.metadata_connection_config)

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


class KubernetesDagRunnerConfig(pipeline_config.PipelineConfig):
  """Runtime configuration parameters specific to execution on Kubernetes."""

  def __init__(self,
               tfx_image: Optional[str] = None,
               supported_launcher_classes: Optional[List[Type[
                   base_component_launcher.BaseComponentLauncher]]] = None,
               **kwargs):
    """Creates a KubernetesDagRunnerConfig object.

    Args:
      tfx_image: The TFX container image to use in the pipeline.
      supported_launcher_classes: Optional list of component launcher classes
        that are supported by the current pipeline. List sequence determines the
        order in which launchers are chosen for each component being run.
      **kwargs: keyword args for PipelineConfig.
    """
    supported_launcher_classes = supported_launcher_classes or [
        in_process_component_launcher.InProcessComponentLauncher,
        kubernetes_component_launcher.KubernetesComponentLauncher,
    ]
    super().__init__(
        supported_launcher_classes=supported_launcher_classes, **kwargs)
    self.tfx_image = tfx_image or _TFX_IMAGE


class KubernetesDagRunner(tfx_runner.TfxRunner):
  """TFX runner on Kubernetes."""

  def __init__(self, config: Optional[KubernetesDagRunnerConfig] = None):
    """Initializes KubernetesDagRunner as a TFX orchestrator.

    Args:
      config: Optional pipeline config for customizing the launching of each
        component. Defaults to pipeline config that supports
        InProcessComponentLauncher and KubernetesComponentLauncher.
    """
    if config is None:
      config = KubernetesDagRunnerConfig()
    super().__init__(config)

  def run(self, pipeline: tfx_pipeline.Pipeline) -> None:
    """Deploys given logical pipeline on Kubernetes.

    Args:
      pipeline: Logical pipeline containing pipeline args and components.
    """
    if not pipeline.pipeline_info.run_id:
      pipeline.pipeline_info.run_id = datetime.datetime.now().isoformat()

    if not kube_utils.is_inside_cluster():
      kubernetes_remote_runner.run_as_kubernetes_job(
          pipeline=pipeline, tfx_image=self._config.tfx_image)
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
        launch_container_component(component, component_launcher_class,
                                   component_config, pipeline)
      # Otherwise, the component should be launchable with the in process
      # component launcher. wrap the component to a container component.
      elif in_process_component_launcher.InProcessComponentLauncher.can_launch(
          component.executor_spec, component_config):
        wrapped_component = self._wrap_container_component(
            component=component,
            component_launcher_class=component_launcher_class,
            component_config=component_config,
            pipeline=pipeline)

        # Component launch info is updated by wrapping the component into a
        # container component. Therefore, these properties need to be reloaded.
        (wrapped_component_launcher_class,
         wrapped_component_config) = config_utils.find_component_launch_info(
             self._config, wrapped_component)

        launch_container_component(wrapped_component,
                                   wrapped_component_launcher_class,
                                   wrapped_component_config, pipeline)
      else:
        raise ValueError('Can not find suitable launcher for component.')

      ran_components.add(component)

  def _wrap_container_component(
      self,
      component: base_node.BaseNode,
      component_launcher_class: Type[
          base_component_launcher.BaseComponentLauncher],
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

    Returns:
      A container component that runs the wrapped component upon execution.
    """

    component_launcher_class_path = '.'.join([
        component_launcher_class.__module__, component_launcher_class.__name__
    ])

    serialized_component = json_utils.dumps(node_wrapper.NodeWrapper(component))

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
        name=component.__class__.__name__,
        outputs={},
        parameters={},
        image=self._config.tfx_image,
        command=_CONTAINER_COMMAND + arguments)().with_id(component.id +
                                                          _WRAPPER_SUFFIX)
