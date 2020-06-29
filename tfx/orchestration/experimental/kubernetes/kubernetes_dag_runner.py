# Lint as: python2, python3
# Copyright 2019 Google LLC. All Rights Reserved.
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

from tfx.components.base import base_node
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration import tfx_runner
from tfx.orchestration.config import base_component_config
from tfx.orchestration.config import config_utils
from tfx.orchestration.config import pipeline_config
from tfx.orchestration.launcher import base_component_launcher
from tfx.orchestration.launcher import docker_component_launcher
from tfx.orchestration.launcher import in_process_component_launcher
from tfx.orchestration.launcher import kubernetes_component_launcher
from tfx.utils import telemetry_utils


# class _ComponentAsDoFn(beam.DoFn):
#   """Wrap component as beam DoFn."""

#   def __init__(self, component: base_node.BaseNode,
#                component_launcher_class: Type[
#                    base_component_launcher.BaseComponentLauncher],
#                component_config: base_component_config.BaseComponentConfig,
#                tfx_pipeline: pipeline.Pipeline):
#     """Initialize the _ComponentAsDoFn.

#     Args:
#       component: Component that to be executed.
#       component_launcher_class: The class of the launcher to launch the
#         component.
#       component_config: component config to launch the component.
#       tfx_pipeline: Logical pipeline that contains pipeline related information.
#     """
#     driver_args = data_types.DriverArgs(enable_cache=tfx_pipeline.enable_cache)
#     metadata_connection = metadata.Metadata(
#         tfx_pipeline.metadata_connection_config)
#     self._component_launcher = component_launcher_class.create(
#         component=component,
#         pipeline_info=tfx_pipeline.pipeline_info,
#         driver_args=driver_args,
#         metadata_connection=metadata_connection,
#         beam_pipeline_args=tfx_pipeline.beam_pipeline_args,
#         additional_pipeline_args=tfx_pipeline.additional_pipeline_args,
#         component_config=component_config)
#     self._component_id = component.id

#   def process(self, element: Any, *signals: Iterable[Any]) -> None:
#     """Executes component based on signals.

#     Args:
#       element: a signal element to trigger the component.
#       *signals: side input signals indicate completeness of upstream components.
#     """
#     for signal in signals:
#       assert not list(signal), 'Signal PCollection should be empty.'
#     self._run_component()

#   def _run_component(self) -> None:
#     absl.logging.info('Component %s is running.', self._component_id)
#     self._component_launcher.launch()
#     absl.logging.info('Component %s is finished.', self._component_id)


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
              #in_process_component_launcher.InProcessComponentLauncher,
              #docker_component_launcher.DockerComponentLauncher,
              kubernetes_component_launcher.kubernetesComponentLauncher,
          ],
      )
    super(KubernetesDagRunner, self).__init__(config)
    self._beam_orchestrator_args = beam_orchestrator_args

  def run(self, tfx_pipeline: pipeline.Pipeline) -> None:
    """
    Args:
      tfx_pipeline: Logical pipeline containing pipeline args and components.
    """

    tfx_pipeline.pipeline_info.run_id = datetime.datetime.now().isoformat()

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
      
      # launch component
      _LaunchAsContainerComponent(component, component_launcher_class,
                                component_config, tfx_pipeline)._run_component()

      ran_components.add(component)
