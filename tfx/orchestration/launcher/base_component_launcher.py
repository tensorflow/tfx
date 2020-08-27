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
"""For component execution, includes driver, executor and publisher."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from typing import Any, Dict, List, Optional, Text

import absl
from six import with_metaclass

from tfx import types
from tfx.components.base import base_node
from tfx.components.base import executor_spec
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.orchestration import publisher
from tfx.orchestration.config import base_component_config


class BaseComponentLauncher(with_metaclass(abc.ABCMeta, object)):
  """Responsible for launching driver, executor and publisher of component."""

  def __init__(
      self,
      component: base_node.BaseNode,
      pipeline_info: data_types.PipelineInfo,
      driver_args: data_types.DriverArgs,
      metadata_connection: metadata.Metadata,
      beam_pipeline_args: List[Text],
      additional_pipeline_args: Dict[Text, Any],
      component_config: Optional[
          base_component_config.BaseComponentConfig] = None,
  ):
    """Initialize a BaseComponentLauncher.

    Args:
      component: The Tfx node to launch.
      pipeline_info: An instance of data_types.PipelineInfo that holds pipeline
        properties.
      driver_args: An instance of data_types.DriverArgs that holds component
        specific driver args.
      metadata_connection: ML metadata connection. The connection is expected to
        not be opened when given to this object.
      beam_pipeline_args: Pipeline arguments for Beam powered Components.
      additional_pipeline_args: Additional pipeline args.
      component_config: Optional component specific config to instrument
        launcher on how to launch a component.

    Raises:
      ValueError: when component and component_config are not launchable by the
      launcher.
    """
    self._pipeline_info = pipeline_info
    self._component_info = data_types.ComponentInfo(
        component_type=component.type,
        component_id=component.id,
        pipeline_info=self._pipeline_info)
    self._driver_args = driver_args

    self._driver_class = component.driver_class
    self._component_executor_spec = component.executor_spec

    self._input_dict = component.inputs.get_all()
    self._output_dict = component.outputs.get_all()
    self._exec_properties = component.exec_properties

    self._metadata_connection = metadata_connection
    self._beam_pipeline_args = beam_pipeline_args

    self._additional_pipeline_args = additional_pipeline_args
    self._component_config = component_config

    if not self.can_launch(self._component_executor_spec,
                           self._component_config):
      raise ValueError(
          'component.executor_spec with type "%s" and component config with'
          ' type "%s" are not launchable by "%s".' % (
              type(self._component_executor_spec).__name__,
              type(self._component_config).__name__,
              type(self).__name__,
          ))

  @classmethod
  def create(
      cls,
      component: base_node.BaseNode,
      pipeline_info: data_types.PipelineInfo,
      driver_args: data_types.DriverArgs,
      metadata_connection: metadata.Metadata,
      beam_pipeline_args: List[Text],
      additional_pipeline_args: Dict[Text, Any],
      component_config: Optional[
          base_component_config.BaseComponentConfig] = None,
  ) -> 'BaseComponentLauncher':
    """Initialize a ComponentLauncher directly from a BaseComponent instance.

    This class method is the contract between `TfxRunner` and
    `BaseComponentLauncher` to support launcher polymorphism. Sublcass of this
    class must make sure it can be initialized by the method.

    Args:
      component: The component to launch.
      pipeline_info: An instance of data_types.PipelineInfo that holds pipeline
        properties.
      driver_args: An instance of data_types.DriverArgs that holds component
        specific driver args.
      metadata_connection: ML metadata connection. The connection is expected to
        not be opened when given to this object.
      beam_pipeline_args: Pipeline arguments for Beam powered Components.
      additional_pipeline_args: Additional pipeline args.
      component_config: Optional component specific config to instrument
        launcher on how to launch a component.

    Returns:
      A new instance of component launcher.
    """
    return cls(
        component=component,
        pipeline_info=pipeline_info,
        driver_args=driver_args,
        metadata_connection=metadata_connection,
        beam_pipeline_args=beam_pipeline_args,
        additional_pipeline_args=additional_pipeline_args,
        component_config=component_config)  # pytype: disable=not-instantiable

  @classmethod
  @abc.abstractmethod
  def can_launch(
      cls, component_executor_spec: executor_spec.ExecutorSpec,
      component_config: base_component_config.BaseComponentConfig) -> bool:
    """Checks if the launcher can launch the executor spec with an optional component config."""
    raise NotImplementedError

  def _run_driver(
      self, input_dict: Dict[Text,
                             types.Channel], output_dict: Dict[Text,
                                                               types.Channel],
      exec_properties: Dict[Text, Any]) -> data_types.ExecutionDecision:
    """Prepare inputs, outputs and execution properties for actual execution."""

    with self._metadata_connection as m:
      driver = self._driver_class(metadata_handler=m)

      execution_decision = driver.pre_execution(
          input_dict=input_dict,
          output_dict=output_dict,
          exec_properties=exec_properties,
          driver_args=self._driver_args,
          pipeline_info=self._pipeline_info,
          component_info=self._component_info)

      return execution_decision

  @abc.abstractmethod
  # TODO(jyzhao): consider returning an execution result.
  def _run_executor(self, execution_id: int,
                    input_dict: Dict[Text, List[types.Artifact]],
                    output_dict: Dict[Text, List[types.Artifact]],
                    exec_properties: Dict[Text, Any]) -> None:
    """Execute underlying component implementation."""
    raise NotImplementedError

  def _run_publisher(self, output_dict: Dict[Text,
                                             List[types.Artifact]]) -> None:
    """Publish execution result to ml metadata."""

    with self._metadata_connection as m:
      p = publisher.Publisher(metadata_handler=m)
      p.publish_execution(
          component_info=self._component_info, output_artifacts=output_dict)

  def launch(self) -> data_types.ExecutionInfo:
    """Execute the component, includes driver, executor and publisher.

    Returns:
      The execution decision of the launch.
    """
    absl.logging.info('Running driver for %s',
                      self._component_info.component_id)
    execution_decision = self._run_driver(self._input_dict, self._output_dict,
                                          self._exec_properties)

    if not execution_decision.use_cached_results:
      absl.logging.info('Running executor for %s',
                        self._component_info.component_id)
      self._run_executor(execution_decision.execution_id,
                         execution_decision.input_dict,
                         execution_decision.output_dict,
                         execution_decision.exec_properties)

    absl.logging.info('Running publisher for %s',
                      self._component_info.component_id)
    self._run_publisher(output_dict=execution_decision.output_dict)

    return data_types.ExecutionInfo(
        input_dict=execution_decision.input_dict,
        output_dict=execution_decision.output_dict,
        exec_properties=execution_decision.exec_properties,
        execution_id=execution_decision.execution_id)
