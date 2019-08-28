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

import os
import tensorflow as tf
from typing import Any, Dict, List, Text, Type
from ml_metadata.proto import metadata_store_pb2
from tfx import types
from tfx.components.base import base_component
from tfx.components.base import base_driver
from tfx.components.base import base_executor
from tfx.components.base import executor_spec
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.orchestration import publisher


class BaseComponentLauncher(object):
  """Responsible for launching driver, executor and publisher of component."""

  # pyformat: disable
  def __init__(self, component_info: data_types.ComponentInfo,
               driver_class: Type[base_driver.BaseDriver],
               component_executor_spec: executor_spec.ExecutorSpec,
               input_dict: Dict[Text, types.Channel],
               output_dict: Dict[Text, types.Channel],
               exec_properties: Dict[Text, Any],
               pipeline_info: data_types.PipelineInfo,
               driver_args: data_types.DriverArgs,
               metadata_connection_config: metadata_store_pb2.ConnectionConfig,
               additional_pipeline_args: Dict[Text, Any]):
    # pyformat: enable
    """Initialize a ComponentLauncher.

    Args:
      component_info: ComponentInfo of the component.
      driver_class: The driver class to run for this component.
      component_executor_spec: The executor spec to specify what to execute
        when launching this component.
      input_dict: Dictionary of input artifacts consumed by this component.
      output_dict: Dictionary of output artifacts produced by this component.
      exec_properties: Dictionary of execution properties.
      pipeline_info: An instance of data_types.PipelineInfo that holds pipeline
        properties.
      driver_args: An instance of data_types.DriverArgs that holds component
        specific driver args.
      metadata_connection_config: ML metadata connection config.
      additional_pipeline_args: Additional pipeline args, includes,
        - beam_pipeline_args: Beam pipeline args for beam jobs within executor.
          Executor will use beam DirectRunner as Default.
    """
    self._pipeline_info = pipeline_info
    self._component_info = component_info
    self._driver_args = driver_args

    self._driver_class = driver_class
    self._component_executor_spec = component_executor_spec

    self._input_dict = input_dict
    self._output_dict = output_dict
    self._exec_properties = exec_properties

    self._metadata_connection_config = metadata_connection_config
    self._additional_pipeline_args = additional_pipeline_args

  def _run_driver(self, input_dict: Dict[Text, types.Channel],
                  output_dict: Dict[Text, types.Channel],
                  exec_properties: Dict[Text, Any]
                 ) -> data_types.ExecutionDecision:
    """Prepare inputs, outputs and execution properties for actual execution."""
    tf.logging.info('Run driver for %s', self._component_info.component_id)

    with metadata.Metadata(self._metadata_connection_config) as m:
      driver = self._driver_class(metadata_handler=m)

      execution_decision = driver.pre_execution(
          input_dict=input_dict,
          output_dict=output_dict,
          exec_properties=exec_properties,
          driver_args=self._driver_args,
          pipeline_info=self._pipeline_info,
          component_info=self._component_info)

      return execution_decision

  # TODO(jyzhao): consider returning an execution result.
  def _run_executor(self, execution_id: int,
                    input_dict: Dict[Text, List[types.Artifact]],
                    output_dict: Dict[Text, List[types.Artifact]],
                    exec_properties: Dict[Text, Any]) -> None:
    """Execute underlying component implementation."""
    tf.logging.info('Run executor for %s', self._component_info.component_id)

    executor_context = base_executor.BaseExecutor.Context(
        beam_pipeline_args=self._additional_pipeline_args.get(
            'beam_pipeline_args'),
        tmp_dir=os.path.join(self._pipeline_info.pipeline_root, '.temp', ''),
        unique_id=str(execution_id))

    # TODO(hongyes): move this check to a specific method which can overrided
    # by subclasses.
    if not isinstance(self._component_executor_spec,
                      executor_spec.ExecutorClassSpec):
      raise TypeError(
          'component_executor_spec must be an instance of ExecutorClassSpec.')

    # Type hint of component will cause not-instantiable error as
    # ExecutorClassSpec.executor_class is Type[BaseExecutor] which has an
    # abstract function.
    executor = self._component_executor_spec.executor_class(
        executor_context)  # type: ignore

    executor.Do(input_dict, output_dict, exec_properties)

  def _run_publisher(self, use_cached_results: bool, execution_id: int,
                     input_dict: Dict[Text, List[types.Artifact]],
                     output_dict: Dict[Text, List[types.Artifact]]) -> None:
    """Publish execution result to ml metadata."""
    tf.logging.info('Run publisher for %s', self._component_info.component_id)

    with metadata.Metadata(self._metadata_connection_config) as m:
      p = publisher.Publisher(metadata_handler=m)
      p.publish_execution(
          execution_id=execution_id,
          input_dict=input_dict,
          output_dict=output_dict,
          use_cached_results=use_cached_results)

  def launch(self) -> int:
    """Execute the component, includes driver, executor and publisher.

    Returns:
      The execution id of the launch.
    """
    execution_decision = self._run_driver(self._input_dict, self._output_dict,
                                          self._exec_properties)

    if not execution_decision.use_cached_results:
      self._run_executor(execution_decision.execution_id,
                         execution_decision.input_dict,
                         execution_decision.output_dict,
                         execution_decision.exec_properties)

    self._run_publisher(execution_decision.use_cached_results,
                        execution_decision.execution_id,
                        execution_decision.input_dict,
                        execution_decision.output_dict)

    return execution_decision.execution_id


# TODO(ajaygopinathan): Combine with BaseComponentLauncher once we either:
#   - have a way to serialize/deserialize components, or
#   - have a clean way to use factory methods to create this class.
class ComponentLauncher(BaseComponentLauncher):
  """Responsible for launching driver, executor and publisher of component.

  Convenient subclass when given a concrete component to launch.
  """

  def __init__(self, component: base_component.BaseComponent,
               pipeline_info: data_types.PipelineInfo,
               driver_args: data_types.DriverArgs,
               metadata_connection_config: metadata_store_pb2.ConnectionConfig,
               additional_pipeline_args: Dict[Text, Any]):
    """Initialize a ComponentLauncher.

    Args:
      component: The component to launch.
      pipeline_info: An instance of data_types.PipelineInfo that holds pipeline
        properties.
      driver_args: An instance of data_types.DriverArgs that holds component
        specific driver args.
      metadata_connection_config: ML metadata connection config.
      additional_pipeline_args: Additional pipeline args, includes,
        - beam_pipeline_args: Beam pipeline args for beam jobs within executor.
          Executor will use beam DirectRunner as Default.
    """
    component_info = data_types.ComponentInfo(
        component_type=component.component_type,
        component_id=component.component_id)
    super(ComponentLauncher, self).__init__(
        component_info=component_info,
        driver_class=component.driver_class,
        component_executor_spec=component.executor_spec,
        input_dict=component.inputs.get_all(),
        output_dict=component.outputs.get_all(),
        exec_properties=component.exec_properties,
        pipeline_info=pipeline_info,
        driver_args=driver_args,
        metadata_connection_config=metadata_connection_config,
        additional_pipeline_args=additional_pipeline_args)
