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
from typing import Any, Dict, List, Text
from ml_metadata.proto import metadata_store_pb2
from tfx.components.base import base_component
from tfx.components.base import base_executor
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.orchestration import publisher
from tfx.utils import channel
from tfx.utils import types


class ComponentLauncher(object):
  """Responsible for launching driver, executor and publisher of component."""

  def __init__(self, component: base_component.BaseComponent,
               pipeline_info: data_types.PipelineInfo,
               driver_args: data_types.DriverArgs,
               metadata_connection_config: metadata_store_pb2.ConnectionConfig,
               additional_pipeline_args: Dict[Text, Any]):
    """Initialize a ComponentLauncher.

    Args:
      component: Component that to be executed.
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
    self._component_info = data_types.ComponentInfo(
        component_type=component.component_type,
        component_id=component.component_id)
    self._driver_args = driver_args

    self._driver_class = component.driver_class
    self._executor_class = component.executor_class

    self._input_dict = component.inputs.get_all()
    self._output_dict = component.outputs.get_all()
    self._exec_properties = component.exec_properties

    self._metadata_connection_config = metadata_connection_config
    self._additional_pipeline_args = additional_pipeline_args

  def _run_driver(self, input_dict: Dict[Text, channel.Channel],
                  output_dict: Dict[Text, channel.Channel],
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
                    input_dict: Dict[Text, List[types.TfxArtifact]],
                    output_dict: Dict[Text, List[types.TfxArtifact]],
                    exec_properties: Dict[Text, Any]) -> None:
    """Execute underlying component implementation."""
    tf.logging.info('Run executor for %s', self._component_info.component_id)

    executor_context = base_executor.BaseExecutor.Context(
        beam_pipeline_args=self._additional_pipeline_args.get(
            'beam_pipeline_args'),
        tmp_dir=os.path.join(self._pipeline_info.pipeline_root, '.temp', ''),
        unique_id=str(execution_id))

    # Type hint of component will cause not-instantiable error as
    # component.executor is Type[BaseExecutor] which has an abstract function.
    executor = self._executor_class(executor_context)  # type: ignore

    executor.Do(input_dict, output_dict, exec_properties)

  def _run_publisher(self, use_cached_results: bool, execution_id: int,
                     input_dict: Dict[Text, List[types.TfxArtifact]],
                     output_dict: Dict[Text, List[types.TfxArtifact]]) -> None:
    """Publish execution result to ml metadata."""
    tf.logging.info('Run publisher for %s', self._component_info.component_id)

    with metadata.Metadata(self._metadata_connection_config) as m:
      p = publisher.Publisher(metadata_handler=m)
      p.publish_execution(
          execution_id=execution_id,
          input_dict=input_dict,
          output_dict=output_dict,
          use_cached_results=use_cached_results)

  def launch(self) -> None:
    """Execute the component, includes driver, executor and publisher."""
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
