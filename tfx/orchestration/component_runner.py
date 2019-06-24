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
from typing import Any, Dict, List, Optional, Text
from ml_metadata.proto import metadata_store_pb2
from tfx.components.base import base_component
from tfx.components.base import base_executor
from tfx.orchestration import data_types
from tfx.utils import types


class ComponentRunner(object):
  """Runner for component."""

  # TODO(jyzhao): consider provide another spec layer for the params.
  def __init__(
      self,
      component: base_component.BaseComponent,
      pipeline_run_id: Text,
      pipeline_name: Text,
      pipeline_root: Text,
      # Placeholder for future driver/publisher related params.
      metadata_connection_config: metadata_store_pb2.ConnectionConfig = None,  # pylint: disable=unused-argument
      enable_cache: bool = False,  # pylint: disable=unused-argument
      additional_pipeline_args: Dict[Text, Any] = None):
    """Initialize a ComponentRunner.

    Args:
      component: Component that to be executed.
      pipeline_run_id: The unique id of current pipeline run.
      pipeline_name: The unique name of this pipeline.
      pipeline_root: The root path of the pipeline outputs.
      metadata_connection_config: ML metadata connection config.
      enable_cache: Whether to enable cache functionality.
      additional_pipeline_args: Additional pipeline args, includes,
        - beam_pipeline_args: Beam pipeline args for beam jobs within executor.
          Executor will use beam DirectRunner as Default.
    """
    # TODO(jyzhao): change to component_id after cl/254503786
    self._name = component.component_name
    self._pipeline_run_id = pipeline_run_id

    self._executor_class = component.executor_class

    self._input_dict = dict(
        (k, list(v.get())) for k, v in component.inputs.get_all().items())
    self._output_dict = dict(
        (k, list(v.get())) for k, v in component.outputs.get_all().items())
    self._exec_properties = component.exec_properties

    # TODO(jyzhao): documentation of root path and its usage.
    self._project_path = os.path.join(pipeline_root, pipeline_name)
    self._additional_pipeline_args = additional_pipeline_args or {}

  def _run_driver(self, input_dict: Dict[Text, List[types.TfxArtifact]],
                  output_dict: Dict[Text, List[types.TfxArtifact]],
                  exec_properties: Dict[Text, Any]
                 ) -> data_types.ExecutionDecision:
    """Prepare inputs, outputs and execution properties for actual execution."""
    # TODO(jyzhao): support driver after go/tfx-oss-artifact-passing.
    tf.logging.info('Run driver for %s', self._name)
    # Return a fake result that makes sure execution_decision.execution_needed
    # is true to always trigger the executor.
    return data_types.ExecutionDecision(input_dict, output_dict,
                                        exec_properties, 1)

  def _run_executor(self, input_dict: Dict[Text, List[types.TfxArtifact]],
                    output_dict: Dict[Text, List[types.TfxArtifact]],
                    exec_properties: Dict[Text, Any]) -> None:
    """Execute underlying component implementation."""
    tf.logging.info('Run executor for %s', self._name)

    executor_context = base_executor.BaseExecutor.Context(
        beam_pipeline_args=self._additional_pipeline_args.get(
            'beam_pipeline_args'),
        tmp_dir=os.path.join(self._project_path, '.temp', ''),
        # TODO(jyzhao): change to execution id that generated in driver.
        unique_id=self._pipeline_run_id)

    # Type hint of component will cause not-instantiable error as
    # component.executor is Type[BaseExecutor] which has an abstract function.
    executor = self._executor_class(executor_context)  # type: ignore

    executor.Do(input_dict, output_dict, exec_properties)

  def _run_publisher(self, is_executed: bool, execution_id: Optional[int],
                     input_dict: Dict[Text, List[types.TfxArtifact]],
                     output_dict: Dict[Text, List[types.TfxArtifact]],
                     exec_properties: Dict[Text, Any]) -> None:
    """Publish execution result to ml metadata."""
    tf.logging.info('Run publisher for %s', self._name)
    tf.logging.info('Execution decision: %s', is_executed)
    tf.logging.info('Execution id: %s', execution_id)
    tf.logging.info('Inputs: %s', input_dict)
    tf.logging.info('Outputs: %s', output_dict)
    tf.logging.info('Execution properties: %s', exec_properties)
    # TODO(jyzhao): support publisher after go/tfx-oss-artifact-passing.

  def run(self) -> None:
    """Execute the component, includes driver, executor and publisher."""
    execution_decision = self._run_driver(self._input_dict, self._output_dict,
                                          self._exec_properties)

    if execution_decision.execution_needed:
      self._run_executor(execution_decision.input_dict,
                         execution_decision.output_dict,
                         execution_decision.exec_properties)

    self._run_publisher(execution_decision.execution_needed,
                        execution_decision.execution_id,
                        execution_decision.input_dict,
                        execution_decision.output_dict,
                        execution_decision.exec_properties)
