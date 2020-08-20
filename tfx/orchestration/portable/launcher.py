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
"""This module defines a generic Launcher for all TFleX nodes."""

import traceback
from typing import Any, Dict, List, Optional, Text, Type, TypeVar

from absl import logging
import attr
import tensorflow as tf
from tfx import types
from tfx.orchestration import metadata
from tfx.orchestration.portable import base_executor_operator
from tfx.orchestration.portable import cache_utils
from tfx.orchestration.portable import execution_publish_utils
from tfx.orchestration.portable import inputs_utils
from tfx.orchestration.portable import outputs_utils
from tfx.orchestration.portable import python_executor_operator
from tfx.orchestration.portable.mlmd import context_lib
from tfx.proto.orchestration import execution_result_pb2
from tfx.proto.orchestration import local_deployment_config_pb2
from tfx.proto.orchestration import pipeline_pb2

from google.protobuf import message
from ml_metadata.proto import metadata_store_pb2

# Subclasses of BaseExecutorOperator
ExecutorOperator = TypeVar(
    'ExecutorOperator', bound=base_executor_operator.BaseExecutorOperator)

DEFAULT_EXECUTOR_OPERATORS = {
    local_deployment_config_pb2.ExecutableSpec.PythonClassExecutableSpec:
        python_executor_operator.PythonExecutorOperator
}


# TODO(b/165359991): Restore 'auto_attribs=True' once we drop Python3.5 support.
@attr.s
class _PrepareExecutionResult:
  """A wrapper class using as the return value of _prepare_execution()."""

  # The information used by executor operators.
  execution_info = attr.ib(
      type=base_executor_operator.ExecutionInfo, default=None)
  # Contexts of the execution, usually used by Publisher.
  contexts = attr.ib(type=List[metadata_store_pb2.Context], default=None)
  # TODO(b/156126088): Update the following documentation when this bug is
  # closed.
  # Whether an execution is needed. An execution is not needed when:
  # 1) Not all the required input are ready.
  # 2) The input value doesn't meet the driver's requirement.
  # 3) Cache result is used.
  is_execution_needed = attr.ib(type=bool, default=False)


class Launcher(object):
  """Launcher is the main entrance of nodes in TFleX.

     It handles TFX internal details like artifact resolving, execution
     triggering and result publishing.
  """

  def __init__(
      self,
      pipeline_node: pipeline_pb2.PipelineNode,
      mlmd_connection: metadata.Metadata,
      pipeline_info: pipeline_pb2.PipelineInfo,
      pipeline_runtime_spec: pipeline_pb2.PipelineRuntimeSpec,
      executor_spec: Optional[message.Message] = None,
      custom_driver_spec: Optional[message.Message] = None,
      platform_spec: Optional[message.Message] = None,
      custom_executor_operators: Optional[Dict[Any,
                                               Type[ExecutorOperator]]] = None):
    """Initializes a Launcher.

    Args:
      pipeline_node: The specification of the node that this launcher lauches.
      mlmd_connection: ML metadata connection.
      pipeline_info: The information of the pipeline that this node runs in.
      pipeline_runtime_spec: The runtime information of the pipeline that this
        node runs in.
      executor_spec: Specification for the executor of the node. This is
        expected for all components nodes. This will be used to determine the
        specific ExecutorOperator class to be used to execute and will be passed
        into ExecutorOperator.
      custom_driver_spec: Specification for custom driver. This is expected only
        for advanced use cases.
      platform_spec: Platform config that will be used as auxiliary info of the
        node execution. This will be passed to ExecutorOperator along with the
        `executor_spec`.
      custom_executor_operators: a map of ExecutorSpec to its ExecutorOperation
        implementation.

    Raises:
      ValueError: when component and component_config are not launchable by the
      launcher.
    """
    del custom_driver_spec

    self._pipeline_node = pipeline_node
    self._mlmd_connection = mlmd_connection
    self._pipeline_info = pipeline_info
    self._pipeline_runtime_spec = pipeline_runtime_spec
    self._executor_operators = {}
    self._executor_operators.update(DEFAULT_EXECUTOR_OPERATORS)
    self._executor_operators.update(custom_executor_operators or {})

    self._executor_operator = self._executor_operators[type(executor_spec)](
        executor_spec, platform_spec)
    self._output_resolver = outputs_utils.OutputsResolver(
        pipeline_node=self._pipeline_node,
        pipeline_info=self._pipeline_info,
        pipeline_runtime_spec=self._pipeline_runtime_spec)

  def _prepare_execution(self) -> _PrepareExecutionResult:
    """Prepare inputs, outputs and execution properties for actual execution."""
    # TODO(b/150979622): handle the edge case that the component get evicted
    # between successful pushlish and stateful working dir being clean up.
    # Otherwise following retries will keep failing because of duplicate
    # publishes.
    with self._mlmd_connection as m:
      # 1.Prepares all contexts.
      contexts = context_lib.register_contexts_if_not_exists(
          metadata_handler=m, node_contexts=self._pipeline_node.contexts)

      # 2. Resolves inputs an execution properties.
      exec_properties = inputs_utils.resolve_parameters(
          node_parameters=self._pipeline_node.parameters)
      input_artifacts = inputs_utils.resolve_input_artifacts(
          metadata_handler=m, node_inputs=self._pipeline_node.inputs)
      # 3. If not all required inputs are met. Return ExecutionInfo with
      # is_execution_needed being false. No publish will happen so down stream
      # nodes won't be triggered.
      if input_artifacts is None:
        return _PrepareExecutionResult(
            execution_info=base_executor_operator.ExecutionInfo(),
            contexts=contexts,
            is_execution_needed=False)

      # 4. Registers execution in metadata.
      execution = execution_publish_utils.register_execution(
          metadata_handler=m,
          execution_type=self._pipeline_node.node_info.type,
          contexts=contexts,
          input_artifacts=input_artifacts,
          exec_properties=exec_properties)

      # 5. Resolve output
      output_artifacts = self._output_resolver.generate_output_artifacts(
          execution.id)
      # TODO(b/150979622): support custom driver

      # 6. Check cached result
      cache_context = cache_utils.get_cache_context(
          metadata_handler=m,
          pipeline_node=self._pipeline_node,
          pipeline_info=self._pipeline_info,
          input_artifacts=input_artifacts,
          output_artifacts=output_artifacts,
          parameters=exec_properties)
      contexts.append(cache_context)
      cached_outputs = cache_utils.get_cached_outputs(
          metadata_handler=m, cache_context=cache_context)

      # 7. Should cache be used?
      if (self._pipeline_node.execution_options.caching_options.enable_cache and
          cached_outputs):
        # Publishes cache result
        execution_publish_utils.publish_cached_execution(
            metadata_handler=m,
            contexts=contexts,
            execution_id=execution.id,
            output_artifacts=cached_outputs)
        return _PrepareExecutionResult(
            execution_info=base_executor_operator.ExecutionInfo(
                execution_metadata=execution),
            contexts=contexts,
            is_execution_needed=False)

      # 8. Going to trigger executor.
      return _PrepareExecutionResult(
          execution_info=base_executor_operator.ExecutionInfo(
              execution_metadata=execution,
              input_dict=input_artifacts,
              output_dict=output_artifacts,
              exec_properties=exec_properties,
              executor_output_uri=self._output_resolver.get_executor_output_uri(
                  execution.id),
              stateful_working_dir=(
                  self._output_resolver.get_stateful_working_directory()),
              pipeline_node=self._pipeline_node,
              pipeline_info=self._pipeline_info),
          contexts=contexts,
          is_execution_needed=True)

  def _run_executor(
      self, execution_info: base_executor_operator.ExecutionInfo
  ) -> execution_result_pb2.ExecutorOutput:
    """Executes underlying component implementation."""

    logging.info('Going to run a new execution: %s', execution_info)

    outputs_utils.make_output_dirs(execution_info.output_dict)
    try:
      return self._executor_operator.run_executor(execution_info)
    except Exception as e:  # pylint: disable=broad-except
      outputs_utils.remove_output_dirs(execution_info.output_dict)
      logging.error(
          'Execution failed with error %s '
          'and this is the stack trace \n %s', e, traceback.format_exc())
      raise

  def _publish_successful_execution(
      self, execution_id: int, contexts: List[metadata_store_pb2.Context],
      output_dict: Dict[Text, List[types.Artifact]],
      executor_output: execution_result_pb2.ExecutorOutput) -> None:
    """Publishes succeeded execution result to ml metadata."""
    with self._mlmd_connection as m:
      execution_publish_utils.publish_succeeded_execution(
          metadata_handler=m,
          execution_id=execution_id,
          contexts=contexts,
          output_artifacts=output_dict,
          executor_output=executor_output)

  def _publish_failed_execution(
      self, execution_id: int,
      contexts: List[metadata_store_pb2.Context]) -> None:
    """Publishes failed execution to ml metadata."""
    with self._mlmd_connection as m:
      execution_publish_utils.publish_failed_execution(
          metadata_handler=m, execution_id=execution_id, contexts=contexts)

  def _clean_up(self, execution_info: base_executor_operator.ExecutionInfo):
    tf.io.gfile.rmtree(execution_info.stateful_working_dir)

  def launch(self) -> Optional[metadata_store_pb2.Execution]:
    """Executes the component, includes driver, executor and publisher.

    Returns:
      The metadata of this execution that is registered in MLMD. It can be None
      if the driver decides not to run the execution.
    """
    logging.debug('Running driver for %s', self._pipeline_node)
    prepare_exeucion_result = self._prepare_execution()
    (execution_info, contexts,
     is_execution_needed) = (prepare_exeucion_result.execution_info,
                             prepare_exeucion_result.contexts,
                             prepare_exeucion_result.is_execution_needed)
    if is_execution_needed:
      try:
        executor_output = self._run_executor(execution_info)
      except Exception:  # pylint: disable=broad-except
        self._publish_failed_execution(execution_info.execution_metadata.id,
                                       contexts)
        return execution_info.execution_metadata

      self._clean_up(execution_info)
      logging.info('Publishing output artifacts %s for exeuction %s',
                   execution_info.output_dict,
                   execution_info.execution_metadata.id)
      self._publish_successful_execution(execution_info.execution_metadata.id,
                                         contexts, execution_info.output_dict,
                                         executor_output)
    return execution_info.execution_metadata
