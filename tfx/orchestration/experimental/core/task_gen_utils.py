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
"""Utilities for task generation."""

import itertools
from typing import Dict, Iterable, List, Optional, Sequence

import attr
from tfx import types
from tfx.orchestration import data_types_utils
from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.portable import inputs_utils
from tfx.orchestration.portable import outputs_utils
from tfx.orchestration.portable.mlmd import context_lib
from tfx.orchestration.portable.mlmd import execution_lib
from tfx.proto.orchestration import pipeline_pb2

from ml_metadata.proto import metadata_store_pb2


@attr.s(auto_attribs=True)
class ResolvedInfo:
  contexts: List[metadata_store_pb2.Context]
  exec_properties: Dict[str, types.Property]
  input_artifacts: Optional[Dict[str, List[types.Artifact]]]


def _generate_task_from_execution(metadata_handler: metadata.Metadata,
                                  pipeline: pipeline_pb2.Pipeline,
                                  node: pipeline_pb2.PipelineNode,
                                  execution: metadata_store_pb2.Execution,
                                  is_cancelled: bool = False) -> task_lib.Task:
  """Generates `ExecNodeTask` given execution."""
  contexts = metadata_handler.store.get_contexts_by_execution(execution.id)
  exec_properties = _extract_properties(execution)
  input_artifacts = execution_lib.get_artifacts_dict(
      metadata_handler, execution.id, metadata_store_pb2.Event.INPUT)
  outputs_resolver = outputs_utils.OutputsResolver(node, pipeline.pipeline_info,
                                                   pipeline.runtime_spec,
                                                   pipeline.execution_mode)
  return task_lib.ExecNodeTask(
      node_uid=task_lib.NodeUid.from_pipeline_node(pipeline, node),
      execution=execution,
      contexts=contexts,
      exec_properties=exec_properties,
      input_artifacts=input_artifacts,
      output_artifacts=outputs_resolver.generate_output_artifacts(execution.id),
      executor_output_uri=outputs_resolver.get_executor_output_uri(
          execution.id),
      stateful_working_dir=outputs_resolver.get_stateful_working_directory(
          execution.id),
      pipeline=pipeline,
      is_cancelled=is_cancelled)


def generate_task_from_active_execution(
    metadata_handler: metadata.Metadata,
    pipeline: pipeline_pb2.Pipeline,
    node: pipeline_pb2.PipelineNode,
    executions: Iterable[metadata_store_pb2.Execution],
    is_cancelled: bool = False,
) -> Optional[task_lib.Task]:
  """Generates task from active execution (if any).

  Returns `None` if a task cannot be generated from active execution.

  Args:
    metadata_handler: A handler to access MLMD db.
    pipeline: The pipeline containing the node.
    node: The pipeline node for which to generate a task.
    executions: A sequence of all executions for the given node.
    is_cancelled: Sets `is_cancelled` in ExecNodeTask.

  Returns:
    A `Task` proto if active execution exists for the node. `None` otherwise.

  Raises:
    RuntimeError: If there are multiple active executions for the node.
  """
  active_executions = [
      e for e in executions if execution_lib.is_execution_active(e)
  ]
  if not active_executions:
    return None
  if len(active_executions) > 1:
    raise RuntimeError(
        'Unexpected multiple active executions for the node: {}\n executions: '
        '{}'.format(node.node_info.id, active_executions))
  return _generate_task_from_execution(
      metadata_handler,
      pipeline,
      node,
      active_executions[0],
      is_cancelled=is_cancelled)


def _extract_properties(
    execution: metadata_store_pb2.Execution) -> Dict[str, types.Property]:
  result = {}
  for key, prop in itertools.chain(execution.properties.items(),
                                   execution.custom_properties.items()):
    value = data_types_utils.get_metadata_value(prop)
    if value is None:
      raise ValueError(f'Unexpected property with empty value; key: {key}')
    result[key] = value
  return result


def generate_resolved_info(metadata_handler: metadata.Metadata,
                           node: pipeline_pb2.PipelineNode) -> ResolvedInfo:
  """Returns a `ResolvedInfo` object for executing the node.

  Args:
    metadata_handler: A handler to access MLMD db.
    node: The pipeline node for which to generate.

  Returns:
    A `ResolvedInfo` with input resolutions.
  """
  # Register node contexts.
  contexts = context_lib.prepare_contexts(
      metadata_handler=metadata_handler, node_contexts=node.contexts)

  # Resolve execution properties.
  exec_properties = inputs_utils.resolve_parameters(
      node_parameters=node.parameters)

  # Resolve inputs.
  input_artifacts = inputs_utils.resolve_input_artifacts(
      metadata_handler=metadata_handler, node_inputs=node.inputs)

  return ResolvedInfo(
      contexts=contexts,
      exec_properties=exec_properties,
      input_artifacts=input_artifacts)


def get_executions(
    metadata_handler: metadata.Metadata,
    node: pipeline_pb2.PipelineNode) -> List[metadata_store_pb2.Execution]:
  """Returns all executions for the given pipeline node.

  This finds all executions having the same set of contexts as the pipeline
  node.

  Args:
    metadata_handler: A handler to access MLMD db.
    node: The pipeline node for which to obtain executions.

  Returns:
    List of executions for the given node in MLMD db.
  """
  # Get all the contexts associated with the node.
  contexts = []
  for context_spec in node.contexts.contexts:
    context = metadata_handler.store.get_context_by_type_and_name(
        context_spec.type.name, data_types_utils.get_value(context_spec.name))
    if context is None:
      # If no context is registered, it's certain that there is no
      # associated execution for the node.
      return []
    contexts.append(context)
  return execution_lib.get_executions_associated_with_all_contexts(
      metadata_handler, contexts)


def is_latest_execution_successful(
    executions: Sequence[metadata_store_pb2.Execution]) -> bool:
  """Returns `True` if the latest execution was successful.

  Latest execution will have the most recent `create_time_since_epoch`.

  Args:
    executions: A sequence of executions.

  Returns:
    `True` if latest execution (per `create_time_since_epoch` was successful.
    `False` if `executions` is empty or if latest execution was not successful.
  """
  execution = get_latest_execution(executions)
  return execution_lib.is_execution_successful(
      execution) if execution else False


def get_latest_successful_execution(
    executions: Iterable[metadata_store_pb2.Execution]
) -> Optional[metadata_store_pb2.Execution]:
  """Returns the latest successful execution or `None` if no successful executions exist."""
  successful_executions = [
      e for e in executions if execution_lib.is_execution_successful(e)
  ]
  return get_latest_execution(successful_executions)


def get_latest_execution(
    executions: Iterable[metadata_store_pb2.Execution]
) -> Optional[metadata_store_pb2.Execution]:
  """Returns latest execution or `None` if iterable is empty."""
  sorted_executions = sorted(
      executions, key=lambda e: e.create_time_since_epoch, reverse=True)
  return sorted_executions[0] if sorted_executions else None
