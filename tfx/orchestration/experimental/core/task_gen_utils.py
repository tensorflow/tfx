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

from typing import Dict, Iterable, List, Optional, Sequence, Text

import attr
from tfx import types
from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.portable import inputs_utils
from tfx.orchestration.portable.mlmd import common_utils
from tfx.orchestration.portable.mlmd import context_lib
from tfx.orchestration.portable.mlmd import execution_lib
from tfx.proto.orchestration import pipeline_pb2

from ml_metadata.proto import metadata_store_pb2


@attr.s
class ResolvedInfo:
  contexts = attr.ib(type=List[metadata_store_pb2.Context])
  exec_properties = attr.ib(Dict[Text, types.Property])
  input_artifacts = attr.ib(Optional[Dict[Text, List[types.Artifact]]])


def generate_task_from_active_execution(
    pipeline: pipeline_pb2.Pipeline, node: pipeline_pb2.PipelineNode,
    executions: Iterable[metadata_store_pb2.Execution]
) -> Optional[task_lib.Task]:
  """Generates task from active execution (if any).

  Returns `None` if a task cannot be generated from active execution.

  Args:
    pipeline: The pipeline containing the node.
    node: The pipeline node for which to generate a task.
    executions: A sequence of all executions for the given node.

  Returns:
    A `Task` proto if active execution exists for the node. `None` otherwise.

  Raises:
    RuntimeError: If there are multiple active executions for the node.
  """
  active_executions = []
  for execution in executions:
    if execution.last_known_state not in (metadata_store_pb2.Execution.NEW,
                                          metadata_store_pb2.Execution.RUNNING):
      continue
    active_executions.append(execution)

  if active_executions:
    if len(active_executions) > 1:
      raise RuntimeError(
          'Unexpected multiple active executions for the node: {}\n'
          'executions: {}'.format(node.node_info.id, active_executions))
    return task_lib.ExecNodeTask.create(pipeline, node, active_executions[0].id)
  return None


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
  contexts = context_lib.register_contexts_if_not_exists(
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
        context_spec.type.name, common_utils.get_value(context_spec.name))
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
  sorted_executions = sorted(
      executions, key=lambda e: e.create_time_since_epoch, reverse=True)
  return (execution_lib.is_execution_successful(sorted_executions[0])
          if sorted_executions else False)


def get_latest_successful_execution(
    executions: Iterable[metadata_store_pb2.Execution]
) -> Optional[metadata_store_pb2.Execution]:
  """Returns the latest successful execution or `None` if no successful executions exist."""
  successful_executions = [
      e for e in executions if execution_lib.is_execution_successful(e)
  ]
  if successful_executions:
    return sorted(
        successful_executions,
        key=lambda e: e.create_time_since_epoch,
        reverse=True)[0]
  return None


def is_feasible_node(node: pipeline_pb2.PipelineNode) -> bool:
  """Returns whether the node is feasible for task generation.

  Currently, ExampleGen is ignored for task generation as the component is
  expected to contain the entire driver + executor + publish lifecycle which is
  managed outside the scope of the task generator.

  TODO(goutham): This is a short-term heuristic and may need revision when a
  standard approach to detect such nodes is instituted.

  Args:
    node: A pipeline node.

  Returns:
    `True` if the node is feasible, `False` otherwise.
  """
  return node.node_info.type.name != 'ExampleGen'
