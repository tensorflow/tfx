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

from absl import logging
import attr
from tfx import types
from tfx.orchestration import data_types_utils
from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.portable import inputs_utils
from tfx.orchestration.portable import outputs_utils
from tfx.orchestration.portable.input_resolution import exceptions
from tfx.orchestration.portable.mlmd import context_lib
from tfx.orchestration.portable.mlmd import execution_lib
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import proto_utils
from tfx.utils import typing_utils

import ml_metadata as mlmd
from ml_metadata.proto import metadata_store_pb2
from google.protobuf import any_pb2


@attr.s(auto_attribs=True)
class ResolvedInfo:
  contexts: List[metadata_store_pb2.Context]
  exec_properties: Dict[str, types.ExecPropertyTypes]
  input_artifacts: List[Optional[typing_utils.ArtifactMultiMap]]


def _generate_task_from_execution(metadata_handler: metadata.Metadata,
                                  pipeline: pipeline_pb2.Pipeline,
                                  node: pipeline_pb2.PipelineNode,
                                  execution: metadata_store_pb2.Execution,
                                  is_cancelled: bool = False) -> task_lib.Task:
  """Generates `ExecNodeTask` given execution."""
  contexts = metadata_handler.store.get_contexts_by_execution(execution.id)
  exec_properties = extract_properties(execution)
  input_artifacts = execution_lib.get_artifacts_dict(
      metadata_handler, execution.id, [metadata_store_pb2.Event.INPUT])
  outputs_resolver = outputs_utils.OutputsResolver(node, pipeline.pipeline_info,
                                                   pipeline.runtime_spec,
                                                   pipeline.execution_mode)
  output_artifacts = outputs_resolver.generate_output_artifacts(execution.id)
  outputs_utils.make_output_dirs(output_artifacts)
  return task_lib.ExecNodeTask(
      node_uid=task_lib.NodeUid.from_pipeline_node(pipeline, node),
      execution_id=execution.id,
      contexts=contexts,
      exec_properties=exec_properties,
      input_artifacts=input_artifacts,
      output_artifacts=output_artifacts,
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
) -> List[task_lib.Task]:
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
  tasks = []
  for active_execution in active_executions:
    tasks.append(
        _generate_task_from_execution(
            metadata_handler,
            pipeline,
            node,
            active_execution,
            is_cancelled=is_cancelled))
  return tasks


def extract_properties(
    execution: metadata_store_pb2.Execution
) -> Dict[str, types.ExecPropertyTypes]:
  """Extracts execution properties from mlmd Execution."""
  result = {}
  for key, prop in itertools.chain(execution.properties.items(),
                                   execution.custom_properties.items()):
    if execution_lib.is_schema_key(key):
      continue

    schema_key = execution_lib.get_schema_key(key)
    schema = None
    if schema_key in execution.custom_properties:
      schema = proto_utils.json_to_proto(
          data_types_utils.get_metadata_value(
              execution.custom_properties[schema_key]),
          pipeline_pb2.Value.Schema())
    value = data_types_utils.get_parsed_value(prop, schema)

    if value is None:
      raise ValueError(f'Unexpected property with empty value; key: {key}')
    result[key] = value
  return result


def generate_resolved_info(
    metadata_handler: metadata.Metadata,
    node: pipeline_pb2.PipelineNode) -> Optional[ResolvedInfo]:
  """Returns a `ResolvedInfo` object for executing the node or `None` to skip.

  Args:
    metadata_handler: A handler to access MLMD db.
    node: The pipeline node for which to generate.

  Returns:
    A `ResolvedInfo` with input resolutions or `None` if execution should be
    skipped.

  Raises:
    NotImplementedError: Multiple dicts returned by inputs_utils
      resolve_input_artifacts_v2, which is currently not supported.
  """
  # Register node contexts.
  contexts = context_lib.prepare_contexts(
      metadata_handler=metadata_handler, node_contexts=node.contexts)

  # Resolve execution properties.
  exec_properties = data_types_utils.build_parsed_value_dict(
      inputs_utils.resolve_parameters_with_schema(
          node_parameters=node.parameters))

  # Resolve inputs.
  try:
    resolved_input_artifacts = inputs_utils.resolve_input_artifacts_v2(
        metadata_handler=metadata_handler, pipeline_node=node)
  except exceptions.InputResolutionError as e:
    logging.warning('Input resolution error raised for node: %s; error: %s',
                    node.node_info.id, e)
    resolved_input_artifacts = None
  else:
    if isinstance(resolved_input_artifacts, inputs_utils.Skip):
      return None
    assert isinstance(resolved_input_artifacts, inputs_utils.Trigger)
    assert resolved_input_artifacts
    # TODO(b/197741942): Support multiple dicts.
    if len(resolved_input_artifacts) > 1:
      raise NotImplementedError(
          'Handling more than one input dicts not implemented.')

  return ResolvedInfo(
      contexts=contexts,
      exec_properties=exec_properties,
      input_artifacts=resolved_input_artifacts)


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
  if not node.contexts.contexts:
    return []
  # Get all the contexts associated with the node.
  contexts = []
  for i, context_spec in enumerate(node.contexts.contexts):
    context_type = context_spec.type.name
    context_name = data_types_utils.get_value(context_spec.name)
    contexts.append(
        f"(contexts_{i}.type = '{context_type}' AND contexts_{i}.name = '{context_name}')"
    )
  filter_query = ' AND '.join(contexts)
  return metadata_handler.store.get_executions(
      list_options=mlmd.ListOptions(filter_query=filter_query))


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
  sorted_executions = execution_lib.sort_executions_newest_to_oldest(executions)
  return sorted_executions[0] if sorted_executions else None


# TODO(b/182944474): Raise error in _get_executor_spec if executor spec is
# missing for a non-system node.
def get_executor_spec(pipeline: pipeline_pb2.Pipeline,
                      node_id: str) -> Optional[any_pb2.Any]:
  """Returns executor spec for given node_id if it exists in pipeline IR, None otherwise."""
  if not pipeline.deployment_config.Is(
      pipeline_pb2.IntermediateDeploymentConfig.DESCRIPTOR):
    return None
  depl_config = pipeline_pb2.IntermediateDeploymentConfig()
  pipeline.deployment_config.Unpack(depl_config)
  return depl_config.executor_specs.get(node_id)
