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

import collections
import itertools
import time
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence
import uuid

from absl import logging
import attr
from tfx import types
from tfx.orchestration import data_types_utils
from tfx.orchestration import metadata
from tfx.orchestration import node_proto_view
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration import mlmd_connection_manager as mlmd_cm
from tfx.orchestration.portable import inputs_utils
from tfx.orchestration.portable import outputs_utils
from tfx.orchestration.portable.input_resolution import exceptions
from tfx.orchestration.portable.mlmd import context_lib
from tfx.orchestration.portable.mlmd import execution_lib
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import proto_utils
from tfx.utils import typing_utils

from google.protobuf import any_pb2
import ml_metadata as mlmd
from ml_metadata import errors
from ml_metadata.proto import metadata_store_pb2

_EXECUTION_SET_SIZE = '__execution_set_size__'
_EXECUTION_TIMESTAMP = '__execution_timestamp__'
_EXTERNAL_EXECUTION_INDEX = '__external_execution_index__'


@attr.s(auto_attribs=True)
class InputAndParam:
  input_artifacts: Optional[typing_utils.ArtifactMultiMap] = None
  exec_properties: Optional[MutableMapping[str, types.ExecPropertyTypes]] = None


@attr.s(auto_attribs=True)
class ResolvedInfo:
  contexts: List[metadata_store_pb2.Context]
  input_and_params: List[InputAndParam]


def generate_task_from_execution(
    metadata_handler: metadata.Metadata,
    pipeline: pipeline_pb2.Pipeline,
    node: node_proto_view.NodeProtoView,
    execution: metadata_store_pb2.Execution,
    cancel_type: Optional[task_lib.NodeCancelType] = None) -> task_lib.Task:
  """Generates `ExecNodeTask` given execution."""
  if not execution_lib.is_execution_active(execution):
    raise RuntimeError(f'Execution is not active: {execution}.')

  contexts = metadata_handler.store.get_contexts_by_execution(execution.id)
  exec_properties = extract_properties(execution)
  input_artifacts = execution_lib.get_input_artifacts(
      metadata_handler, execution.id)
  outputs_resolver = outputs_utils.OutputsResolver(node, pipeline.pipeline_info,
                                                   pipeline.runtime_spec,
                                                   pipeline.execution_mode)
  output_artifacts = outputs_resolver.generate_output_artifacts(execution.id)
  outputs_utils.make_output_dirs(output_artifacts)
  return task_lib.ExecNodeTask(
      node_uid=task_lib.NodeUid.from_node(pipeline, node),
      execution_id=execution.id,
      contexts=contexts,
      exec_properties=exec_properties,
      input_artifacts=input_artifacts,
      output_artifacts=output_artifacts,
      executor_output_uri=outputs_resolver.get_executor_output_uri(
          execution.id),
      stateful_working_dir=outputs_resolver.get_stateful_working_directory(
          execution.id),
      tmp_dir=outputs_resolver.make_tmp_dir(execution.id),
      pipeline=pipeline,
      cancel_type=cancel_type)


def generate_cancel_task_from_running_execution(
    metadata_handler: metadata.Metadata,
    pipeline: pipeline_pb2.Pipeline,
    node: node_proto_view.NodeProtoView,
    executions: Iterable[metadata_store_pb2.Execution],
    cancel_type: task_lib.NodeCancelType,
) -> Optional[task_lib.Task]:
  """Generates cancellation ExecNodeTask from running execution (if any).

  Returns `None` if a task cannot be generated from running execution.

  Args:
    metadata_handler: A handler to access MLMD db.
    pipeline: The pipeline containing the node.
    node: The pipeline node for which to generate a task.
    executions: A sequence of all executions for the given node.
    cancel_type: Sets `cancel_type` in ExecNodeTask.

  Returns:
    An `ExecNodeTask` if running execution exists for the node. `None`
    otherwise.

  Raises:
    RuntimeError: If there are multiple running executions for the node.
  """
  running_executions = [
      e for e in executions if execution_lib.is_execution_running(e)
  ]
  if not running_executions:
    return None
  if len(running_executions) > 1:
    raise RuntimeError(
        'A node can have only one running execution, but get multiple running '
        f'executions for node {node.node_info.id}')
  return generate_task_from_execution(
      metadata_handler,
      pipeline,
      node,
      running_executions[0],
      cancel_type=cancel_type)


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


def resolve_exec_properties(
    node: node_proto_view.NodeProtoView) -> Dict[str, types.ExecPropertyTypes]:
  """Resolves execution properties for executing the node."""
  return data_types_utils.build_parsed_value_dict(
      inputs_utils.resolve_parameters_with_schema(
          node_parameters=node.parameters))


def generate_resolved_info(
    mlmd_connection_manager: mlmd_cm.MLMDConnectionManager,
    node: node_proto_view.NodeProtoView) -> ResolvedInfo:
  """Returns a `ResolvedInfo` object for executing the node or `None` to skip.

  Args:
    mlmd_connection_manager: MLMDConnectionManager instance for handling
      multiple mlmd db connections.
    node: The pipeline node for which to generate.

  Returns:
    A `ResolvedInfo` with input resolutions. If execution should be skipped,
    ResolvedInfo has empty input_and_params.

  Raises:
    InputResolutionError: If there are some errors when we try to resolve input.
  """
  # Register node contexts.
  contexts = context_lib.prepare_contexts(
      metadata_handler=mlmd_connection_manager.primary_mlmd_handle,
      node_contexts=node.contexts)

  # Resolve execution properties.
  exec_properties = resolve_exec_properties(node)

  # Define Input and Param.
  input_and_params = []

  # Resolve inputs.
  try:
    resolved_input_artifacts = inputs_utils.resolve_input_artifacts(
        metadata_handler=mlmd_connection_manager, pipeline_node=node)
  except exceptions.InputResolutionError as e:
    logging.exception('[%s] Input resolution error: %s', node.node_info.id, e)
    raise
  else:
    if isinstance(resolved_input_artifacts, inputs_utils.Skip):
      return ResolvedInfo(contexts=contexts, input_and_params=[])
    assert isinstance(resolved_input_artifacts, inputs_utils.Trigger)
    assert resolved_input_artifacts

  if resolved_input_artifacts:
    for input_artifacts in resolved_input_artifacts:
      try:
        dynamic_exec_properties = inputs_utils.resolve_dynamic_parameters(
            node_parameters=node.parameters, input_artifacts=input_artifacts)
      except exceptions.InputResolutionError as e:
        logging.exception('[%s] Parameter resolution error: %s',
                          node.node_info.id, e)
        raise

      if not dynamic_exec_properties:
        cur_exec_properties = exec_properties
      else:
        cur_exec_properties = {**exec_properties, **dynamic_exec_properties}

      input_and_params.append(
          InputAndParam(
              input_artifacts=input_artifacts,
              exec_properties=cur_exec_properties))
  return ResolvedInfo(contexts=contexts, input_and_params=input_and_params)


def get_executions(
    metadata_handler: metadata.Metadata,
    node: node_proto_view.NodeProtoView) -> List[metadata_store_pb2.Execution]:
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


def get_latest_executions_set(
    executions: Iterable[metadata_store_pb2.Execution]
) -> List[metadata_store_pb2.Execution]:  # pylint: disable=g-doc-args
  """Returns latest set of executions, ascendingly ordered by __external_execution_index__.

  When _EXECUTION_SET_SIZE > 1 and there are retry executions, e.g., consider
  the following executions with `__execution_set_size__ == 2`, which have the
  same `__execution_timestamp__` but different `create_time_since_epoch`,

      Execution(id=0, __external_execution_index__=0, state=FAILED,
      __execution_timestamp__=1234, create_time_since_epoch=100)
      Execution(id=1, __external_execution_index__=1, state=NEW,
      __execution_timestamp__=1234, create_time_since_epoch=150)
      Execution(id=2, __external_execution_index__=0, state=FAILED,
      __execution_timestamp__=1234, create_time_since_epoch=200)
      Execution(id=3, __external_execution_index__=0, state=FAILED,
      __execution_timestamp__=1234, create_time_since_epoch=250)

  This function returns the latest execution of each
  __external_execution_index__, which in this case will be:
      Execution(id=3, __external_execution_index__=0, state=FAILED,
      __execution_timestamp__=1234, create_time_since_epoch=250)
      Execution(id=1, __external_execution_index__=1, state=NEW,
      __execution_timestamp__=1234, create_time_since_epoch=150)

  Raises:
    RuntimeError: if the size of latest execution set is not
    __execution_set_size__.
  """
  # Sorted by create_time_since_epoch.
  sorted_executions = execution_lib.sort_executions_newest_to_oldest(executions)
  if not sorted_executions:
    return []

  size = sorted_executions[0].custom_properties.get(_EXECUTION_SET_SIZE)
  if not size:
    return [sorted_executions[0]]

  timestamp = sorted_executions[0].custom_properties.get(
      _EXECUTION_TIMESTAMP).int_value
  sorted_execution_by_idx_map = collections.defaultdict(list)
  for e in sorted_executions:
    sorted_execution_by_idx_map[e.custom_properties[
        _EXTERNAL_EXECUTION_INDEX].int_value].append(e)
  latest_execution_set = []
  for idx in sorted(sorted_execution_by_idx_map.keys()):
    # Only add executions with the same latest timestamp.
    if sorted_execution_by_idx_map[idx][0].custom_properties[
        _EXECUTION_TIMESTAMP].int_value == timestamp:
      latest_execution_set.append(sorted_execution_by_idx_map[idx][0])
  if len(latest_execution_set) != size.int_value:
    raise RuntimeError('Expected the `latest_execution_set` to have exactly '
                       f'{size.int_value} executions, got '
                       f'{len(latest_execution_set)} instead')
  return latest_execution_set


def get_num_of_failures_from_failed_execution(
    executions: Iterable[metadata_store_pb2.Execution],
    failed_execution: metadata_store_pb2.Execution) -> int:
  """Returns the num of failed executions.

  Only the executions that have the same timestamp and external execution
  index as the failed execution will be counted.

  Args:
    executions: An iterable of executions.
    failed_execution: A failed execution whose timestamp and external execution
    index will be tested against to count the total number of failed execution.
  """
  target_timestamp = failed_execution.custom_properties[
      _EXECUTION_TIMESTAMP].int_value
  target_external_execution_index = failed_execution.custom_properties[
      _EXTERNAL_EXECUTION_INDEX].int_value
  # pylint: disable=g-complex-comprehension
  return len([
      e for e in executions
      if (e.last_known_state == metadata_store_pb2.Execution.FAILED and
          e.custom_properties[_EXECUTION_TIMESTAMP].int_value ==
          target_timestamp and e.custom_properties[_EXTERNAL_EXECUTION_INDEX]
          .int_value == target_external_execution_index)
  ])


def get_oldest_active_execution(
    executions: Iterable[metadata_store_pb2.Execution]
) -> Optional[metadata_store_pb2.Execution]:
  """Returns the oldest active execution or `None` if no active executions exist.

  Args:
    executions: A list of executions

  Returns:
    Execution if the oldest active execution exist or `None` if not exist.
  """
  active_executions = [
      e for e in executions if execution_lib.is_execution_active(e)
  ]
  if not active_executions:
    return None

  sorted_executions = execution_lib.sort_executions_newest_to_oldest(
      active_executions)
  return sorted_executions[-1] if sorted_executions else None


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


def register_retry_execution(
    metadata_handle: metadata.Metadata,
    node: node_proto_view.NodeProtoView,
    failed_execution: metadata_store_pb2.Execution
) -> metadata_store_pb2.Execution:
  """Generates a retry execution from a failed execution and put it in MLMD."""
  # Set a new execution name and put the state to RUNNING.
  exec_properties = resolve_exec_properties(node)
  # TODO(b/224800273): We also need to resolve and set dynamic execution
  # properties.
  retry_execution = execution_lib.prepare_execution(
      metadata_handler=metadata_handle,
      execution_type=node.node_info.type,
      state=metadata_store_pb2.Execution.RUNNING,
      exec_properties=exec_properties,
      execution_name=str(uuid.uuid4()))
  # Only copy necessary custom_properties from the failed execution.
  # LINT.IfChange(retry_execution_custom_properties)
  retry_execution.custom_properties[_EXECUTION_SET_SIZE].CopyFrom(
      failed_execution.custom_properties[_EXECUTION_SET_SIZE])
  retry_execution.custom_properties[_EXECUTION_TIMESTAMP].CopyFrom(
      failed_execution.custom_properties[_EXECUTION_TIMESTAMP])
  retry_execution.custom_properties[_EXTERNAL_EXECUTION_INDEX].CopyFrom(
      failed_execution.custom_properties[
          _EXTERNAL_EXECUTION_INDEX])
  # LINT.ThenChange(:execution_custom_properties)

  contexts = metadata_handle.store.get_contexts_by_execution(
      failed_execution.id)
  input_artifacts = execution_lib.get_input_artifacts(
      metadata_handle, failed_execution.id)
  return execution_lib.put_execution(
      metadata_handle,
      retry_execution,
      contexts,
      input_artifacts=input_artifacts)


def register_executions(
    metadata_handler: metadata.Metadata,
    execution_type: metadata_store_pb2.ExecutionType,
    contexts: Sequence[metadata_store_pb2.Context],
    input_and_params: List[InputAndParam]
) -> Sequence[metadata_store_pb2.Execution]:
  """Registers multiple executions in MLMD.

  Along with the execution:
  -  the input artifacts will be linked to the executions.
  -  the contexts will be linked to both the executions and its input artifacts.

  Args:
    metadata_handler: A handler to access MLMD.
    execution_type: The type of the execution.
    contexts: MLMD contexts to associate with the executions.
    input_and_params: A list of InputAndParams, which includes input_dicts
    (dictionaries of artifacts. One execution will be registered for each of the
    input_dict) and corresponding exec_properties.

  Returns:
    A list of MLMD executions that are registered in MLMD, with id populated.
      All registered executions have a state of NEW.
  """
  timestamp = int(time.time() * 1e6)
  executions = []
  for index, input_and_param in enumerate(input_and_params):
    # Prepare executions.
    execution = execution_lib.prepare_execution(
        metadata_handler,
        execution_type,
        metadata_store_pb2.Execution.NEW,
        input_and_param.exec_properties,
        execution_name=str(uuid.uuid4()))
    # LINT.IfChange(execution_custom_properties)
    execution.custom_properties[_EXECUTION_SET_SIZE].int_value = len(
        input_and_params)
    execution.custom_properties[_EXECUTION_TIMESTAMP].int_value = timestamp
    execution.custom_properties[_EXTERNAL_EXECUTION_INDEX].int_value = index
    executions.append(execution)
  # LINT.ThenChange(:retry_execution_custom_properties)

  if len(executions) == 1:
    return [
        execution_lib.put_execution(
            metadata_handler,
            executions[0],
            contexts,
            input_artifacts=input_and_params[0].input_artifacts)
    ]

  return execution_lib.put_executions(
      metadata_handler, executions, contexts,
      [input_and_param.input_artifacts for input_and_param in input_and_params])


def update_external_artifact_type(local_mlmd_handle: metadata.Metadata,
                                  artifacts: Sequence[types.artifact.Artifact]):
  """Copies artifact types of external artifacts to local db.

  Args:
    local_mlmd_handle: A handle to access local MLMD db.
    artifacts: A list of artifacts.
  """
  local_type_id_by_name = {}
  for artifact in artifacts:
    if not artifact.artifact_type.HasField('id'):
      type_name = artifact.type_name
      if type_name not in local_type_id_by_name:
        try:
          local_artifact_type = local_mlmd_handle.store.get_artifact_type(
              type_name=type_name)
          local_type_id_by_name[type_name] = local_artifact_type.id
        except errors.NotFoundError:
          external_artifact_type = artifact.artifact_type
          new_type_id = local_mlmd_handle.store.put_artifact_type(
              external_artifact_type)
          local_type_id_by_name[type_name] = new_type_id

      local_artifact_type_id = local_type_id_by_name[type_name]
      artifact.type_id = local_artifact_type_id
      artifact.artifact_type.id = local_artifact_type_id
