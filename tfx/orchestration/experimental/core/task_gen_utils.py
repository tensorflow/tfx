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
import textwrap
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple, Type
import uuid

from absl import logging
import attr
from tfx import types
from tfx.orchestration import data_types_utils
from tfx.orchestration import metadata
from tfx.orchestration import node_proto_view
from tfx.orchestration.experimental.core import constants
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration import mlmd_connection_manager as mlmd_cm
from tfx.orchestration.portable import inputs_utils
from tfx.orchestration.portable import outputs_utils
from tfx.orchestration.portable.input_resolution import exceptions
from tfx.orchestration.portable.mlmd import context_lib
from tfx.orchestration.portable.mlmd import event_lib
from tfx.orchestration.portable.mlmd import execution_lib
from tfx.orchestration.portable.mlmd import filter_query_builder as q
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import proto_utils
from tfx.utils import status as status_lib
from tfx.utils import typing_utils

from google.protobuf import any_pb2
import ml_metadata as mlmd
from ml_metadata import errors
from ml_metadata.proto import metadata_store_pb2


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
    node: node_proto_view.NodeProtoView,
    skip_errors: Iterable[Type[exceptions.InputResolutionError]] = (),
) -> ResolvedInfo:
  """Returns a `ResolvedInfo` object for executing the node or `None` to skip.

  Args:
    mlmd_connection_manager: MLMDConnectionManager instance for handling
      multiple mlmd db connections.
    node: The pipeline node for which to generate.
    skip_errors: A list of errors to skip on the given error types.

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

  result = ResolvedInfo(
      contexts=contexts,
      input_and_params=[],
  )

  # Resolve execution properties.
  exec_properties = resolve_exec_properties(node)

  # Resolve inputs.
  try:
    resolved_input_artifacts = inputs_utils.resolve_input_artifacts(
        metadata_handler=mlmd_connection_manager, pipeline_node=node
    )
  except exceptions.InputResolutionError as e:
    for skip_error in skip_errors:
      if isinstance(e, skip_error):
        logging.info('[%s] Input resolution skipped: %s', node.node_info.id, e)
        return result
    raise
  if not resolved_input_artifacts:
    return result

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

      result.input_and_params.append(
          InputAndParam(
              input_artifacts=input_artifacts,
              exec_properties=cur_exec_properties,
          )
      )
  return result


def get_executions(
    metadata_handler: metadata.Metadata,
    node: node_proto_view.NodeProtoView,
    only_active: bool = False,
) -> List[metadata_store_pb2.Execution]:
  """Returns all executions for the given pipeline node.

  This finds all executions having the same set of contexts as the pipeline
  node.

  Args:
    metadata_handler: A handler to access MLMD db.
    node: The pipeline node for which to obtain executions.
    only_active: If set to true, only active executions are returned. Otherwise,
      all executions are returned. Active executions mean executions with NEW or
      RUNNING last_known_state.

  Returns:
    List of executions for the given node in MLMD db.
  """
  if not node.contexts.contexts:
    return []
  # Get all the contexts associated with the node.
  filter_query = q.And([])
  for i, context_spec in enumerate(node.contexts.contexts):
    context_type = context_spec.type.name
    context_name = data_types_utils.get_value(context_spec.name)
    filter_query.append(
        q.And([
            f"contexts_{i}.type = '{context_type}'",
            f"contexts_{i}.name = '{context_name}'",
        ])
    )
  if only_active:
    filter_query.append(
        q.Or(['last_known_state = NEW', 'last_known_state = RUNNING'])
    )
  return metadata_handler.store.get_executions(
      list_options=mlmd.ListOptions(
          # TODO(b/274559409): Decide whether to keep explicit order or not.
          # Due to implementation detail, `is_asc = false` (default) with filter
          # query has very bad time complexity, thus enforcing `is_asc = true`
          # here.
          order_by=mlmd.OrderByField.ID,
          is_asc=True,
          filter_query=str(filter_query),
      )
  )


def get_latest_executions_set(
    executions: Iterable[metadata_store_pb2.Execution],
) -> List[metadata_store_pb2.Execution]:  # pylint: disable=g-doc-args
  """Returns latest set of executions, ascendingly ordered by __external_execution_index__.

  Use the following executions as an example:

      Execution(id=0, __external_execution_index__=0, state=FAILED,
        create_time_since_epoch=100)
      Execution(id=1, __external_execution_index__=1, state=NEW,
        create_time_since_epoch=150)
      Execution(id=2, __external_execution_index__=0, state=FAILED,
        create_time_since_epoch=200)
      Execution(id=3, __external_execution_index__=0, state=FAILED,
        create_time_since_epoch=250)

  This function returns the latest execution of each
  __external_execution_index__, which in this case will be:
      Execution(id=3, __external_execution_index__=0, state=FAILED,
        create_time_since_epoch=250)
      Execution(id=1, __external_execution_index__=1, state=NEW,
        create_time_since_epoch=150)

  """
  # Sorted by create_time_since_epoch.
  sorted_executions = execution_lib.sort_executions_newest_to_oldest(executions)
  if not sorted_executions:
    return []

  sorted_execution_by_idx_map = collections.defaultdict(list)
  for e in sorted_executions:
    sorted_execution_by_idx_map[e.custom_properties[
        _EXTERNAL_EXECUTION_INDEX].int_value].append(e)

  latest_execution_set = []
  for idx in sorted(sorted_execution_by_idx_map.keys()):
    latest_execution_set.append(sorted_execution_by_idx_map[idx][0])

  return latest_execution_set


def get_num_of_failures_from_failed_execution(
    executions: Iterable[metadata_store_pb2.Execution],
    failed_execution: metadata_store_pb2.Execution) -> int:
  """Returns the num of failed executions.

  Only the executions that have the same external execution index as the failed
  execution will be counted.

  Args:
    executions: An iterable of executions.
    failed_execution: A failed execution whose execution index will be tested
    against to count the total number of failed execution.
  """
  target_index = failed_execution.custom_properties[
      _EXTERNAL_EXECUTION_INDEX
  ].int_value

  failed_executions = [
      e
      for e in executions
      if (
          e.last_known_state == metadata_store_pb2.Execution.FAILED
          and e.custom_properties[_EXTERNAL_EXECUTION_INDEX].int_value
          == target_index
      )
  ]
  return len(failed_executions)


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


def register_executions_from_existing_executions(
    metadata_handle: metadata.Metadata,
    node: node_proto_view.NodeProtoView,
    existing_executions: List[metadata_store_pb2.Execution],
) -> Sequence[metadata_store_pb2.Execution]:
  """Registers a list of new executions from a list of failed/canceled executions."""
  if not existing_executions:
    return []

  exec_properties = resolve_exec_properties(node)
  new_executions = []
  input_artifacts = []
  for existing_execution in existing_executions:
    # TODO(b/224800273): We also need to resolve and set dynamic execution
    # properties.
    new_execution = execution_lib.prepare_execution(
        metadata_handler=metadata_handle,
        execution_type=node.node_info.type,
        state=metadata_store_pb2.Execution.NEW,
        exec_properties=exec_properties,
        execution_name=str(uuid.uuid4()),
    )
    # Only copy necessary custom_properties from the failed/canceled execution.
    # LINT.IfChange(new_execution_custom_properties)
    new_execution.custom_properties[_EXTERNAL_EXECUTION_INDEX].CopyFrom(
        existing_execution.custom_properties[_EXTERNAL_EXECUTION_INDEX]
    )
    # LINT.ThenChange(:execution_custom_properties)
    new_executions.append(new_execution)
    input_artifacts.append(
        execution_lib.get_input_artifacts(
            metadata_handle, existing_execution.id
        )
    )

  contexts = metadata_handle.store.get_contexts_by_execution(
      existing_executions[0].id
  )
  return execution_lib.put_executions(
      metadata_handle,
      new_executions,
      contexts,
      input_artifacts_maps=input_artifacts,
  )


def register_executions(
    metadata_handler: metadata.Metadata,
    execution_type: metadata_store_pb2.ExecutionType,
    contexts: Sequence[metadata_store_pb2.Context],
    input_and_params: List[InputAndParam],
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
    execution.custom_properties[_EXTERNAL_EXECUTION_INDEX].int_value = index
    executions.append(execution)
  # LINT.ThenChange(:new_execution_custom_properties)

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


def get_unprocessed_inputs(
    metadata_handle: metadata.Metadata,
    executions: Sequence[metadata_store_pb2.Execution],
    resolved_info: ResolvedInfo,
    node: node_proto_view.NodeProtoView,
) -> List[InputAndParam]:
  """Get a list of unprocessed input from resolved_info.

  Args:
    metadata_handle: A handle to access local MLMD db.
    executions: A list of executions
    resolved_info: Resolved input of a node. It may contain processed and
      unprocessed input.
    node: The pipeline node of the input.

  Returns:
    A list of InputAndParam that have not been processed.
  """
  # Finds out the keys that should be ignored.
  input_triggers = node.execution_options.async_trigger.input_triggers
  ignore_keys = set(
      [key for key, trigger in input_triggers.items() if trigger.no_trigger]
  )

  # Gets the processed inputs.
  processed_inputs: List[Dict[str, Tuple[int, ...]]] = []
  events = metadata_handle.store.get_events_by_execution_ids(
      [e.id for e in executions]
  )
  for execution in executions:
    input_events = [
        e
        for e in events
        if e.type == metadata_store_pb2.Event.INPUT
        and event_lib.is_valid_input_event(e)
        and e.execution_id == execution.id
    ]
    ids_by_key = event_lib.reconstruct_artifact_id_multimap(input_events)
    # Filters out the keys starting with '_' and the keys should be ingored.
    ids_by_key = {
        k: tuple(sorted(v))
        for k, v in ids_by_key.items()
        if not k.startswith('_') and k not in ignore_keys
    }
    processed_inputs.append(ids_by_key)

  # Some input artifacts are from external pipelines, so we need to find out the
  # external_id to id mapping in the local db.
  local_id_by_external_id: Dict[str, int] = {}
  for input_and_param in resolved_info.input_and_params:
    for artifact in itertools.chain(*input_and_param.input_artifacts.values()):
      if artifact.mlmd_artifact.external_id:
        local_id_by_external_id[artifact.mlmd_artifact.external_id] = -1
  if local_id_by_external_id:
    try:
      for artifact in metadata_handle.store.get_artifacts_by_external_ids(
          external_ids=local_id_by_external_id
      ):
        local_id_by_external_id[artifact.external_id] = artifact.id
    except errors.NotFoundError:
      # If all the external ids do not exist in local db, we get NotFoundError.
      # It is safe to pass, and we will handle them in the following code.
      pass
    except Exception as e:  # pylint:disable=broad-except
      logging.exception('Error when getting artifacts by external ids: %s', e)
      return []

  # Finds out the unprocessed inputs.
  unprocessed_inputs = []
  for input_and_param in resolved_info.input_and_params:
    resolved_input_ids_by_key = collections.defaultdict(list)
    for key, artifacts in input_and_param.input_artifacts.items():
      for a in artifacts:
        if a.id:
          resolved_input_ids_by_key[key].append(a.id)
        elif a.mlmd_artifact.external_id:
          resolved_input_ids_by_key[key].append(
              local_id_by_external_id[a.mlmd_artifact.external_id]
          )
      resolved_input_ids_by_key[key] = tuple(resolved_input_ids_by_key[key])

    # Filters out the keys starting with '_' and the keys should be ingored.
    resolved_input_ids_by_key = {
        k: tuple(sorted(v))
        for k, v in resolved_input_ids_by_key.items()
        if not k.startswith('_') and k not in ignore_keys
    }

    for processed in processed_inputs:
      if processed == resolved_input_ids_by_key:
        break
    else:
      unprocessed_inputs.append(input_and_param)

  return unprocessed_inputs


def interpret_status_from_failed_execution(
    execution: metadata_store_pb2.Execution,
) -> status_lib.Status:
  """Interprets `Status` from given failed execution.

  Args:
    execution: An execution with last_known_state=FAILED.

  Returns:
    A `Status` object interpreted from the execution state.

  Raises:
    ValueError: If the given execution has `last_known_state` other than
    `FAILED`.
  """
  if not execution_lib.is_execution_failed(execution):
    raise ValueError(
        'Must be called with an execution with last_known_state = FAILED.'
    )
  # If execution result is available, that will have the most proximate cause
  # for the failed execution.
  execution_result = execution_lib.get_execution_result(
      execution, ignore_parse_errors=True
  )
  if execution_result is not None:
    # We expect the error code to be non-OK but if by any chance it is OK,
    # we account it as UNKNOWN.
    error_code = execution_result.code or status_lib.Code.UNKNOWN
    error_msg = execution_result.result_message or None
  else:
    error_code_value = execution.custom_properties.get(
        constants.EXECUTION_ERROR_CODE_KEY
    )
    if error_code_value is not None:
      # If error code is set, we expect it to be non-OK. By any chance if it is
      # OK, we account it as UNKNOWN.
      error_code = (
          data_types_utils.get_metadata_value(error_code_value)
          or status_lib.Code.UNKNOWN
      )
    else:
      error_code = status_lib.Code.UNKNOWN
    error_msg_value = execution.custom_properties.get(
        constants.EXECUTION_ERROR_MSG_KEY
    )
    error_msg = (
        data_types_utils.get_metadata_value(error_msg_value)
        if error_msg_value is not None
        else None
    )
  error_msg = textwrap.shorten(error_msg, width=512) if error_msg else None
  return status_lib.Status(code=error_code, message=error_msg)
