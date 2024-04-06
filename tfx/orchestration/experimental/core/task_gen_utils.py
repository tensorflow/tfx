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
import json
import sys
import textwrap
from typing import Callable, Dict, Iterable, List, MutableMapping, Optional, Sequence, Type
import uuid

from absl import logging
import attr
from tfx import types
from tfx.dsl.compiler import constants as context_constants
from tfx.dsl.compiler import placeholder_utils
from tfx.orchestration import data_types_utils
from tfx.orchestration import metadata
from tfx.orchestration import node_proto_view
from tfx.orchestration.experimental.core import constants
from tfx.orchestration.experimental.core import mlmd_state
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration import mlmd_connection_manager as mlmd_cm
from tfx.orchestration.portable import data_types
from tfx.orchestration.portable import inputs_utils
from tfx.orchestration.portable import outputs_utils
from tfx.orchestration.portable.input_resolution import exceptions
from tfx.orchestration.portable.mlmd import common_utils
from tfx.orchestration.portable.mlmd import context_lib
from tfx.orchestration.portable.mlmd import event_lib
from tfx.orchestration.portable.mlmd import execution_lib
from tfx.orchestration.portable.mlmd import filter_query_builder as q
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import proto_utils
from tfx.utils import status as status_lib
from tfx.utils import typing_utils

from tfx.orchestration.experimental.core import deployment_config_utils
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
    metadata_handle: metadata.Metadata,
    pipeline: pipeline_pb2.Pipeline,
    node: node_proto_view.NodeProtoView,
    execution: metadata_store_pb2.Execution,
    cancel_type: Optional[task_lib.NodeCancelType] = None,
) -> task_lib.Task:
  """Generates `ExecNodeTask` given execution."""
  if not execution_lib.is_execution_active(execution):
    raise RuntimeError(f'Execution is not active: {execution}.')

  contexts = metadata_handle.store.get_contexts_by_execution(execution.id)
  exec_properties = extract_properties(execution)
  input_artifacts = execution_lib.get_input_artifacts(
      metadata_handle, execution.id
  )
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
          execution),
      tmp_dir=outputs_resolver.make_tmp_dir(execution.id),
      pipeline=pipeline,
      cancel_type=cancel_type)


def generate_cancel_task_from_running_execution(
    metadata_handle: metadata.Metadata,
    pipeline: pipeline_pb2.Pipeline,
    node: node_proto_view.NodeProtoView,
    executions: Iterable[metadata_store_pb2.Execution],
    cancel_type: task_lib.NodeCancelType,
) -> Optional[task_lib.Task]:
  """Generates cancellation ExecNodeTask from running execution (if any).

  Returns `None` if a task cannot be generated from running execution.

  Args:
    metadata_handle: A handler to access MLMD db.
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
      metadata_handle,
      pipeline,
      node,
      running_executions[0],
      cancel_type=cancel_type,
  )


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


def _create_placeholder_context(
    pipeline: pipeline_pb2.Pipeline,
    node: node_proto_view.NodeProtoView,
    input_artifacts: typing_utils.ArtifactMultiMap,
) -> placeholder_utils.ResolutionContext:
  """Collects context information into an object for placeholder resolution."""
  exec_info = data_types.ExecutionInfo(
      input_dict={key: list(value) for key, value in input_artifacts.items()},
      pipeline_node=node.raw_proto(),
      pipeline_info=pipeline.pipeline_info,
      pipeline_run_id=pipeline.runtime_spec.pipeline_run_id.field_value.string_value,
      top_level_pipeline_run_id=pipeline.runtime_spec.top_level_pipeline_run_id,
      frontend_url=pipeline.runtime_spec.frontend_url,
  )

  if not pipeline.deployment_config.Is(
      pipeline_pb2.IntermediateDeploymentConfig.DESCRIPTOR
  ):
    return placeholder_utils.ResolutionContext(exec_info=exec_info)
  depl_config = pipeline_pb2.IntermediateDeploymentConfig()
  pipeline.deployment_config.Unpack(depl_config)
  return placeholder_utils.ResolutionContext(
      exec_info=exec_info,
      executor_spec=deployment_config_utils.get_node_executor_spec(
          depl_config, node.node_info.id
      ),
      platform_config=deployment_config_utils.get_node_platform_config(
          depl_config, node.node_info.id
      ),
      pipeline_platform_config=deployment_config_utils.get_pipeline_platform_config(
          depl_config
      ),
  )


def generate_resolved_info(
    mlmd_handle_like: mlmd_cm.HandleLike,
    node: node_proto_view.NodeProtoView,
    pipeline: pipeline_pb2.Pipeline,
    skip_errors: Iterable[Type[exceptions.InputResolutionError]] = (),
) -> ResolvedInfo:
  """Returns a `ResolvedInfo` object for executing the node or `None` to skip.

  Args:
    mlmd_handle_like: An instance of mlmd handle which connect one MLMD DB, or a
      MLMDConnectionManager which manages connections to multiple MLMD DBs.
    node: The pipeline node for which to generate.
    pipeline: The pipeline proto from which the node was taken (for context).
    skip_errors: A list of errors to skip on the given error types.

  Returns:
    A `ResolvedInfo` with input resolutions. If execution should be skipped,
    ResolvedInfo has empty input_and_params.

  Raises:
    InputResolutionError: If there are some errors when we try to resolve input.
  """
  # Register node contexts.
  contexts = context_lib.prepare_contexts(
      metadata_handle=mlmd_cm.get_handle(mlmd_handle_like),
      node_contexts=node.contexts,
  )

  result = ResolvedInfo(
      contexts=contexts,
      input_and_params=[],
  )

  # Resolve execution properties.
  exec_properties = resolve_exec_properties(node)

  # Resolve inputs.
  try:
    resolved_input_artifacts: Sequence[typing_utils.ArtifactMultiMap] = (
        inputs_utils.resolve_input_artifacts(
            metadata_handle=mlmd_handle_like, pipeline_node=node
        )
    )
  except exceptions.InputResolutionError as e:
    for skip_error in skip_errors:
      if isinstance(e, skip_error):
        logging.info('[%s] Input resolution skipped: %s', node.node_info.id, e)
        return result
    raise
  if not resolved_input_artifacts:
    return result

  for input_artifacts in resolved_input_artifacts:
    try:
      dynamic_exec_properties = inputs_utils.resolve_dynamic_parameters(
          node_parameters=node.parameters,
          context=_create_placeholder_context(pipeline, node, input_artifacts),
      )
    except exceptions.InputResolutionError as e:
      logging.exception(
          '[%s] Parameter resolution error: %s', node.node_info.id, e
      )
      raise

    result.input_and_params.append(
        InputAndParam(
            input_artifacts=input_artifacts,
            exec_properties={**exec_properties, **dynamic_exec_properties},
        )
    )

  return result


def get_executions(
    metadata_handle: metadata.Metadata,
    node: node_proto_view.NodeProtoView,
    limit: Optional[int] = None,
    backfill_token: str = '',
    additional_filters: Optional[List[str]] = None,
) -> List[metadata_store_pb2.Execution]:
  """Returns all executions for the given pipeline node.

  This finds all executions having the same set of contexts as the pipeline
  node.

  Args:
    metadata_handle: A handler to access MLMD db.
    node: The pipeline node for which to obtain executions.
    limit: limit the number of executions return by the function. Executions are
      ordered descendingly by CREATE_TIME, so the newest executions will return.
    backfill_token: If non-empty, only executions with custom property
      `__backfill_token__` set to the value are returned. Should only be set
      when backfilling in ASYNC mode.
    additional_filters: Additional filters to select executions.

  Returns:
    List of executions ordered descendingly by CREATE_TIME for the given node.
  """
  if not node.contexts.contexts:
    return []
  # Get all the contexts associated with the node.
  filter_query = q.And([])

  # "node" context or "pipeline_run" context is a strict sub-context of a
  # "pipeline" context thus we can remove "pipeline" context from the filter
  # query to improve performance.
  filter_contexts = node.contexts.contexts
  context_types = {context.type.name for context in filter_contexts}

  if (
      context_constants.PIPELINE_RUN_CONTEXT_TYPE_NAME in context_types
      or context_constants.NODE_CONTEXT_TYPE_NAME in context_types
  ):
    context_types.discard(context_constants.PIPELINE_CONTEXT_TYPE_NAME)
    filter_contexts = [
        q for q in filter_contexts if q.type.name in context_types
    ]

  for i, context_spec in enumerate(filter_contexts):
    context_type = context_spec.type.name
    context_name = data_types_utils.get_value(context_spec.name)
    filter_query.append(
        q.And([
            f"contexts_{i}.type = '{context_type}'",
            f"contexts_{i}.name = '{context_name}'",
        ])
    )

  if backfill_token:
    filter_query.append(
        (
            'custom_properties.__backfill_token__.string_value ='
            f" '{backfill_token}'"
        ),
    )

  if additional_filters:
    filter_query.extend(additional_filters)

  return metadata_handle.store.get_executions(
      list_options=mlmd.ListOptions(
          order_by=mlmd.OrderByField.CREATE_TIME,
          is_asc=False,
          filter_query=str(filter_query),
          limit=limit,
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


def get_next_active_execution_to_run(
    executions: Sequence[metadata_store_pb2.Execution],
) -> Optional[metadata_store_pb2.Execution]:
  """Returns next active execution to run or `None` if no active executions exist.

  The active execution with lowest index will be returned.

  Args:
    executions: A list of executions

  Returns:
    An active execution or `None` if there is no active execution.
  """
  active_executions = [
      e for e in executions if execution_lib.is_execution_active(e)
  ]
  if not active_executions:
    return None

  # Sorts active executions by index.
  sorted_active_executions = sorted(
      active_executions,
      key=lambda e: e.custom_properties[_EXTERNAL_EXECUTION_INDEX].int_value,
  )
  return sorted_active_executions[0]


def register_executions_from_existing_executions(
    metadata_handle: metadata.Metadata,
    pipeline: pipeline_pb2.Pipeline,
    node: node_proto_view.NodeProtoView,
    existing_executions: List[metadata_store_pb2.Execution],
) -> Sequence[metadata_store_pb2.Execution]:
  """Registers a list of new executions from a list of failed/canceled executions."""
  if not existing_executions:
    return []

  exec_properties = resolve_exec_properties(node)
  exec_type = common_utils.register_type_if_not_exist(
      metadata_handle, node.node_info.type
  )
  new_executions = []
  input_artifacts = []
  for existing_execution in existing_executions:
    input_artifacts_for_existing_execution = execution_lib.get_input_artifacts(
        metadata_handle, existing_execution.id
    )
    try:
      dynamic_exec_properties = inputs_utils.resolve_dynamic_parameters(
          node.parameters,
          _create_placeholder_context(
              pipeline, node, input_artifacts_for_existing_execution
          ),
      )
    except exceptions.InputResolutionError as e:
      logging.exception(
          '[%s] Parameter resolution error: %s', node.node_info.id, e
      )
      raise

    combined_exec_properties = {**exec_properties, **dynamic_exec_properties}
    logging.info(
        'exec properties for execution id: %s: %s',
        existing_execution.id,
        exec_properties,
    )
    logging.info(
        'dynamic exec properties for execution id: %s: %s',
        existing_execution.id,
        dynamic_exec_properties,
    )
    logging.info(
        'combined exec properties for execution id: %s: %s',
        existing_execution.id,
        combined_exec_properties,
    )
    new_execution = execution_lib.prepare_execution(
        metadata_handle=metadata_handle,
        execution_type=exec_type,
        state=metadata_store_pb2.Execution.NEW,
        exec_properties=combined_exec_properties,
        execution_name=str(uuid.uuid4()),
    )
    if node.execution_options.reset_stateful_working_dir:
      # TODO(b/258539860): We may consider removing stateful working dir when
      # users choose to NOT reuse it upon execution retries.
      stateful_working_dir_index = (
          outputs_utils.get_stateful_working_dir_index())
    else:
      # Potentially old executions may have been run under a different state of
      # stateful_working_dir but we only respect the current one in this check.
      # For SYNC pipelines this should only change after an update,
      # but for ASYNC it may happen after a stop/start.
      stateful_working_dir_index = outputs_utils.get_stateful_working_dir_index(
          existing_execution
      )
    # Only copy necessary custom_properties from the failed/canceled execution.
    # LINT.IfChange(new_execution_custom_properties)
    data_types_utils.set_metadata_value(
        new_execution.custom_properties[constants.STATEFUL_WORKING_DIR_INDEX],
        stateful_working_dir_index,
    )
    new_execution.custom_properties[_EXTERNAL_EXECUTION_INDEX].CopyFrom(
        existing_execution.custom_properties[_EXTERNAL_EXECUTION_INDEX]
    )
    # LINT.ThenChange(:execution_custom_properties)
    new_executions.append(new_execution)
    input_artifacts.append(input_artifacts_for_existing_execution)

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
    metadata_handle: metadata.Metadata,
    execution_type: metadata_store_pb2.ExecutionType,
    contexts: Sequence[metadata_store_pb2.Context],
    input_and_params: Sequence[InputAndParam],
) -> Sequence[metadata_store_pb2.Execution]:
  """Registers multiple executions in MLMD.

  Along with the execution:
  -  the input artifacts will be linked to the executions.
  -  the contexts will be linked to both the executions and its input artifacts.

  Args:
    metadata_handle: A handler to access MLMD.
    execution_type: The type of the execution.
    contexts: MLMD contexts to associate with the executions.
    input_and_params: A list of InputAndParams, which includes input_dicts
      (dictionaries of artifacts. One execution will be registered for each of
      the input_dict) and corresponding exec_properties.

  Returns:
    A list of MLMD executions that are registered in MLMD, with id populated.
      All registered executions have a state of NEW.
  """
  executions = []
  registered_execution_type = common_utils.register_type_if_not_exist(
      metadata_handle, execution_type
  )
  for index, input_and_param in enumerate(input_and_params):
    # Prepare executions.
    execution = execution_lib.prepare_execution(
        metadata_handle,
        registered_execution_type,
        metadata_store_pb2.Execution.NEW,
        input_and_param.exec_properties,
        execution_name=str(uuid.uuid4()),
    )
    # LINT.IfChange(execution_custom_properties)
    data_types_utils.set_metadata_value(
        execution.custom_properties[constants.STATEFUL_WORKING_DIR_INDEX],
        outputs_utils.get_stateful_working_dir_index(execution),
    )
    execution.custom_properties[_EXTERNAL_EXECUTION_INDEX].int_value = index
    # LINT.ThenChange(:new_execution_custom_properties)
    executions.append(execution)

  if len(executions) == 1:
    return [
        execution_lib.put_execution(
            metadata_handle,
            executions[0],
            contexts,
            input_artifacts=input_and_params[0].input_artifacts,
        )
    ]

  return execution_lib.put_executions(
      metadata_handle,
      executions,
      contexts,
      [input_and_param.input_artifacts for input_and_param in input_and_params],
  )


def update_external_artifact_type(
    local_mlmd_handle: metadata.Metadata,
    artifacts: Sequence[types.artifact.Artifact],
) -> Sequence[types.artifact.Artifact]:
  """Copies artifact types of external artifacts to local db.

  Args:
    local_mlmd_handle: A handle to access local MLMD db.
    artifacts: A list of artifacts.

  Returns:
    A list of updated artifacts
  """
  updated_artifacts = []
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
      updated_artifacts.append(artifact)

  return updated_artifacts


def get_unprocessed_inputs(
    metadata_handle: metadata.Metadata,
    resolved_info: ResolvedInfo,
    node: node_proto_view.NodeProtoView,
) -> List[InputAndParam]:
  """Get a list of unprocessed input from resolved_info.

  Args:
    metadata_handle: A handle to access local MLMD db.
    resolved_info: Resolved input of a node. It may contain processed and
      unprocessed input.
    node: The pipeline node of the input.

  Returns:
    A list of InputAndParam that have not been processed.
  """
  if not resolved_info.input_and_params:
    return []

  # Finds out the keys that should be ignored.
  input_triggers = node.execution_options.async_trigger.input_triggers
  ignore_keys = {
      k for k, t in input_triggers.items() if k.startswith('_') or t.no_trigger
  }

  max_timestamp_in_each_input: List[int] = []
  for input_and_param in resolved_info.input_and_params:
    max_timestamp_in_one_input = 0
    for key, artifacts in input_and_param.input_artifacts.items():
      if key in ignore_keys or not artifacts:
        continue
      max_timestamp_in_one_input = max(
          max_timestamp_in_one_input,
          max(a.mlmd_artifact.create_time_since_epoch for a in artifacts),
      )
    max_timestamp_in_each_input.append(max_timestamp_in_one_input)

  # A resolved input whose artifacts with max timestamp T is not an input
  # to a execution having creation timestamp < T. So, we only need to
  # get executions with timestamp larger than the minimum timestamp of all
  # the inputs in resolved_info.
  executions = get_executions(
      metadata_handle,
      node,
      additional_filters=[
          (
              'create_time_since_epoch >='
              f' {min(max_timestamp_in_each_input, default=0)}'
          ),
          q.Or([
              'last_known_state = COMPLETE',
              'last_known_state = CACHED',
              'last_known_state = FAILED',
              'last_known_state = CANCELED',
          ]),
      ],
  )

  # Get the successful, failed and canceled executions, and group them by input.
  successful_executions_by_input = collections.defaultdict(list)
  failed_executions_by_input = collections.defaultdict(list)
  cancelled_executions_by_input = collections.defaultdict(list)
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
    input_ids_by_key = event_lib.reconstruct_artifact_id_multimap(input_events)
    # Filters out the keys starting with '_' and the keys should be ignored.
    input_ids_by_key = {
        k: tuple(sorted(v))
        for k, v in input_ids_by_key.items()
        if k not in ignore_keys
    }
    encoded_input = json.dumps(input_ids_by_key, sort_keys=True)
    if execution_lib.is_execution_successful(execution):
      successful_executions_by_input[encoded_input].append(execution)
    elif execution_lib.is_execution_failed(execution):
      failed_executions_by_input[encoded_input].append(execution)
    elif execution_lib.is_execution_canceled(execution):
      cancelled_executions_by_input[encoded_input].append(execution)

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
  # By default, the retry limit in async pipeline is infinite.
  retry_limit = sys.maxsize
  if node.execution_options.HasField('max_execution_retries'):
    retry_limit = node.execution_options.max_execution_retries
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

    # Filters out the keys starting with '_' and the keys should be ignored.
    resolved_input_ids_by_key = {
        k: tuple(sorted(v))
        for k, v in resolved_input_ids_by_key.items()
        if k not in ignore_keys
    }

    encoded_input = json.dumps(resolved_input_ids_by_key, sort_keys=True)
    if len(failed_executions_by_input[encoded_input]) >= retry_limit + 1:
      # This input has failed and has also reached its retry limit.
      logging.info(
          'Node %s has reach retry limit of %d.',
          node.node_info.id,
          retry_limit,
      )
    elif encoded_input not in successful_executions_by_input:
      # This input should be processed.
      failed_or_cancelled_executions = (
          failed_executions_by_input[encoded_input]
          + cancelled_executions_by_input[encoded_input]
      )
      # If the previous stateful_working_dir_index should be reused, save the
      # index into input_and_param.exec_properties
      if (
          not node.execution_options.reset_stateful_working_dir
          and failed_or_cancelled_executions
      ):
        execution_for_retry = execution_lib.sort_executions_newest_to_oldest(
            failed_or_cancelled_executions
        )[0]

        if input_and_param.exec_properties is None:
          input_and_param.exec_properties = {}
        input_and_param.exec_properties[
            constants.STATEFUL_WORKING_DIR_INDEX
        ] = outputs_utils.get_stateful_working_dir_index(execution_for_retry)
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


def generate_tasks_from_one_input(
    metadata_handle: metadata.Metadata,
    node: node_proto_view.NodeProtoView,
    execution: metadata_store_pb2.Execution,
    input_and_param: InputAndParam,
    contexts: Sequence[metadata_store_pb2.Context],
    pipeline: pipeline_pb2.Pipeline,
    execution_node_state: str,
    backfill_token: str = '',
    execution_commit_fn: Optional[
        Callable[
            [
                Optional[metadata_store_pb2.Execution],
                metadata_store_pb2.Execution,
            ],
            None,
        ]
    ] = None,
) -> Sequence[task_lib.Task]:
  """Generates tasks for node an execution.

  Args:
    metadata_handle: Handle to interact with MLMD.
    node: Node to tasks for.
    execution: Metadata execution to generate tasks for.
    input_and_param: Inputs and param for node execution.
    contexts: Contexts for node execution.
    pipeline: Pipeline for this execution.
    execution_node_state: What state the execution should be set to. Should
      always be pstate.NodeState.RUNNING but we can't import pstate here due to
      circular dependencies.
    backfill_token: The backfill token for the execution, if applicable.
    execution_commit_fn: Optional function to be provided when the new execution
      is updated.

  Returns:
    A list of tasks for the node. Guaranteed to be in the form of:
    [UpdateNodeStateTask, ExecNodeTask].
  """

  with mlmd_state.mlmd_execution_atomic_op(
      metadata_handle, execution.id, on_commit=execution_commit_fn
  ) as execution:
    execution.last_known_state = metadata_store_pb2.Execution.RUNNING
  outputs_resolver = outputs_utils.OutputsResolver(
      node,
      pipeline.pipeline_info,
      pipeline.runtime_spec,
      pipeline.execution_mode,
  )
  output_artifacts = outputs_resolver.generate_output_artifacts(execution.id)
  outputs_utils.make_output_dirs(output_artifacts)

  node_uid = task_lib.NodeUid.from_node(pipeline, node)
  tasks = []
  tasks.append(
      task_lib.UpdateNodeStateTask(
          node_uid=node_uid,
          state=execution_node_state,
          backfill_token=backfill_token,
      )
  )
  tasks.append(
      task_lib.ExecNodeTask(
          node_uid=node_uid,
          execution_id=execution.id,
          contexts=contexts,
          input_artifacts=input_and_param.input_artifacts,
          exec_properties=input_and_param.exec_properties,
          output_artifacts=output_artifacts,
          executor_output_uri=outputs_resolver.get_executor_output_uri(
              execution.id
          ),
          stateful_working_dir=outputs_resolver.get_stateful_working_directory(
              execution
          ),
          tmp_dir=outputs_resolver.make_tmp_dir(execution.id),
          pipeline=pipeline,
      )
  )
  return tasks
