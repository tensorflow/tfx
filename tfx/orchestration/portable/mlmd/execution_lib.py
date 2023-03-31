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
"""Portable libraries for execution related APIs."""

import collections
import copy
import itertools
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from absl import logging
from tfx import types
from tfx.orchestration import data_types_utils
from tfx.orchestration import metadata
from tfx.orchestration.portable import outputs_utils
from tfx.orchestration.portable.mlmd import artifact_lib
from tfx.orchestration.portable.mlmd import common_utils
from tfx.orchestration.portable.mlmd import event_lib
from tfx.proto.orchestration import execution_result_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import proto_utils
from tfx.utils import typing_utils

from google.protobuf import json_format
import ml_metadata as mlmd
from ml_metadata.proto import metadata_store_pb2

ArtifactState = types.artifact.ArtifactState

# LINT.IfChange(execution_result)
_EXECUTION_RESULT = '__execution_result__'
# LINT.ThenChange()
_PROPERTY_SCHEMA_PREFIX = '__schema__'
_PROPERTY_SCHEMA_SUFFIX = '__'


def is_execution_successful(execution: metadata_store_pb2.Execution) -> bool:
  """Whether or not an execution is successful.

  Args:
    execution: An execution message.

  Returns:
    A bool value indicating whether or not the execution is successful.
  """
  return (execution.last_known_state == metadata_store_pb2.Execution.COMPLETE or
          execution.last_known_state == metadata_store_pb2.Execution.CACHED)


def is_execution_active(execution: metadata_store_pb2.Execution) -> bool:
  """Returns `True` if an execution is active.

  Args:
    execution: An execution message.

  Returns:
    A bool value indicating whether or not the execution is active.
  """
  return (execution.last_known_state == metadata_store_pb2.Execution.NEW or
          execution.last_known_state == metadata_store_pb2.Execution.RUNNING)


def is_execution_running(execution: metadata_store_pb2.Execution) -> bool:
  """Returns `True` if an execution is running.

  Args:
    execution: An execution message.

  Returns:
    A bool value indicating whether or not the execution is running.
  """
  return execution.last_known_state == metadata_store_pb2.Execution.RUNNING


def is_execution_canceled(execution: metadata_store_pb2.Execution) -> bool:
  """Whether or not an execution is canceled.

  Args:
    execution: An execution message.

  Returns:
    A bool value indicating whether or not the execution is canceled.
  """
  return execution.last_known_state == metadata_store_pb2.Execution.CANCELED


def is_execution_failed(execution: metadata_store_pb2.Execution) -> bool:
  """Whether or not an execution is failed.

  Args:
    execution: An execution message.

  Returns:
    A bool value indicating whether or not the execution is failed.
  """
  return execution.last_known_state == metadata_store_pb2.Execution.FAILED


def is_internal_key(key: str) -> bool:
  """Returns `True` if the key is an internal-only execution property key."""
  return key.startswith('__')


def remove_internal_keys(d: Dict[str, Any]) -> Dict[str, Any]:
  return {k: v for k, v in d.items() if not is_internal_key(k)}


def is_schema_key(key: str) -> bool:
  """Returns `True` if the input key corresponds to a schema stored in execution property."""
  return re.fullmatch(r'^__schema__.*__$', key) is not None


def get_schema_key(key: str) -> str:
  """Returns key for storing execution property schema."""
  return _PROPERTY_SCHEMA_PREFIX + key + _PROPERTY_SCHEMA_SUFFIX


def sort_executions_newest_to_oldest(
    executions: Iterable[metadata_store_pb2.Execution]
) -> List[metadata_store_pb2.Execution]:
  """Returns MLMD executions in sorted order, newest to oldest.

  Args:
    executions: An iterable of MLMD executions.

  Returns:
    Executions sorted newest to oldest (based on MLMD execution creation time).
  """
  return sorted(
      executions, key=lambda e: e.create_time_since_epoch, reverse=True)


def prepare_execution(
    metadata_handler: metadata.Metadata,
    execution_type: metadata_store_pb2.ExecutionType,
    state: metadata_store_pb2.Execution.State,
    exec_properties: Optional[Mapping[str, types.ExecPropertyTypes]] = None,
    execution_name: str = '',
) -> metadata_store_pb2.Execution:
  """Creates an execution proto based on the information provided.

  Args:
    metadata_handler: A handler to access MLMD store.
    execution_type: A metadata_pb2.ExecutionType message describing the type of
      the execution.
    state: The state of the execution.
    exec_properties: Execution properties that need to be attached.
    execution_name: Name of the execution.

  Returns:
    A metadata_store_pb2.Execution message.
  """
  execution = metadata_store_pb2.Execution()
  execution.last_known_state = state
  execution.type_id = common_utils.register_type_if_not_exist(
      metadata_handler, execution_type).id
  if execution_name:
    execution.name = execution_name

  exec_properties = exec_properties or {}
  # For every execution property, put it in execution.properties if its key is
  # in execution type schema. Otherwise, put it in execution.custom_properties.
  for k, v in exec_properties.items():
    value = pipeline_pb2.Value()
    value = data_types_utils.set_parameter_value(value, v)

    if value.HasField('schema'):
      # Stores schema in custom_properties for non-primitive types to allow
      # parsing in later stages.
      data_types_utils.set_metadata_value(
          execution.custom_properties[get_schema_key(k)],
          proto_utils.proto_to_json(value.schema))

    if (execution_type.properties.get(k) ==
        data_types_utils.get_metadata_value_type(v)):
      execution.properties[k].CopyFrom(value.field_value)
    else:
      execution.custom_properties[k].CopyFrom(value.field_value)
  logging.debug('Prepared EXECUTION:\n %s', execution)
  return execution


def _create_artifact_and_event_pairs(
    metadata_handler: metadata.Metadata,
    artifact_dict: typing_utils.ArtifactMultiMap,
    event_type: metadata_store_pb2.Event.Type,
) -> List[Tuple[metadata_store_pb2.Artifact, metadata_store_pb2.Event]]:
  """Creates a list of [Artifact, Event] tuples.

  The result of this function will be used in a MLMD put_execution() call.

  Args:
    metadata_handler: A handler to access MLMD store.
    artifact_dict: The source of artifacts to work on. For each unique artifact
      in the dict, creates a tuple for that. Note that all artifacts of the same
      key in the artifact_dict are expected to share the same artifact type. If
      the same artifact is used for multiple keys, several event paths will be
      generated for the same event.
    event_type: The event type of the event to be attached to the artifact

  Returns:
    A list of [Artifact, Event] tuples
  """
  if not artifact_dict:
    return []

  result = []
  artifact_event_map = dict()
  for key, artifact_list in artifact_dict.items():
    artifact_type = None
    for index, artifact in enumerate(artifact_list):
      if (artifact.mlmd_artifact.HasField('id') and
          artifact.id in artifact_event_map):
        event_lib.add_event_path(
            artifact_event_map[artifact.id][1], key=key, index=index)
      else:
        # TODO(b/153904840): If artifact id is present, skip putting the
        # artifact into the pair when MLMD API is ready.
        event = event_lib.generate_event(
            event_type=event_type, key=key, index=index)
        # Reuses already registered type in the same list whenever possible as
        # the artifacts in the same list share the same artifact type.
        if artifact_type:
          assert artifact_type.name == artifact.artifact_type.name, (
              'Artifacts under the same key should share the same artifact '
              'type.')
        artifact_type = common_utils.register_type_if_not_exist(
            metadata_handler, artifact.artifact_type)
        artifact.set_mlmd_artifact_type(artifact_type)
        if artifact.mlmd_artifact.HasField('id'):
          artifact_event_map[artifact.id] = (artifact.mlmd_artifact, event)
        else:
          result.append((artifact.mlmd_artifact, event))
  result.extend(list(artifact_event_map.values()))
  return result


def put_execution(
    metadata_handler: metadata.Metadata,
    execution: metadata_store_pb2.Execution,
    contexts: Sequence[metadata_store_pb2.Context],
    input_artifacts: Optional[typing_utils.ArtifactMultiMap] = None,
    output_artifacts: Optional[typing_utils.ArtifactMultiMap] = None,
    input_event_type: metadata_store_pb2.Event.Type = metadata_store_pb2.Event
    .INPUT,
    output_event_type: metadata_store_pb2.Event.Type = metadata_store_pb2.Event
    .OUTPUT
) -> metadata_store_pb2.Execution:
  """Writes an execution-centric subgraph to MLMD.

  This function mainly leverages metadata.put_execution() method to write the
  execution centric subgraph to MLMD.

  Args:
    metadata_handler: A handler to access MLMD.
    execution: The execution to be written to MLMD.
    contexts: MLMD contexts to associated with the execution.
    input_artifacts: Input artifacts of the execution. Each artifact will be
      linked with the execution through an event with type input_event_type.
      Each artifact will also be linked with every context in the `contexts`
      argument.
    output_artifacts: Output artifacts of the execution. Each artifact will be
      linked with the execution through an event with type output_event_type.
      Each artifact will also be linked with every context in the `contexts`
      argument.
    input_event_type: The type of the input event, default to be INPUT.
    output_event_type: The type of the output event, default to be OUTPUT.

  Returns:
    An MLMD execution that is written to MLMD, with id pupulated.
  """
  artifact_and_events = []
  if input_artifacts:
    artifact_and_events.extend(
        _create_artifact_and_event_pairs(
            metadata_handler=metadata_handler,
            artifact_dict=input_artifacts,
            event_type=input_event_type))
  if output_artifacts:
    outputs_utils.tag_output_artifacts_with_version(output_artifacts)
    artifact_and_events.extend(
        _create_artifact_and_event_pairs(
            metadata_handler=metadata_handler,
            artifact_dict=output_artifacts,
            event_type=output_event_type))
  execution_id, artifact_ids, contexts_ids = (
      metadata_handler.store.put_execution(
          execution=execution,
          artifact_and_events=artifact_and_events,
          contexts=contexts,
          reuse_context_if_already_exist=True,
          reuse_artifact_if_already_exist_by_external_id=True))
  execution.id = execution_id
  for artifact_and_event, a_id in zip(artifact_and_events, artifact_ids):
    artifact, _ = artifact_and_event
    artifact.id = a_id
  for context, c_id in zip(contexts, contexts_ids):
    context.id = c_id

  return execution


def put_executions(
    metadata_handler: metadata.Metadata,
    executions: Sequence[metadata_store_pb2.Execution],
    contexts: Sequence[metadata_store_pb2.Context],
    input_artifacts_maps: Optional[Sequence[
        typing_utils.ArtifactMultiMap]] = None,
    output_artifacts_maps: Optional[Sequence[
        typing_utils.ArtifactMultiMap]] = None,
    input_event_type: metadata_store_pb2.Event.Type = metadata_store_pb2.Event
    .INPUT,
    output_event_type: metadata_store_pb2.Event.Type = metadata_store_pb2.Event
    .OUTPUT
) -> Sequence[metadata_store_pb2.Execution]:
  """Writes an execution-centric subgraph to MLMD.

  This function mainly leverages metadata.put_lineage_subgraph() method to write
  the execution centric subgraph to MLMD.

  Args:
    metadata_handler: A handler to access MLMD.
    executions: A list of executions to be written to MLMD.
    contexts: A list of MLMD contexts to associated with all the executions.
    input_artifacts_maps: A list of ArtifactMultiMap for input. Each of the
      ArtifactMultiMap links with one execution. None or empty
      input_artifacts_maps means no input for the executions.
    output_artifacts_maps: A list of ArtifactMultiMap for output. Each of the
      ArtifactMultiMap links with one execution. None or empty
      output_artifacts_maps means no input for the executions.
    input_event_type: The type of the input event, default to be INPUT.
    output_event_type: The type of the output event, default to be OUTPUT.

  Returns:
    A list of MLMD executions that are written to MLMD, with id pupulated.
  """
  if input_artifacts_maps and len(executions) != len(input_artifacts_maps):
    raise ValueError(
        f'The number of executions {len(executions)} should be the same as '
        f'the number of input ArtifactMultiMap {len(input_artifacts_maps)}.')
  if output_artifacts_maps and len(executions) != len(output_artifacts_maps):
    raise ValueError(
        f'The number of executions {len(executions)} should be the same as '
        f'the number of output ArtifactMultiMap {len(output_artifacts_maps)}.')

  artifacts = []
  artifact_event_edges = []
  if input_artifacts_maps:
    for idx, input_artifacts in enumerate(input_artifacts_maps):
      artifact_and_event_pairs = _create_artifact_and_event_pairs(
          metadata_handler, input_artifacts, event_type=input_event_type
      )
      for artifact, event in artifact_and_event_pairs:
        artifacts.append(artifact)
        artifact_event_edges.append((idx, len(artifacts) - 1, event))
  if output_artifacts_maps:
    for idx, output_artifacts in enumerate(output_artifacts_maps):
      outputs_utils.tag_output_artifacts_with_version(output_artifacts)
      artifact_and_event_pairs = _create_artifact_and_event_pairs(
          metadata_handler, output_artifacts, event_type=output_event_type)
      for artifact, event in artifact_and_event_pairs:
        artifacts.append(artifact)
        artifact_event_edges.append((idx, len(artifacts) - 1, event))

  try:
    execution_ids, artifact_ids, context_ids = (
        metadata_handler.store.put_lineage_subgraph(
            executions,
            artifacts,
            contexts,
            artifact_event_edges,
            reuse_context_if_already_exist=True,
            reuse_artifact_if_already_exist_by_external_id=True,
        )
    )
    for execution, execution_id in zip(executions, execution_ids):
      execution.id = execution_id
    for artifact, artifact_id in zip(artifacts, artifact_ids):
      artifact.id = artifact_id
    for context, context_id in zip(contexts, context_ids):
      context.id = context_id
    return executions
  except mlmd.errors.UnimplementedError:
    logging.warning(
        'PutLineageSubgraph API do not exist. Falling back to PutExecution.'
    )
    result = []
    for execution, input_artifacts, output_artifacts in itertools.zip_longest(
        executions, input_artifacts_maps or [], output_artifacts_maps or []
    ):
      result.append(
          put_execution(
              metadata_handler=metadata_handler,
              execution=execution,
              contexts=contexts,
              input_artifacts=input_artifacts,
              output_artifacts=output_artifacts,
              input_event_type=input_event_type,
              output_event_type=output_event_type,
          )
      )
    return result


def _artifact_maps_contain_same_uris(
    left: typing_utils.ArtifactMultiMap,
    right: typing_utils.ArtifactMultiMap) -> bool:
  """Returns true if the artifact maps contain exactly the same URIs."""
  if left.keys() != right.keys():
    return False

  for key, left_artifact_list in left.items():
    right_artifact_list = right[key]
    if len(left_artifact_list) != len(right_artifact_list):
      return False
    for left_artifact, right_artifact in zip(left_artifact_list,
                                             right_artifact_list):
      if left_artifact.uri != right_artifact.uri:
        return False
  return True


def register_pending_output_artifacts(
    metadata_handle: metadata.Metadata, execution_id: int,
    output_artifacts: typing_utils.ArtifactMultiMap) -> None:
  """Registers pending output artifacts for a given execution in MLMD.

  This operation does not modify the execution itself, but simply creates the
  output artifacts in MLMD with a PENDING state and adds events with type
  PENDING_OUTPUT to associate them with the execution. The output artifacts will
  be modified in-place to add IDs as registered by MLMD.

  This function is idempotent if called more than once with the same output
  artifact dict. In that case, the function will find the already registered
  output artifacts for the execution and reuse their IDs. This function cannot,
  however, be called for the same execution more than once with different
  output artifact dicts.

  Args:
    metadata_handle: A handle to access MLMD
    execution_id: The ID of an existing execution to apply output artifacts to.
    output_artifacts: The output artifact dict to register. Artifacts will be
      modified in place to add IDs after creation in MLMD.

  Raises:
    ValueError if the specified execution is not active, or if this function
    was called once before for the same execution but with a different output
    artifact dict.
  """
  [execution] = metadata_handle.store.get_executions_by_id([execution_id])
  if not is_execution_running(execution):
    raise ValueError('Cannot register output artifacts on inactive '
                     f'execution ID={execution_id} with state = '
                     f'{execution.last_known_state}')

  for artifact in itertools.chain.from_iterable(output_artifacts.values()):
    artifact.state = ArtifactState.PENDING

  existing_pending_output_artifacts = get_pending_output_artifacts(
      metadata_handle, execution_id)
  if existing_pending_output_artifacts:
    # Reuse existing pending output artifacts if this function was already
    # called with the same arguments and for the same execution.
    if not _artifact_maps_contain_same_uris(output_artifacts,
                                            existing_pending_output_artifacts):
      raise ValueError('Pending output artifacts were already registered for'
                       'this execution, but with a different set of output'
                       'artifacts.\nProvided output_artifacts:\n'
                       f'{output_artifacts}\nExisting pending output '
                       f'artifacts:\n{existing_pending_output_artifacts}')
    # The output maps have the same content so copy over existing artifact IDs.
    for key, existing_artifact_list in (
        existing_pending_output_artifacts.items()):
      provided_artifact_list = output_artifacts[key]
      for existing_artifact, provided_artifact in zip(existing_artifact_list,
                                                      provided_artifact_list):
        provided_artifact.id = existing_artifact.id
        provided_artifact.type_id = existing_artifact.type_id
  else:
    # Register the pending output artifacts for this execution.
    contexts = metadata_handle.store.get_contexts_by_execution(execution_id)
    _ = put_execution(
        metadata_handle,
        execution,
        contexts,
        output_artifacts=output_artifacts,
        output_event_type=metadata_store_pb2.Event.PENDING_OUTPUT)


def get_executions_associated_with_all_contexts(
    metadata_handler: metadata.Metadata,
    contexts: Iterable[metadata_store_pb2.Context]
) -> List[metadata_store_pb2.Execution]:
  """Returns executions that are associated with all given contexts.

  Args:
    metadata_handler: A handler to access MLMD.
    contexts: MLMD contexts for which to fetch associated executions.

  Returns:
    A list of executions associated with all given contexts.
  """
  executions_dict = None
  for context in contexts:
    executions = metadata_handler.store.get_executions_by_context(context.id)
    if executions_dict is None:
      executions_dict = {e.id: e for e in executions}
    else:
      executions_dict = {e.id: e for e in executions if e.id in executions_dict}
  return list(executions_dict.values()) if executions_dict else []


def get_artifact_ids_by_event_type_for_execution_id(
    metadata_handler: metadata.Metadata,
    execution_id: int) -> Dict['metadata_store_pb2.Event.Type', Set[int]]:
  """Returns artifact ids corresponding to the execution id grouped by event type.

  Args:
    metadata_handler: A handler to access MLMD.
    execution_id: Id of the execution for which to get artifact ids.

  Returns:
    A `dict` mapping event type to `set` of artifact ids.
  """
  events = metadata_handler.store.get_events_by_execution_ids([execution_id])
  result = collections.defaultdict(set)
  for event in events:
    result[event.type].add(event.artifact_id)
  return result


def get_input_artifacts(
    metadata_handle: metadata.Metadata,
    execution_id: int) -> typing_utils.ArtifactMultiDict:
  """Gets input artifacts of the execution.

  Each execution is associated with a single input artifacts multimap, where the
  key is the same key associated with `NodeInputs.inputs` in the pipeline IR.
  Artifacts and event information are fetched from the MLMD.

  Args:
    metadata_handle: A Metadata instance that is in entered state.
    execution_id: A valid MLMD execution ID. If the execution_id does not exist,
        this function will return an empty dict instead of raising an error.

  Returns:
    A reconstructed input artifacts multimap.
  """
  events = metadata_handle.store.get_events_by_execution_ids([execution_id])
  input_events = [e for e in events if event_lib.is_valid_input_event(e)]
  artifacts = artifact_lib.get_artifacts_by_ids(
      metadata_handle, [e.artifact_id for e in input_events])
  return event_lib.reconstruct_artifact_multimap(artifacts, input_events)


def get_output_artifacts(
    metadata_handle: metadata.Metadata,
    execution_id: int) -> typing_utils.ArtifactMultiDict:
  """Gets output artifacts of the execution.

  Each execution is associated with a single output artifacts multimap, where
  the key is the same key associated with `NodeOutputs.outputs` in the pipeline
  IR. Artifacts and event information are fetched from the MLMD.

  Args:
    metadata_handle: A Metadata instance that is in entered state.
    execution_id: A valid MLMD execution ID. If the execution_id does not exist,
        this function will return an empty dict instead of raising an error.

  Returns:
    A reconstructed output artifacts multimap.
  """
  events = metadata_handle.store.get_events_by_execution_ids([execution_id])
  output_events = [e for e in events if event_lib.is_valid_output_event(e)]
  artifacts = artifact_lib.get_artifacts_by_ids(
      metadata_handle, [e.artifact_id for e in output_events])
  return event_lib.reconstruct_artifact_multimap(artifacts, output_events)


def get_pending_output_artifacts(
    metadata_handle: metadata.Metadata,
    execution_id: int) -> typing_utils.ArtifactMultiDict:
  """Gets pending output artifacts of the execution.

  Each execution is associated with a single pending output artifacts multimap,
  where the key is the same key associated with `NodeOutputs.outputs` in the
  pipeline IR. Artifacts and event information are fetched from MLMD.

  Args:
    metadata_handle: A Metadata instance that is in entered state.
    execution_id: A valid MLMD execution ID. If the execution_id does not exist,
      this function will return an empty dict instead of raising an error.

  Returns:
    A reconstructed pending output artifacts multimap.
  """
  events = metadata_handle.store.get_events_by_execution_ids([execution_id])
  pending_output_events = [
      e for e in events if e.type == metadata_store_pb2.Event.PENDING_OUTPUT
  ]
  artifacts = artifact_lib.get_artifacts_by_ids(
      metadata_handle, [e.artifact_id for e in pending_output_events])
  return event_lib.reconstruct_artifact_multimap(artifacts,
                                                 pending_output_events)


def set_execution_result(execution_result: execution_result_pb2.ExecutionResult,
                         execution: metadata_store_pb2.Execution):
  """Sets execution result as a custom property of execution.

  Args:
    execution_result: The result of execution. It is typically generated by
      executor.
    execution: The execution to set to.
  """
  # TODO(b/161832842): Switch to PROTO value type to circumvent TypeError which
  # may be raised when converting embedded `Any` protos.
  try:
    execution.custom_properties[_EXECUTION_RESULT].string_value = (
        json_format.MessageToJson(execution_result))
  except TypeError:
    logging.exception(
        'Unable to set execution_result as custom property of the execution '
        'due to error, will attempt again by clearing `metadata_details`...'
    )
    try:
      execution_result = copy.deepcopy(execution_result)
      execution_result.ClearField('metadata_details')
      execution.custom_properties[_EXECUTION_RESULT].string_value = (
          json_format.MessageToJson(execution_result)
      )
    except TypeError:
      logging.exception(
          'Skipped setting execution_result as custom property of the '
          'execution due to error'
      )


def get_execution_result(
    execution: metadata_store_pb2.Execution, ignore_parse_errors=False
) -> Optional[execution_result_pb2.ExecutionResult]:
  """Gets execution result stored as custom property of execution if it exists.

  The custom property should have been set using `set_execution_result`. If
  ignore_parse_errors=True, JSON parsing errors will lead to returning `None`.

  Args:
    execution: The execution from which to read execution result.
    ignore_parse_errors: If True, JSON parsing errors are ignored and a None is
      returned.

  Returns:
    An `ExecutionResult` object if one exists.
  """
  value = execution.custom_properties.get(_EXECUTION_RESULT)
  if not value:
    return None
  execution_result = execution_result_pb2.ExecutionResult()
  try:
    json_format.Parse(value.string_value, execution_result)
  except json_format.ParseError:
    logging.exception(
        'Error parsing json object, %s',
        'ignoring' if ignore_parse_errors else 're-raising',
    )
    if ignore_parse_errors:
      return None
    raise
  return execution_result
