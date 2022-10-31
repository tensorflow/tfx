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
"""TFX ml metadata library."""

import collections
import copy
import hashlib
import itertools
import os
import random
import time
import types
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import absl
from tfx.dsl.io import fileio
from tfx.orchestration import data_types
from tfx.orchestration.portable.mlmd import event_lib
from tfx.types import artifact_utils
from tfx.types.artifact import Artifact
from tfx.types.artifact import ArtifactState

import ml_metadata as mlmd
from ml_metadata.proto import metadata_store_pb2

# Number of times to retry initialization of connection.
_MAX_INIT_RETRY = 10

# Maximum number of executions we look at for previous result.
MAX_EXECUTIONS_FOR_CACHE = 100
# Execution state constant. We should replace this with MLMD enum once that is
# ready.
EXECUTION_STATE_CACHED = 'cached'
EXECUTION_STATE_COMPLETE = 'complete'
EXECUTION_STATE_NEW = 'new'
FINAL_EXECUTION_STATES = frozenset(
    (EXECUTION_STATE_CACHED, EXECUTION_STATE_COMPLETE))
# Context type, the following three types of contexts are supported:
#  - pipeline level context is shared within one pipeline, across multiple
#    pipeline runs.
#  - pipeline run level context is shared within one pipeline run, across
#    all component executions in that pipeline run.
#  - component run level context is shared within one component run.
_CONTEXT_TYPE_PIPELINE = 'pipeline'
_CONTEXT_TYPE_PIPELINE_RUN = 'run'
_CONTEXT_TYPE_COMPONENT_RUN = 'component_run'
# Keys of context type properties.
_CONTEXT_TYPE_KEY_COMPONENT_ID = 'component_id'
_CONTEXT_TYPE_KEY_PIPELINE_NAME = 'pipeline_name'
_CONTEXT_TYPE_KEY_RUN_ID = 'run_id'
# Keys of execution type properties.
_EXECUTION_TYPE_KEY_CHECKSUM = 'checksum_md5'
_EXECUTION_TYPE_KEY_COMPONENT_ID = 'component_id'
_EXECUTION_TYPE_KEY_PIPELINE_NAME = 'pipeline_name'
_EXECUTION_TYPE_KEY_PIPELINE_ROOT = 'pipeline_root'
_EXECUTION_TYPE_KEY_RUN_ID = 'run_id'
_EXECUTION_TYPE_KEY_STATE = 'state'
_EXECUTION_TYPE_RESERVED_KEYS = frozenset(
    (_EXECUTION_TYPE_KEY_CHECKSUM, _EXECUTION_TYPE_KEY_PIPELINE_NAME,
     _EXECUTION_TYPE_KEY_PIPELINE_ROOT, _EXECUTION_TYPE_KEY_RUN_ID,
     _EXECUTION_TYPE_KEY_COMPONENT_ID, _EXECUTION_TYPE_KEY_STATE))
# Keys for artifact properties.
_ARTIFACT_TYPE_KEY_STATE = 'state'

# pyformat: disable
ConnectionConfigType = Union[
    metadata_store_pb2.ConnectionConfig,
    metadata_store_pb2.MetadataStoreClientConfig]
# pyformat: enable


def sqlite_metadata_connection_config(
    metadata_db_uri: str) -> metadata_store_pb2.ConnectionConfig:
  """Convenience function to create file based metadata connection config.

  Args:
    metadata_db_uri: uri to metadata db.

  Returns:
    A metadata_store_pb2.ConnectionConfig based on given metadata db uri.
  """
  fileio.makedirs(os.path.dirname(metadata_db_uri))
  connection_config = metadata_store_pb2.ConnectionConfig()
  connection_config.sqlite.filename_uri = metadata_db_uri
  connection_config.sqlite.connection_mode = (
      metadata_store_pb2.SqliteMetadataSourceConfig.READWRITE_OPENCREATE)
  return connection_config


def mysql_metadata_connection_config(
    host: str, port: int, database: str, username: str,
    password: str) -> metadata_store_pb2.ConnectionConfig:
  """Convenience function to create mysql-based metadata connection config.

  Args:
    host: The name or network address of the instance of MySQL to connect to.
    port: The port MySQL is using to listen for connections.
    database: The name of the database to use.
    username: The MySQL login account being used.
    password: The password for the MySQL account being used.

  Returns:
    A metadata_store_pb2.ConnectionConfig based on given metadata db uri.
  """
  return metadata_store_pb2.ConnectionConfig(
      mysql=metadata_store_pb2.MySQLDatabaseConfig(
          host=host,
          port=port,
          database=database,
          user=username,
          password=password))


# TODO(ruoyu): Figure out the story mutable UDFs. We should not reuse previous
# run when having different UDFs.
class Metadata:
  """Helper class to handle metadata I/O.

  Not thread-safe without external synchronisation.
  """

  def __init__(self, connection_config: ConnectionConfigType) -> None:
    self._connection_config = connection_config
    self._store = None
    self._users = 0

  def __enter__(self) -> 'Metadata':
    # TODO(ruoyu): Establishing a connection pool instead of newing
    # a connection every time. Until then, check self._store before usage
    # in every method.
    connection_error = None
    for _ in range(_MAX_INIT_RETRY):
      try:
        self._store = mlmd.MetadataStore(self._connection_config)
      except RuntimeError as err:
        # MetadataStore could raise Aborted error if multiple concurrent
        # connections try to execute initialization DDL in database.
        # This is safe to retry.
        connection_error = err
        time.sleep(random.random())
        continue
      else:
        self._users += 1
        return self

    raise RuntimeError(
        'Failed to establish connection to Metadata storage with error: %s' %
        connection_error)

  def __exit__(self,
               exc_type: Optional[Type[Exception]] = None,
               exc_value: Optional[Exception] = None,
               exc_tb: Optional[types.TracebackType] = None) -> None:
    self._users -= 1
    if self._users == 0:
      self._store = None

  @property
  def store(self) -> mlmd.MetadataStore:
    """Returns underlying MetadataStore.

    Raises:
      RuntimeError: if this instance is not in enter state.
    """
    if self._store is None:
      raise RuntimeError('Metadata object is not in enter state')
    return self._store

  def _prepare_artifact_type(
      self, artifact_type: metadata_store_pb2.ArtifactType
  ) -> metadata_store_pb2.ArtifactType:
    """Prepares artifact types."""
    if artifact_type.id:
      return artifact_type
    # Types can be evolved by adding new fields in newer releases.
    # Here when upserting types:
    # a) we enable `can_add_fields` so that type updates made in the current
    #    release are backward compatible with older release;
    # b) we enable `can_omit_fields` so that the current release is forward
    #    compatible with any type updates made by future release.
    type_id = self.store.put_artifact_type(
        artifact_type=artifact_type, can_add_fields=True, can_omit_fields=True)
    artifact_type.id = type_id
    return artifact_type

  def update_artifact_state(self, artifact: metadata_store_pb2.Artifact,
                            new_state: str) -> None:
    """Update the state of a given artifact."""
    if not artifact.id:
      raise ValueError('Artifact id missing for %s' % artifact)
    # TODO(b/146936257): unify artifact access logic by wrapping raw MLMD
    # artifact protos into tfx.types.Artifact objects at a lower level.
    if _ARTIFACT_TYPE_KEY_STATE in artifact.properties:
      artifact.properties[_ARTIFACT_TYPE_KEY_STATE].string_value = new_state
    else:
      artifact.custom_properties[
          _ARTIFACT_TYPE_KEY_STATE].string_value = new_state
    self.store.put_artifacts([artifact])

  def _upsert_artifacts(self, tfx_artifact_list: List[Artifact],
                        state: str) -> None:
    """Updates or inserts a list of artifacts.

    This call will also update original tfx artifact list to contain the
    artifact type info and artifact id.

    Args:
      tfx_artifact_list: A list of tfx.types.Artifact. This will be updated with
        MLMD artifact type info and MLMD artifact id.
      state: the artifact state to set.
    """
    for raw_artifact in tfx_artifact_list:
      if not raw_artifact.type_id:
        artifact_type = self._prepare_artifact_type(raw_artifact.artifact_type)
        raw_artifact.set_mlmd_artifact_type(artifact_type)
      raw_artifact.state = state
      if state == ArtifactState.PUBLISHED:
        raw_artifact.mlmd_artifact.state = metadata_store_pb2.Artifact.LIVE
    artifact_ids = self.store.put_artifacts(
        [x.mlmd_artifact for x in tfx_artifact_list])
    for a, aid in zip(tfx_artifact_list, artifact_ids):
      a.id = aid

  def publish_artifacts(self, tfx_artifact_list: List[Artifact]) -> None:
    """Publishes artifacts to MLMD.

    This call will also update original tfx artifact list to contain the
    artifact type info and artifact id.

    Args:
      tfx_artifact_list: A list of tfx.types.Artifact which will be updated
    """
    self._upsert_artifacts(tfx_artifact_list, ArtifactState.PUBLISHED)

  def get_artifacts_by_uri(self, uri: str) -> List[metadata_store_pb2.Artifact]:
    """Fetches artifacts given uri."""
    return self.store.get_artifacts_by_uri(uri)

  def get_artifacts_by_type(
      self, type_name: str) -> List[metadata_store_pb2.Artifact]:
    """Fetches artifacts given artifact type name."""
    return self.store.get_artifacts_by_type(type_name)

  def get_published_artifacts_by_type_within_context(
      self, type_names: List[str],
      context_id: int) -> Dict[str, List[metadata_store_pb2.Artifact]]:
    """Fetches artifacts given artifact type name and context id."""
    result = dict((type_name, []) for type_name in type_names)
    all_artifacts_in_context = self.store.get_artifacts_by_context(context_id)
    for type_name in type_names:
      try:
        artifact_type = self.store.get_artifact_type(type_name)
        if artifact_type is None:
          raise mlmd.errors.NotFoundError('No type found.')
      except mlmd.errors.NotFoundError:
        absl.logging.warning('Artifact type %s not registered' % type_name)
        continue

      result[type_name] = [
          a for a in all_artifacts_in_context
          if a.type_id == artifact_type.id and
          a.state == metadata_store_pb2.Artifact.LIVE
      ]
    return result

  @staticmethod
  def _get_legacy_producer_component_id(
      execution: metadata_store_pb2.Execution) -> str:
    return execution.properties[_EXECUTION_TYPE_KEY_COMPONENT_ID].string_value

  def get_qualified_artifacts(
      self,
      contexts: List[metadata_store_pb2.Context],
      type_name: str,
      producer_component_id: Optional[str] = None,
      output_key: Optional[str] = None,
  ) -> List[Artifact]:
    """Gets qualified artifacts that have the right producer info.

    Args:
      contexts: context constraints to filter artifacts
      type_name: type constraint to filter artifacts
      producer_component_id: producer constraint to filter artifacts
      output_key: output key constraint to filter artifacts

    Returns:
      A list of qualified `tfx.types.Artifact`s.
    """

    def _is_qualified_execution(execution):
      if producer_component_id is None:
        return True
      actual = self._get_legacy_producer_component_id(execution)
      return actual == producer_component_id

    def _is_qualified_event(event):
      return event_lib.is_valid_output_event(event, output_key)

    try:
      artifact_type = self.store.get_artifact_type(type_name)
      if not artifact_type:
        raise mlmd.errors.NotFoundError(
            None, None, 'No artifact type found for %s.' % type_name)
    except mlmd.errors.NotFoundError:
      return []

    # Gets the executions that are associated with all contexts.
    assert contexts, 'Must have at least one context.'
    executions_dict = {}
    valid_execution_ids = None
    for context in contexts:
      executions = self.store.get_executions_by_context(context.id)
      if valid_execution_ids is None:
        valid_execution_ids = {e.id for e in executions}
      else:
        valid_execution_ids &= {e.id for e in executions}
      executions_dict.update({e.id: e for e in executions})

    executions_within_context = [
        executions_dict[v] for v in valid_execution_ids]

    # Filters the executions to match producer component id.
    qualified_producer_executions = [
        e.id
        for e in executions_within_context
        if _is_qualified_execution(e)
    ]
    # Gets the output events that have the matched output key.
    qualified_output_events = [
        ev for ev in self.store.get_events_by_execution_ids(
            qualified_producer_executions) if _is_qualified_event(ev)
    ]
    # Gets the candidate artifacts from output events.
    candidate_artifacts = self.store.get_artifacts_by_id(
        list(set(ev.artifact_id for ev in qualified_output_events)))
    # Filters the artifacts that have the right artifact type and state.
    qualified_artifacts = [
        a for a in candidate_artifacts if a.type_id == artifact_type.id and
        a.state == metadata_store_pb2.Artifact.LIVE
    ]
    return [
        artifact_utils.deserialize_artifact(artifact_type, a)
        for a in qualified_artifacts
    ]

  def _prepare_event(self,
                     event_type: metadata_store_pb2.Event.Type,
                     execution_id: Optional[int] = None,
                     artifact_id: Optional[int] = None,
                     key: Optional[str] = None,
                     index: Optional[int] = None) -> metadata_store_pb2.Event:
    """Commits a single event to the repository."""
    event = metadata_store_pb2.Event()
    event.type = event_type
    if execution_id:
      event.execution_id = execution_id
    if artifact_id:
      event.artifact_id = artifact_id
    if key is not None:
      step = event.path.steps.add()
      step.key = key
    if index is not None:
      step = event.path.steps.add()
      step.index = index
    return event

  # TODO(b/143081379): We might need to revisit schema evolution story.
  def _prepare_execution_type(self, type_name: str,
                              exec_properties: Dict[str, Any]) -> int:
    """Gets execution type given execution type name and properties.

    Uses existing type if schema is superset of what is needed. Otherwise tries
    to register new execution type.

    Args:
      type_name: the name of the execution type
      exec_properties: the execution properties included by the execution

    Returns:
      execution type id
    Raises:
      ValueError if new execution type conflicts with existing schema in MLMD.
    """
    existing_execution_type = None
    try:
      existing_execution_type = self.store.get_execution_type(type_name)
      if existing_execution_type is None:
        raise RuntimeError('Execution type is None for %s.' % type_name)
      # If exec_properties contains new entries, execution type schema will be
      # updated in MLMD.
      if all(k in existing_execution_type.properties
             for k in exec_properties.keys()):
        return existing_execution_type.id
      else:
        raise mlmd.errors.NotFoundError('No qualified execution type found.')
    except mlmd.errors.NotFoundError:
      execution_type = metadata_store_pb2.ExecutionType(name=type_name)
      execution_type.properties[
          _EXECUTION_TYPE_KEY_STATE] = metadata_store_pb2.STRING
      # TODO(b/172673657): Make property schema registration more explicit to
      # the component author.
      for k in exec_properties.keys():
        assert k not in _EXECUTION_TYPE_RESERVED_KEYS, (
            'execution properties with reserved key %s') % k
        # TODO(b/172629873): Currently, all execution properties are registered
        # as strings.
        execution_type.properties[k] = metadata_store_pb2.STRING
      # TODO(ruoyu): Find a better place / solution to the checksum logic.
      if 'module_file' in exec_properties:
        execution_type.properties[
            _EXECUTION_TYPE_KEY_CHECKSUM] = metadata_store_pb2.STRING
      execution_type.properties[
          _EXECUTION_TYPE_KEY_PIPELINE_NAME] = metadata_store_pb2.STRING
      execution_type.properties[
          _EXECUTION_TYPE_KEY_PIPELINE_ROOT] = metadata_store_pb2.STRING
      execution_type.properties[
          _EXECUTION_TYPE_KEY_RUN_ID] = metadata_store_pb2.STRING
      execution_type.properties[
          _EXECUTION_TYPE_KEY_COMPONENT_ID] = metadata_store_pb2.STRING

      try:
        # As exec_properties infers an execution_type dynamically. The stored
        # type may have more properties which are inferred from other instances
        # and the current exec_properties may introduce new keys at run time.
        # Here we enable `can_add_fields` and `can_omit_fields` so that the
        # stored execution_type are evolved to union all keys set by different
        # exec_properties instances.
        execution_type_id = self.store.put_execution_type(
            execution_type=execution_type,
            can_add_fields=True,
            can_omit_fields=True)
        absl.logging.debug('Registering an execution type with id %s.' %
                           execution_type_id)
        return execution_type_id
      except mlmd.errors.AlreadyExistsError:
        # The conflict should not happen as all property value type is STRING.
        warning_str = (
            'Conflicting properties in exec_properties comparing with '
            'existing execution type with the same type name. Existing type: '
            '%s, New type: %s') % (existing_execution_type, execution_type)
        absl.logging.warning(warning_str)
        raise ValueError(warning_str)

  def _update_execution_proto(
      self,
      execution: metadata_store_pb2.Execution,
      pipeline_info: Optional[data_types.PipelineInfo] = None,
      component_info: Optional[data_types.ComponentInfo] = None,
      state: Optional[str] = None,
      exec_properties: Optional[Dict[str, Any]] = None,
  ) -> metadata_store_pb2.Execution:
    """Updates the execution proto with given type and state."""
    if state is not None:
      execution.properties[
          _EXECUTION_TYPE_KEY_STATE].string_value = state
    # Forward-compatible change to leverage built-in schema to track states.
    if state == EXECUTION_STATE_CACHED:
      execution.last_known_state = metadata_store_pb2.Execution.CACHED
    elif state == EXECUTION_STATE_COMPLETE:
      execution.last_known_state = metadata_store_pb2.Execution.COMPLETE
    elif state == EXECUTION_STATE_NEW:
      execution.last_known_state = metadata_store_pb2.Execution.RUNNING

    exec_properties = exec_properties or {}
    # TODO(ruoyu): Enforce a formal rule for execution schema change.
    for k, v in exec_properties.items():
      # We always convert execution properties to unicode.
      if isinstance(v, bytes):
        # Decode byte string into a Unicode string.
        v = v.decode('utf-8')
      else:
        # For all other types, store its string representation.
        v = str(v)
      execution.properties[k].string_value = v
    # We also need to checksum UDF file to identify different binary being
    # used. Do we have a better way to checksum a file than hashlib.md5?
    # TODO(ruoyu): Find a better place / solution to the checksum logic.
    # TODO(ruoyu): SHA instead of MD5.
    if 'module_file' in exec_properties and exec_properties[
        'module_file'] and fileio.exists(exec_properties['module_file']):
      contents = fileio.open(exec_properties['module_file'], 'rb').read()
      execution.properties['checksum_md5'].string_value = (
          hashlib.md5(contents).hexdigest().encode('utf-8'))
    if pipeline_info:
      execution.properties[
          'pipeline_name'].string_value = pipeline_info.pipeline_name
      execution.properties[
          'pipeline_root'].string_value = pipeline_info.pipeline_root
      if pipeline_info.run_id:
        execution.properties['run_id'].string_value = pipeline_info.run_id
    if component_info:
      execution.properties[
          'component_id'].string_value = component_info.component_id
    return execution

  def _prepare_execution(
      self,
      state: str,
      exec_properties: Dict[str, Any],
      pipeline_info: data_types.PipelineInfo,
      component_info: data_types.ComponentInfo,
  ) -> metadata_store_pb2.Execution:
    """Creates an execution proto based on the information provided."""
    execution = metadata_store_pb2.Execution()
    execution.type_id = self._prepare_execution_type(
        component_info.component_type, exec_properties)
    self._update_execution_proto(
        execution=execution,
        pipeline_info=pipeline_info,
        component_info=component_info,
        exec_properties=exec_properties,
        state=state)
    absl.logging.debug('Prepared EXECUTION:\n %s', execution)
    return execution

  def _artifact_and_event_pairs(
      self,
      artifact_dict: Dict[str, List[Artifact]],
      event_type: metadata_store_pb2.Event.Type,
      new_state: Optional[str] = None,
      registered_artifacts_ids: Optional[Set[int]] = None
  ) -> List[Tuple[metadata_store_pb2.Artifact,
                  Optional[metadata_store_pb2.Event]]]:
    """Creates a list of [Artifact, [Optional]Event] tuples.

    The result of this function will be used in a MLMD put_execution() call. The
    artifacts will be linked to certain contexts. If an artifact is attached
    with an event, it will be linked with the execution through the event
    created.

    When the id of an artifact is in the registered_artifacts_ids, no event is
    attached to it. Otherwise, an event with given type will be attached to the
    artifact.

    Args:
      artifact_dict: the source of artifacts to work on. For each artifact in
        the dict, creates a tuple for that
      event_type: the event type of the event to be attached to the artifact
      new_state: new state of the artifacts
      registered_artifacts_ids: artifact ids to bypass event creation since they
        are regarded already registered

    Returns:
      A list of [Artifact, [Optional]Event] tuples
    """
    registered_artifacts_ids = registered_artifacts_ids or {}
    result = []
    for key, a_list in artifact_dict.items():
      for index, a in enumerate(a_list):
        if new_state:
          a.state = new_state
          # Forward-compatible change to leverage native artifact schema to
          # log state info.
          if new_state == ArtifactState.PUBLISHED:
            a.mlmd_artifact.state = metadata_store_pb2.Artifact.LIVE
        if a.id and a.id in registered_artifacts_ids:
          result.append(tuple([a.mlmd_artifact]))
        else:
          a.set_mlmd_artifact_type(self._prepare_artifact_type(a.artifact_type))
          result.append(
              (a.mlmd_artifact,
               self._prepare_event(event_type=event_type, key=key,
                                   index=index)))
    return result

  def update_execution(
      self,
      execution: metadata_store_pb2.Execution,
      component_info: data_types.ComponentInfo,
      input_artifacts: Optional[Dict[str, List[Artifact]]] = None,
      output_artifacts: Optional[Dict[str, List[Artifact]]] = None,
      exec_properties: Optional[Dict[str, Any]] = None,
      execution_state: Optional[str] = None,
      artifact_state: Optional[str] = None,
      contexts: Optional[List[metadata_store_pb2.Context]] = None) -> None:
    """Updates the given execution in MLMD based on given information.

    All artifacts provided will be registered if not already. Registered id will
    be reflected inline.

    Args:
      execution: the execution to be updated. It is required that the execution
        passed in has an id.
      component_info: the information of the current running component
      input_artifacts: artifacts to be declared as inputs of the execution
      output_artifacts: artifacts to be declared as outputs of the execution
      exec_properties: execution properties of the execution
      execution_state: state the execution to be updated to
      artifact_state: state the artifacts to be updated to
      contexts: a list of contexts the execution and artifacts to be linked to

    Raises:
      RuntimeError: if the execution to be updated has no id.
    """
    if not execution.id:
      raise RuntimeError('No id attached to the execution to be updated.')
    events = self.store.get_events_by_execution_ids([execution.id])
    registered_input_artifact_ids = set(
        e.artifact_id
        for e in events
        if e.type == metadata_store_pb2.Event.INPUT)
    registered_output_artifact_ids = set(
        e.artifact_id
        for e in events
        if e.type == metadata_store_pb2.Event.OUTPUT)
    artifacts_and_events = []
    if input_artifacts:
      artifacts_and_events.extend(
          self._artifact_and_event_pairs(
              artifact_dict=input_artifacts,
              event_type=metadata_store_pb2.Event.INPUT,
              new_state=artifact_state,
              registered_artifacts_ids=registered_input_artifact_ids))
    if output_artifacts:
      artifacts_and_events.extend(
          self._artifact_and_event_pairs(
              artifact_dict=output_artifacts,
              event_type=metadata_store_pb2.Event.OUTPUT,
              new_state=artifact_state,
              registered_artifacts_ids=registered_output_artifact_ids))
    # If execution properties change, we need to potentially update execution
    # schema.
    if exec_properties:
      execution.type_id = self._prepare_execution_type(
          component_info.component_type, exec_properties)
    if exec_properties or execution_state:
      self._update_execution_proto(
          execution=execution,
          exec_properties=exec_properties,
          state=execution_state,
          pipeline_info=component_info.pipeline_info,
          component_info=component_info)
    _, a_ids, _ = self.store.put_execution(execution, artifacts_and_events,
                                           contexts or [])
    for artifact_and_event, a_id in zip(artifacts_and_events, a_ids):
      artifact_and_event[0].id = a_id

  def register_execution(
      self,
      pipeline_info: data_types.PipelineInfo,
      component_info: data_types.ComponentInfo,
      contexts: List[metadata_store_pb2.Context],
      exec_properties: Optional[Dict[str, Any]] = None,
      input_artifacts: Optional[Dict[str, List[Artifact]]] = None
  ) -> metadata_store_pb2.Execution:
    """Registers a new execution in metadata.

    Args:
      pipeline_info: optional pipeline info of the execution.
      component_info: optional component info of the execution.
      contexts: contexts for current run, all contexts will be linked to the
        execution. In addition, a component run context will be added to the
        contexts list.
      exec_properties: the execution properties of the execution.
      input_artifacts: input artifacts of the execution.

    Returns:
      execution id of the new execution.
    """
    input_artifacts = input_artifacts or {}
    exec_properties = exec_properties or {}
    execution = self._prepare_execution(EXECUTION_STATE_NEW, exec_properties,
                                        pipeline_info, component_info)
    artifacts_and_events = self._artifact_and_event_pairs(
        artifact_dict=input_artifacts,
        event_type=metadata_store_pb2.Event.INPUT)
    component_run_context = self._prepare_context(
        context_type_name=_CONTEXT_TYPE_COMPONENT_RUN,
        context_name=component_info.component_run_context_name,
        properties={
            _CONTEXT_TYPE_KEY_PIPELINE_NAME: pipeline_info.pipeline_name,
            _CONTEXT_TYPE_KEY_RUN_ID: pipeline_info.run_id,
            _CONTEXT_TYPE_KEY_COMPONENT_ID: component_info.component_id
        })
    # Tries to register the execution along with a component run context. If the
    # context already exists, reuse the context and update the existing
    # execution.
    try:
      execution_id, a_ids, context_ids = self.store.put_execution(
          execution=execution,
          artifact_and_events=artifacts_and_events,
          contexts=contexts + [component_run_context])
      execution.id = execution_id
      component_run_context.id = context_ids[-1]
    except mlmd.errors.AlreadyExistsError:
      component_run_context = self.get_component_run_context(component_info)
      absl.logging.debug(
          'Component run context already exists. Reusing the context %s.',
          component_run_context.name)
      [previous_execution] = self.store.get_executions_by_context(
          context_id=component_run_context.id)
      execution.id = previous_execution.id
      _, a_ids, _ = self.store.put_execution(
          execution=execution,
          artifact_and_events=artifacts_and_events,
          contexts=contexts + [component_run_context])
    contexts.append(component_run_context)
    for artifact_and_event, a_id in zip(artifacts_and_events, a_ids):
      artifact_and_event[0].id = a_id
    return execution

  def publish_execution(
      self,
      component_info: data_types.ComponentInfo,
      output_artifacts: Optional[Dict[str, List[Artifact]]] = None,
      exec_properties: Optional[Dict[str, Any]] = None) -> None:
    """Publishes an execution with input and output artifacts info.

    This method will publish any execution with non-final states. It will
    register unseen artifacts and publish events for them.

    Args:
      component_info: component information.
      output_artifacts: output artifacts produced by the execution.
      exec_properties: execution properties for the execution to be published.
    """
    component_run_context = self.get_component_run_context(component_info)
    [execution] = self.store.get_executions_by_context(component_run_context.id)
    contexts = [
        component_run_context,
        self.get_pipeline_run_context(component_info.pipeline_info),
        self.get_pipeline_context(component_info.pipeline_info)
    ]
    contexts = [ctx for ctx in contexts if ctx is not None]
    # If execution state is already in final state, skips publishing.
    if execution.properties[
        _EXECUTION_TYPE_KEY_STATE].string_value in FINAL_EXECUTION_STATES:
      return
    self.update_execution(
        execution=execution,
        component_info=component_info,
        output_artifacts=output_artifacts,
        exec_properties=exec_properties,
        execution_state=EXECUTION_STATE_COMPLETE,
        artifact_state=ArtifactState.PUBLISHED,
        contexts=contexts)

  def _is_eligible_previous_execution(
      self, current_execution: metadata_store_pb2.Execution,
      target_execution: metadata_store_pb2.Execution) -> bool:
    """Compare if the previous execution is same as current execution.

    This method will ignore ID and time related fields.

    Args:
      current_execution: the current execution.
      target_execution: the previous execution to be compared with.

    Returns:
      whether the previous and current executions are the same.
    """
    current_execution.properties['run_id'].string_value = ''
    target_execution.properties['run_id'].string_value = ''
    current_execution.id = target_execution.id
    # Skip comparing time sensitive fields.
    # The execution might not have the create_time_since_epoch or
    # create_time_since_epoch field if the execution is created by an old
    # version before this field is introduced.
    if hasattr(current_execution, 'create_time_since_epoch'):
      current_execution.ClearField('create_time_since_epoch')
    if hasattr(target_execution, 'create_time_since_epoch'):
      target_execution.ClearField('create_time_since_epoch')
    if hasattr(current_execution, 'last_update_time_since_epoch'):
      current_execution.ClearField('last_update_time_since_epoch')
    if hasattr(target_execution, 'last_update_time_since_epoch'):
      target_execution.ClearField('last_update_time_since_epoch')
    return current_execution == target_execution

  def get_cached_outputs(
      self, input_artifacts: Dict[str, List[Artifact]],
      exec_properties: Dict[str, Any], pipeline_info: data_types.PipelineInfo,
      component_info: data_types.ComponentInfo
  ) -> Optional[Dict[str, List[Artifact]]]:
    """Fetches cached output artifacts if any.

    Returns the output artifacts of a cached execution if any. An eligible
    cached execution should take the same input artifacts, execution properties
    and is associated with the same pipeline context.

    Args:
      input_artifacts: inputs used by the run.
      exec_properties: execution properties used by the run.
      pipeline_info: info of the current pipeline run.
      component_info: info of the current component.

    Returns:
      Dict of cached output artifacts if eligible cached execution is found.
      Otherwise, return None.
    """
    absl.logging.debug(
        ('Trying to fetch cached output artifacts with the following info: \n'
         'input_artifacts: %s \n'
         'exec_properties: %s \n'
         'component_info %s') %
        (input_artifacts, exec_properties, component_info))

    # Step 0: Finds the context of the pipeline. No context means no valid cache
    # results.
    context = self.get_pipeline_context(pipeline_info)
    if context is None:
      absl.logging.warning('Pipeline context not available for %s' %
                           pipeline_info)
      return None

    # Step 1: Finds historical executions related to the context in step 0.
    historical_executions = dict(
        (e.id, e) for e in self._store.get_executions_by_context(context.id))

    # Step 2: Filters historical executions to find those that used all the
    # given inputs as input artifacts. The result of this step is a set of
    # reversely sorted execution ids.
    input_ids = collections.defaultdict(set)
    for key, input_list in input_artifacts.items():
      for single_input in input_list:
        input_ids[key].add(single_input.mlmd_artifact.id)
    artifact_to_executions = collections.defaultdict(set)
    for event in self.store.get_events_by_artifact_ids(
        list(set(itertools.chain.from_iterable(input_ids.values())))):
      if event.type == metadata_store_pb2.Event.INPUT:
        artifact_to_executions[event.artifact_id].add(event.execution_id)
    common_execution_ids = sorted(
        set.intersection(
            set(historical_executions.keys()),
            *(artifact_to_executions.values())),
        reverse=True)

    # Step 3: Filters candidate executions further based on the followings:
    #   - Shares the given properties
    #   - Is in complete state
    # The maximum number of candidates is capped by MAX_EXECUTIONS_FOR_CACHE.
    expected_previous_execution = self._prepare_execution(
        EXECUTION_STATE_COMPLETE,
        exec_properties,
        pipeline_info=pipeline_info,
        component_info=component_info)

    candidate_execution_ids = [
        e_id for e_id in common_execution_ids  # pylint: disable=g-complex-comprehension
        if self._is_eligible_previous_execution(
            copy.deepcopy(expected_previous_execution),
            copy.deepcopy(historical_executions[e_id]))
    ]
    candidate_execution_ids = candidate_execution_ids[
        0:min(len(candidate_execution_ids), MAX_EXECUTIONS_FOR_CACHE)]

    # Step 4: Traverse all candidates, if the input artifacts of a candidate
    # match given input artifacts, return the output artifacts of that execution
    # as result. Note that this is necessary since a candidate execution might
    # use more than the given artifacts.
    candidate_execution_to_events = collections.defaultdict(list)
    for event in self.store.get_events_by_execution_ids(
        candidate_execution_ids):
      candidate_execution_to_events[event.execution_id].append(event)

    for events in candidate_execution_to_events.values():
      # Creates the {key -> artifact id set} for the candidate execution.
      current_input_ids = collections.defaultdict(set)
      for event in events:
        if event.type == metadata_store_pb2.Event.INPUT:
          current_input_ids[event.path.steps[0].key].add(event.artifact_id)
      # If all inputs match, tries to get the outputs of the execution and uses
      # as the cached outputs of the current execution.
      if current_input_ids == input_ids:
        cached_outputs = self._get_outputs_of_events(events)
        if cached_outputs is not None:
          return cached_outputs

    return None

  def get_outputs_of_execution(
      self, execution_id: int) -> Optional[Dict[str, List[Artifact]]]:
    """Fetches outputs produced by a historical execution.

    Args:
      execution_id: the id of the execution that produced the outputs.

    Returns:
      A dict of key -> List[Artifact] as the result
    """
    events = self.store.get_events_by_execution_ids([execution_id])
    return self._get_outputs_of_events(events)

  def _get_outputs_of_events(
      self, events: List[metadata_store_pb2.Event]
  ) -> Optional[Dict[str, List[Artifact]]]:
    """Fetches outputs produced by a list of events.

    Args:
      events: events related to the execution id.

    Returns:
      A dictionary mapping execution ID to a list of artifacts produced in that
      execution.
    """
    result = collections.defaultdict(list)

    output_events = [
        event for event in events
        if event.type in [metadata_store_pb2.Event.OUTPUT]
    ]
    output_events.sort(key=lambda e: e.path.steps[1].index)
    cached_output_artifacts = self.store.get_artifacts_by_id(
        [e.artifact_id for e in output_events])
    artifact_types = self.store.get_artifact_types_by_id(
        [a.type_id for a in cached_output_artifacts])

    for event, mlmd_artifact, artifact_type in zip(output_events,
                                                   cached_output_artifacts,
                                                   artifact_types):
      key = event.path.steps[0].key
      tfx_artifact = artifact_utils.deserialize_artifact(
          artifact_type, mlmd_artifact)
      result[key].append(tfx_artifact)

    return result

  def search_artifacts(self, artifact_name: str,
                       pipeline_info: data_types.PipelineInfo,
                       producer_component_id: str) -> List[Artifact]:
    """Search artifacts that matches given info.

    Args:
      artifact_name: the name of the artifact that set by producer component.
        The name is logged both in artifacts and the events when the execution
        being published.
      pipeline_info: the information of the current pipeline
      producer_component_id: the id of the component that produces the artifact

    Returns:
      A list of Artifacts that matches the given info

    Raises:
      RuntimeError: when no matching execution is found given producer info.
    """
    producer_execution = None
    matching_artifact_ids = set()
    # TODO(ruoyu): We need to revisit this when adding support for async
    # execution.
    context = self.get_pipeline_run_context(pipeline_info)
    if context is None:
      raise RuntimeError('Pipeline run context for %s does not exist' %
                         pipeline_info)
    for execution in self.store.get_executions_by_context(context.id):
      if execution.properties[
          'component_id'].string_value == producer_component_id:
        producer_execution = execution
        break
    if not producer_execution:
      raise RuntimeError('Cannot find matching execution with pipeline name %s,'
                         'run id %s and component id %s' %
                         (pipeline_info.pipeline_name, pipeline_info.run_id,
                          producer_component_id))
    for event in self.store.get_events_by_execution_ids([producer_execution.id
                                                        ]):
      if (event.type == metadata_store_pb2.Event.OUTPUT and
          event.path.steps[0].key == artifact_name):
        matching_artifact_ids.add(event.artifact_id)

    # Get relevant artifacts along with their types.
    artifacts_by_id = self.store.get_artifacts_by_id(
        list(matching_artifact_ids))
    matching_artifact_type_ids = list(set(a.type_id for a in artifacts_by_id))
    matching_artifact_types = self.store.get_artifact_types_by_id(
        matching_artifact_type_ids)
    artifact_types = dict(
        zip(matching_artifact_type_ids, matching_artifact_types))

    result_artifacts = []
    for a in artifacts_by_id:
      tfx_artifact = artifact_utils.deserialize_artifact(
          artifact_types[a.type_id], a)
      result_artifacts.append(tfx_artifact)
    return result_artifacts

  def _register_context_type_if_not_exist(
      self, context_type_name: str,
      properties: Dict[str, 'metadata_store_pb2.PropertyType']) -> int:
    """Registers a context type if not exist, otherwise returns existing one.

    Args:
      context_type_name: the name of the context.
      properties: properties of the context.

    Returns:
      id of the desired context type.
    """
    context_type = metadata_store_pb2.ContextType(name=context_type_name)
    for k, t in properties.items():
      context_type.properties[k] = t
    # Types can be evolved by adding new fields in newer releases.
    # Here when upserting types:
    # a) we enable `can_add_fields` so that type updates made in the current
    #    release are backward compatible with older release;
    # b) we enable `can_omit_fields` so that the current release is forward
    #    compatible with any type updates made by future release.
    context_type_id = self.store.put_context_type(
        context_type, can_add_fields=True, can_omit_fields=True)

    return context_type_id

  def _prepare_context(
      self,
      context_type_name: str,
      context_name: str,
      properties: Optional[Dict[str, Union[int, float, str]]] = None
  ) -> metadata_store_pb2.Context:
    """Prepares a context proto."""
    # TODO(ruoyu): Centralize the type definition / mapping along with Artifact
    # property types.
    properties = properties or {}
    property_type_mapping = {
        int: metadata_store_pb2.INT,
        str: metadata_store_pb2.STRING,
        float: metadata_store_pb2.DOUBLE
    }
    context_type_id = self._register_context_type_if_not_exist(
        context_type_name,
        dict(
            (k, property_type_mapping[type(v)]) for k, v in properties.items()))

    context = metadata_store_pb2.Context(
        type_id=context_type_id, name=context_name)
    for k, v in properties.items():
      if isinstance(v, int):
        context.properties[k].int_value = v
      elif isinstance(v, str):
        context.properties[k].string_value = v
      elif isinstance(v, float):
        context.properties[k].double_value = v
      else:
        raise RuntimeError('Unexpected property type: %s' % type(v))
    return context

  def _register_context_if_not_exist(
      self, context_type_name: str, context_name: str,
      properties: Dict[str, Union[int, float, str]]
  ) -> metadata_store_pb2.Context:
    """Registers a context if not exist, otherwise returns the existing one.

    Args:
      context_type_name: the name of the context type desired.
      context_name: the name of the context.
      properties: properties to set in the context.

    Returns:
      id of the desired context

    Raises:
      RuntimeError: when meeting unexpected property type.
    """
    context = self._prepare_context(
        context_type_name=context_type_name,
        context_name=context_name,
        properties=properties)
    try:
      [context_id] = self.store.put_contexts([context])
      context.id = context_id
    except mlmd.errors.AlreadyExistsError:
      absl.logging.debug('Run context %s already exists.', context_name)
      context = self.store.get_context_by_type_and_name(context_type_name,
                                                        context_name)
      assert context is not None, 'Run context is missing for %s.' % (
          context_name)

    absl.logging.debug('ID of run context %s is %s.', context_name, context.id)
    return context

  def get_component_run_context(
      self, component_info: data_types.ComponentInfo
  ) -> Optional[metadata_store_pb2.Context]:
    """Gets the context for the component run.

    Args:
      component_info: component information for the current component run.

    Returns:
      a matched context or None
    """
    return self.store.get_context_by_type_and_name(
        _CONTEXT_TYPE_COMPONENT_RUN, component_info.component_run_context_name)

  def get_pipeline_context(
      self, pipeline_info: data_types.PipelineInfo
  ) -> Optional[metadata_store_pb2.Context]:
    """Gets the context for the pipeline run.

    Args:
      pipeline_info: pipeline information for the current pipeline run.

    Returns:
      a matched context or None
    """
    return self.store.get_context_by_type_and_name(
        _CONTEXT_TYPE_PIPELINE, pipeline_info.pipeline_context_name)

  def get_pipeline_run_context(
      self, pipeline_info: data_types.PipelineInfo
  ) -> Optional[metadata_store_pb2.Context]:
    """Gets the context for the pipeline run.

    Args:
      pipeline_info: pipeline information for the current pipeline run.

    Returns:
      a matched context or None
    """
    if pipeline_info.run_id:
      return self.store.get_context_by_type_and_name(
          _CONTEXT_TYPE_PIPELINE_RUN, pipeline_info.pipeline_run_context_name)
    else:
      return None

  def register_pipeline_contexts_if_not_exists(
      self,
      pipeline_info: data_types.PipelineInfo,
  ) -> List[metadata_store_pb2.Context]:
    """Creates or fetches the pipeline contexts needed for the run.

    There are two potential contexts:
      - Context for the pipeline.
      - Context for the current pipeline run. This is optional, only available
        when run_id is specified.

    Args:
      pipeline_info: pipeline information for current run.

    Returns:
      a list (of size one or two) of context.
    """
    # Gets the pipeline level context.
    result = []
    pipeline_context = self._register_context_if_not_exist(
        context_type_name=_CONTEXT_TYPE_PIPELINE,
        context_name=pipeline_info.pipeline_context_name,
        properties={
            _CONTEXT_TYPE_KEY_PIPELINE_NAME: pipeline_info.pipeline_name
        })
    result.append(pipeline_context)
    absl.logging.debug('Pipeline context [%s : %s]',
                       pipeline_info.pipeline_context_name, pipeline_context.id)
    # If run id exists, gets the pipeline run level context.
    if pipeline_info.run_id:
      pipeline_run_context = self._register_context_if_not_exist(
          context_type_name=_CONTEXT_TYPE_PIPELINE_RUN,
          context_name=pipeline_info.pipeline_run_context_name,
          properties={
              _CONTEXT_TYPE_KEY_PIPELINE_NAME: pipeline_info.pipeline_name,
              _CONTEXT_TYPE_KEY_RUN_ID: pipeline_info.run_id
          })
      result.append(pipeline_run_context)
      absl.logging.debug('Pipeline run context [%s : %s]',
                         pipeline_info.pipeline_run_context_name,
                         pipeline_run_context.id)
    return result
