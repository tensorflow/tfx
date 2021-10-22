# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Pipeline state management functionality."""

import base64
import contextlib
import threading
import time
from typing import Dict, Iterator, List, Mapping, Optional, Tuple

from absl import logging
import attr
from tfx import types
from tfx.orchestration import data_types_utils
from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import env
from tfx.orchestration.experimental.core import mlmd_state
from tfx.orchestration.experimental.core import orchestration_options
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import task_gen_utils
from tfx.orchestration.portable.mlmd import context_lib
from tfx.orchestration.portable.mlmd import execution_lib
from tfx.proto.orchestration import pipeline_pb2
from tfx.proto.orchestration import run_state_pb2
from tfx.utils import json_utils
from tfx.utils import status as status_lib

from ml_metadata.proto import metadata_store_pb2

_ORCHESTRATOR_RESERVED_ID = '__ORCHESTRATOR__'
_PIPELINE_IR = 'pipeline_ir'
_STOP_INITIATED = 'stop_initiated'
_PIPELINE_RUN_ID = 'pipeline_run_id'
_PIPELINE_STATUS_CODE = 'pipeline_status_code'
_PIPELINE_STATUS_MSG = 'pipeline_status_msg'
_NODE_STATES = 'node_states'
_PIPELINE_RUN_METADATA = 'pipeline_run_metadata'
_UPDATED_PIPELINE_IR = 'updated_pipeline_ir'
_ORCHESTRATOR_EXECUTION_TYPE = metadata_store_pb2.ExecutionType(
    name=_ORCHESTRATOR_RESERVED_ID,
    properties={_PIPELINE_IR: metadata_store_pb2.STRING})

_last_state_change_time_secs = -1.0
_state_change_time_lock = threading.Lock()

_EXECUTION_STATE_TO_RUN_STATE_MAP = {
    metadata_store_pb2.Execution.State.RUNNING:
        run_state_pb2.RunState.RUNNING,
    metadata_store_pb2.Execution.State.FAILED:
        run_state_pb2.RunState.FAILED,
    metadata_store_pb2.Execution.State.COMPLETE:
        run_state_pb2.RunState.COMPLETE,
    metadata_store_pb2.Execution.State.CACHED:
        run_state_pb2.RunState.COMPLETE,
    metadata_store_pb2.Execution.State.CANCELED:
        run_state_pb2.RunState.STOPPED,
}


@attr.s(auto_attribs=True, kw_only=True)
class NodeState(json_utils.Jsonable):
  """Records node state.

  Attributes:
    state: Current state of the node.
    status: Status of the node in state STOPPING or STOPPED.
  """

  STARTING = 'starting'  # Pending work before state can change to STARTED.
  STARTED = 'started'  # Node is ready for execution.
  STOPPING = 'stopping'  # Pending work before state can change to STOPPED.
  STOPPED = 'stopped'  # Node execution is stopped.
  RUNNING = 'running'  # Node is under active execution (i.e. triggered).
  COMPLETE = 'complete'  # Node execution completed successfully.
  SKIPPED = 'skipped'  # Node execution skipped due to conditional.
  PAUSED = 'paused'  # Node was paused and may be resumed in the future.
  FAILED = 'failed'  # Node execution failed due to errors.

  state: str = attr.ib(
      default=STARTED,
      validator=attr.validators.in_([
          STARTING, STARTED, STOPPING, STOPPED, RUNNING, COMPLETE, SKIPPED,
          PAUSED, FAILED
      ]),
      on_setattr=attr.setters.validate)
  status_code: Optional[int] = None
  status_msg: str = ''

  @property
  def status(self) -> Optional[status_lib.Status]:
    if self.status_code is not None:
      return status_lib.Status(code=self.status_code, message=self.status_msg)
    return None

  def update(self,
             state: str,
             status: Optional[status_lib.Status] = None) -> None:
    self.state = state
    if status is not None:
      self.status_code = status.code
      self.status_msg = status.message
    else:
      self.status_code = None
      self.status_msg = ''

  def is_startable(self) -> bool:
    """Returns True if the node can be started."""
    return self.state in set(
        [self.PAUSED, self.STOPPING, self.STOPPED, self.FAILED])

  def is_stoppable(self) -> bool:
    """Returns True if the node can be stopped."""
    return self.state in set(
        [self.STARTING, self.STARTED, self.RUNNING, self.PAUSED])

  def is_success(self) -> bool:
    return is_node_state_success(self.state)

  def is_failure(self) -> bool:
    return is_node_state_failure(self.state)


def is_node_state_success(state: str) -> bool:
  return state in (NodeState.COMPLETE, NodeState.SKIPPED)


def is_node_state_failure(state: str) -> bool:
  return state == NodeState.FAILED


_NODE_STATE_TO_RUN_STATE_MAP = {
    NodeState.STARTING: run_state_pb2.RunState.UNKNOWN,
    NodeState.STARTED: run_state_pb2.RunState.READY,
    NodeState.STOPPING: run_state_pb2.RunState.UNKNOWN,
    NodeState.STOPPED: run_state_pb2.RunState.STOPPED,
    NodeState.RUNNING: run_state_pb2.RunState.RUNNING,
    NodeState.COMPLETE: run_state_pb2.RunState.COMPLETE,
    NodeState.SKIPPED: run_state_pb2.RunState.SKIPPED,
    NodeState.PAUSED: run_state_pb2.RunState.PAUSED,
    NodeState.FAILED: run_state_pb2.RunState.FAILED
}


def record_state_change_time() -> None:
  """Records current time at the point of function call as state change time.

  This function may be called after any operation that changes pipeline state or
  node execution state that requires further processing in the next iteration of
  the orchestration loop. As an optimization, the orchestration loop can elide
  wait period in between iterations when such state change is detected.
  """
  global _last_state_change_time_secs
  with _state_change_time_lock:
    _last_state_change_time_secs = time.time()


def last_state_change_time_secs() -> float:
  """Returns last recorded state change time as seconds since epoch."""
  with _state_change_time_lock:
    return _last_state_change_time_secs


class PipelineState:
  """Context manager class for dealing with pipeline state.

  The state of a pipeline is stored as an MLMD execution and this class provides
  methods for creating, accessing and mutating it. Methods must be invoked
  inside the pipeline state context for thread safety and keeping in-memory
  state in sync with the corresponding state in MLMD. If the underlying pipeline
  execution is mutated, it is automatically committed when exiting the context
  so no separate commit operation is needed.

  Note that `mlmd_state.mlmd_execution_atomic_op` is used under the hood and
  hence any updates made to the pipeline state within the context of one
  PipelineState instance are also reflected inside the context of all other
  PipelineState instances (for the same pipeline) that may be alive within the
  process.

  Attributes:
    mlmd_handle: Handle to MLMD db.
    pipeline: The pipeline proto associated with this `PipelineState` object.
      TODO(b/201294315): Fix self.pipeline going out of sync with the actual
        pipeline proto stored in the underlying MLMD execution in some cases.
    execution_id: Id of the underlying execution in MLMD.
    pipeline_uid: Unique id of the pipeline.
  """

  def __init__(self, mlmd_handle: metadata.Metadata,
               pipeline: pipeline_pb2.Pipeline, execution_id: int):
    """Constructor. Use one of the factory methods to initialize."""
    self.mlmd_handle = mlmd_handle
    # TODO(b/201294315): Fix self.pipeline going out of sync with the actual
    # pipeline proto stored in the underlying MLMD execution in some cases.
    self.pipeline = pipeline
    self.execution_id = execution_id

    # Only set within the pipeline state context.
    self._mlmd_execution_atomic_op_context = None
    self._execution: Optional[metadata_store_pb2.Execution] = None

  @classmethod
  def new(
      cls,
      mlmd_handle: metadata.Metadata,
      pipeline: pipeline_pb2.Pipeline,
      pipeline_run_metadata: Optional[Mapping[str, types.Property]] = None,
  ) -> 'PipelineState':
    """Creates a `PipelineState` object for a new pipeline.

    No active pipeline with the same pipeline uid should exist for the call to
    be successful.

    Args:
      mlmd_handle: A handle to the MLMD db.
      pipeline: IR of the pipeline.
      pipeline_run_metadata: Pipeline run metadata.

    Returns:
      A `PipelineState` object.

    Raises:
      status_lib.StatusNotOkError: If a pipeline with same UID already exists.
    """
    pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
    context = context_lib.register_context_if_not_exists(
        mlmd_handle,
        context_type_name=_ORCHESTRATOR_RESERVED_ID,
        context_name=orchestrator_context_name(pipeline_uid))

    executions = mlmd_handle.store.get_executions_by_context(context.id)
    if any(e for e in executions if execution_lib.is_execution_active(e)):
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.ALREADY_EXISTS,
          message=f'Pipeline with uid {pipeline_uid} already active.')

    exec_properties = {_PIPELINE_IR: _base64_encode_pipeline(pipeline)}
    if pipeline_run_metadata:
      exec_properties[_PIPELINE_RUN_METADATA] = json_utils.dumps(
          pipeline_run_metadata)

    execution = execution_lib.prepare_execution(
        mlmd_handle,
        _ORCHESTRATOR_EXECUTION_TYPE,
        metadata_store_pb2.Execution.NEW,
        exec_properties=exec_properties)
    if pipeline.execution_mode == pipeline_pb2.Pipeline.SYNC:
      data_types_utils.set_metadata_value(
          execution.custom_properties[_PIPELINE_RUN_ID],
          pipeline.runtime_spec.pipeline_run_id.field_value.string_value)

    execution = execution_lib.put_execution(mlmd_handle, execution, [context])
    record_state_change_time()

    return cls(
        mlmd_handle=mlmd_handle, pipeline=pipeline, execution_id=execution.id)

  @classmethod
  def load(cls, mlmd_handle: metadata.Metadata,
           pipeline_uid: task_lib.PipelineUid) -> 'PipelineState':
    """Loads pipeline state from MLMD.

    Args:
      mlmd_handle: A handle to the MLMD db.
      pipeline_uid: Uid of the pipeline state to load.

    Returns:
      A `PipelineState` object.

    Raises:
      status_lib.StatusNotOkError: With code=NOT_FOUND if no active pipeline
      with the given pipeline uid exists in MLMD. With code=INTERNAL if more
      than 1 active execution exists for given pipeline uid.
    """
    context = mlmd_handle.store.get_context_by_type_and_name(
        type_name=_ORCHESTRATOR_RESERVED_ID,
        context_name=orchestrator_context_name(pipeline_uid))
    if not context:
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.NOT_FOUND,
          message=f'No pipeline with uid {pipeline_uid} found.')
    return cls.load_from_orchestrator_context(mlmd_handle, context)

  @classmethod
  def load_from_orchestrator_context(
      cls, mlmd_handle: metadata.Metadata,
      context: metadata_store_pb2.Context) -> 'PipelineState':
    """Loads pipeline state for active pipeline under given orchestrator context.

    Args:
      mlmd_handle: A handle to the MLMD db.
      context: Pipeline context under which to find the pipeline execution.

    Returns:
      A `PipelineState` object.

    Raises:
      status_lib.StatusNotOkError: With code=NOT_FOUND if no active pipeline
      exists for the given context in MLMD. With code=INTERNAL if more than 1
      active execution exists for given pipeline uid.
    """
    pipeline_uid = pipeline_uid_from_orchestrator_context(context)
    active_execution = _get_active_execution(
        pipeline_uid, mlmd_handle.store.get_executions_by_context(context.id))
    pipeline = _get_pipeline_from_orchestrator_execution(active_execution)

    return cls(
        mlmd_handle=mlmd_handle,
        pipeline=pipeline,
        execution_id=active_execution.id)

  @property
  def pipeline_uid(self) -> task_lib.PipelineUid:
    return task_lib.PipelineUid.from_pipeline(self.pipeline)

  @property
  def pipeline_run_id(self) -> Optional[str]:
    """Returns pipeline_run_id in case of sync pipeline, `None` otherwise."""
    if self.pipeline.execution_mode == pipeline_pb2.Pipeline.SYNC:
      return self.pipeline.runtime_spec.pipeline_run_id.field_value.string_value
    return None

  def is_active(self) -> bool:
    """Returns `True` if pipeline is active."""
    self._check_context()
    return execution_lib.is_execution_active(self._execution)

  def initiate_stop(self, status: status_lib.Status) -> None:
    """Updates pipeline state to signal stopping pipeline execution."""
    self._check_context()
    data_types_utils.set_metadata_value(
        self._execution.custom_properties[_STOP_INITIATED], 1)
    data_types_utils.set_metadata_value(
        self._execution.custom_properties[_PIPELINE_STATUS_CODE],
        int(status.code))
    if status.message:
      data_types_utils.set_metadata_value(
          self._execution.custom_properties[_PIPELINE_STATUS_MSG],
          status.message)

  def initiate_update(self, updated_pipeline: pipeline_pb2.Pipeline) -> None:
    """Initiates pipeline update process."""
    self._check_context()

    if self.pipeline.execution_mode != updated_pipeline.execution_mode:
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.INVALID_ARGUMENT,
          message=('Updating execution_mode of an active pipeline is not '
                   'supported'))

    if self.pipeline.execution_mode == pipeline_pb2.Pipeline.SYNC:
      updated_pipeline_run_id = (
          updated_pipeline.runtime_spec.pipeline_run_id.field_value.string_value
      )
      if self.pipeline_run_id != updated_pipeline_run_id:
        raise status_lib.StatusNotOkError(
            code=status_lib.Code.INVALID_ARGUMENT,
            message=(f'For sync pipeline, pipeline_run_id should match; found '
                     f'mismatch: {self.pipeline_run_id} (existing) vs. '
                     f'{updated_pipeline_run_id} (updated)'))

    # TODO(b/194311197): We require that structure of the updated pipeline
    # exactly matches the original. There is scope to relax this restriction.

    def _structure(
        pipeline: pipeline_pb2.Pipeline
    ) -> List[Tuple[str, List[str], List[str]]]:
      return [(node.node_info.id, list(node.upstream_nodes),
               list(node.downstream_nodes))
              for node in get_all_pipeline_nodes(pipeline)]

    if _structure(self.pipeline) != _structure(updated_pipeline):
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.INVALID_ARGUMENT,
          message=('Updated pipeline should have the same structure as the '
                   'original.'))

    data_types_utils.set_metadata_value(
        self._execution.custom_properties[_UPDATED_PIPELINE_IR],
        _base64_encode_pipeline(updated_pipeline))

  def is_update_initiated(self) -> bool:
    self._check_context()
    return self.is_active() and self._execution.custom_properties.get(
        _UPDATED_PIPELINE_IR) is not None

  def apply_pipeline_update(self) -> None:
    """Applies pipeline update that was previously initiated."""
    self._check_context()
    updated_pipeline_ir = _get_metadata_value(
        self._execution.custom_properties.get(_UPDATED_PIPELINE_IR))
    if not updated_pipeline_ir:
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.INVALID_ARGUMENT,
          message='No updated pipeline IR to apply')
    data_types_utils.set_metadata_value(
        self._execution.properties[_PIPELINE_IR], updated_pipeline_ir)
    del self._execution.custom_properties[_UPDATED_PIPELINE_IR]
    self.pipeline = _base64_decode_pipeline(updated_pipeline_ir)

  def is_stop_initiated(self) -> bool:
    self._check_context()
    return self.stop_initiated_reason() is not None

  def stop_initiated_reason(self) -> Optional[status_lib.Status]:
    """Returns status object if stop initiated, `None` otherwise."""
    self._check_context()
    custom_properties = self._execution.custom_properties
    if _get_metadata_value(custom_properties.get(_STOP_INITIATED)) == 1:
      code = _get_metadata_value(custom_properties.get(_PIPELINE_STATUS_CODE))
      if code is None:
        code = status_lib.Code.UNKNOWN
      message = _get_metadata_value(custom_properties.get(_PIPELINE_STATUS_MSG))
      return status_lib.Status(code=code, message=message)
    else:
      return None

  @contextlib.contextmanager
  def node_state_update_context(
      self, node_uid: task_lib.NodeUid) -> Iterator[NodeState]:
    """Context manager for updating the node state."""
    self._check_context()
    if not _is_node_uid_in_pipeline(node_uid, self.pipeline):
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.INVALID_ARGUMENT,
          message=(f'Node {node_uid} does not belong to the pipeline '
                   f'{self.pipeline_uid}'))
    node_states_dict = _get_node_states_dict(self._execution)
    node_state = node_states_dict.setdefault(node_uid.node_id, NodeState())
    old_state = node_state.state
    yield node_state
    if old_state != node_state.state:
      logging.info('Changing node state: %s -> %s; node uid: %s', old_state,
                   node_state.state, node_uid)
    self._save_node_states_dict(node_states_dict)

  def get_node_state(self, node_uid: task_lib.NodeUid) -> NodeState:
    self._check_context()
    if not _is_node_uid_in_pipeline(node_uid, self.pipeline):
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.INVALID_ARGUMENT,
          message=(f'Node {node_uid} does not belong to the pipeline '
                   f'{self.pipeline_uid}'))
    node_states_dict = _get_node_states_dict(self._execution)
    return node_states_dict.get(node_uid.node_id, NodeState())

  def get_node_states_dict(self) -> Dict[task_lib.NodeUid, NodeState]:
    self._check_context()
    result = {}
    for node in get_all_pipeline_nodes(self.pipeline):
      node_uid = task_lib.NodeUid.from_pipeline_node(self.pipeline, node)
      result[node_uid] = self.get_node_state(node_uid)
    return result

  def get_pipeline_execution_state(self) -> metadata_store_pb2.Execution.State:
    """Returns state of underlying pipeline execution."""
    self._check_context()
    return self._execution.last_known_state

  def set_pipeline_execution_state(
      self, state: metadata_store_pb2.Execution.State) -> None:
    """Sets state of underlying pipeline execution."""
    self._check_context()
    if self._execution.last_known_state != state:
      logging.info(
          'Changing pipeline execution state: %s -> %s; pipeline uid: %s',
          metadata_store_pb2.Execution.State.Name(
              self._execution.last_known_state),
          metadata_store_pb2.Execution.State.Name(state), self.pipeline_uid)
      self._execution.last_known_state = state

  def get_property(self, property_key: str) -> Optional[types.Property]:
    """Returns custom property value from the pipeline execution."""
    return _get_metadata_value(
        self._execution.custom_properties.get(property_key))

  def save_property(self, property_key: str, property_value: str) -> None:
    """Saves a custom property to the pipeline execution."""
    self._check_context()
    self._execution.custom_properties[
        property_key].string_value = property_value

  def remove_property(self, property_key: str) -> None:
    """Removes a custom property of the pipeline execution if exists."""
    self._check_context()
    if self._execution.custom_properties.get(property_key):
      del self._execution.custom_properties[property_key]

  def pipeline_creation_time_secs_since_epoch(self) -> int:
    """Returns the pipeline creation time as seconds since epoch."""
    self._check_context()
    # Convert from milliseconds to seconds.
    return self._execution.create_time_since_epoch // 1000

  def get_orchestration_options(
      self) -> orchestration_options.OrchestrationOptions:
    self._check_context()
    return env.get_env().get_orchestration_options(self.pipeline)

  def _save_node_states_dict(self, node_states: Dict[str, NodeState]) -> None:
    data_types_utils.set_metadata_value(
        self._execution.custom_properties[_NODE_STATES],
        json_utils.dumps(node_states))

  def __enter__(self) -> 'PipelineState':
    mlmd_execution_atomic_op_context = mlmd_state.mlmd_execution_atomic_op(
        self.mlmd_handle, self.execution_id, record_state_change_time)
    execution = mlmd_execution_atomic_op_context.__enter__()
    self._mlmd_execution_atomic_op_context = mlmd_execution_atomic_op_context
    self._execution = execution
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    mlmd_execution_atomic_op_context = self._mlmd_execution_atomic_op_context
    self._mlmd_execution_atomic_op_context = None
    self._execution = None
    mlmd_execution_atomic_op_context.__exit__(exc_type, exc_val, exc_tb)

  def _check_context(self) -> None:
    if self._execution is None:
      raise RuntimeError(
          'Operation must be performed within the pipeline state context.')


class PipelineView:
  """Class for reading active or inactive pipeline view."""

  def __init__(self, pipeline_uid: task_lib.PipelineUid,
               context: metadata_store_pb2.Context,
               execution: metadata_store_pb2.Execution):
    self.pipeline_uid = pipeline_uid
    self.context = context
    self.execution = execution
    self._pipeline = None  # lazily set

  @classmethod
  def load_all(cls, mlmd_handle: metadata.Metadata,
               pipeline_uid: task_lib.PipelineUid) -> List['PipelineView']:
    """Loads all pipeline views from MLMD.

    Args:
      mlmd_handle: A handle to the MLMD db.
      pipeline_uid: Uid of the pipeline state to load.

    Returns:
      A list of `PipelineView` objects.

    Raises:
      status_lib.StatusNotOkError: With code=NOT_FOUND if no pipeline
      with the given pipeline uid exists in MLMD.
    """
    context = mlmd_handle.store.get_context_by_type_and_name(
        type_name=_ORCHESTRATOR_RESERVED_ID,
        context_name=orchestrator_context_name(pipeline_uid))
    if not context:
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.NOT_FOUND,
          message=f'No pipeline with uid {pipeline_uid} found.')
    executions = mlmd_handle.store.get_executions_by_context(context.id)
    return [cls(pipeline_uid, context, execution) for execution in executions]

  @classmethod
  def load(cls,
           mlmd_handle: metadata.Metadata,
           pipeline_uid: task_lib.PipelineUid,
           pipeline_run_id: Optional[str] = None) -> 'PipelineView':
    """Loads pipeline view from MLMD.

    Args:
      mlmd_handle: A handle to the MLMD db.
      pipeline_uid: Uid of the pipeline state to load.
      pipeline_run_id: Run id of the pipeline for the synchronous pipeline.

    Returns:
      A `PipelineView` object.

    Raises:
      status_lib.StatusNotOkError: With code=NOT_FOUND if no pipeline
      with the given pipeline uid exists in MLMD. With code=INTERNAL if more
      than 1 active execution exists for given pipeline uid when pipeline_run_id
      is not specified.

    """
    context = mlmd_handle.store.get_context_by_type_and_name(
        type_name=_ORCHESTRATOR_RESERVED_ID,
        context_name=orchestrator_context_name(pipeline_uid))
    if not context:
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.NOT_FOUND,
          message=f'No pipeline with uid {pipeline_uid} found.')
    executions = mlmd_handle.store.get_executions_by_context(context.id)

    if pipeline_run_id is None and executions:
      execution = _get_latest_execution(executions)
      return cls(pipeline_uid, context, execution)

    for execution in executions:
      if execution.custom_properties[
          _PIPELINE_RUN_ID].string_value == pipeline_run_id:
        return cls(pipeline_uid, context, execution)
    raise status_lib.StatusNotOkError(
        code=status_lib.Code.NOT_FOUND,
        message=f'No pipeline with run_id {pipeline_run_id} found.')

  @property
  def pipeline(self) -> pipeline_pb2.Pipeline:
    if not self._pipeline:
      self._pipeline = _get_pipeline_from_orchestrator_execution(self.execution)
    return self._pipeline

  @property
  def pipeline_run_id(self) -> str:
    if _PIPELINE_RUN_ID in self.execution.custom_properties:
      return self.execution.custom_properties[_PIPELINE_RUN_ID].string_value
    return self.pipeline.runtime_spec.pipeline_run_id.field_value.string_value

  @property
  def pipeline_status_message(self) -> str:
    if _PIPELINE_STATUS_MSG in self.execution.custom_properties:
      return self.execution.custom_properties[_PIPELINE_STATUS_MSG].string_value
    return ''

  @property
  def pipeline_run_metadata(self) -> Dict[str, types.Property]:
    pipeline_run_metadata = _get_metadata_value(
        self.execution.custom_properties.get(_PIPELINE_RUN_METADATA))
    return json_utils.loads(
        pipeline_run_metadata) if pipeline_run_metadata else {}

  def get_pipeline_run_state(self) -> run_state_pb2.RunState:
    """Returns current pipeline run state."""
    if self.execution.last_known_state in _EXECUTION_STATE_TO_RUN_STATE_MAP:
      return run_state_pb2.RunState(
          state=_EXECUTION_STATE_TO_RUN_STATE_MAP[
              self.execution.last_known_state],
          status_msg=self.pipeline_status_message)
    return run_state_pb2.RunState(
        state=run_state_pb2.RunState.UNKNOWN,
        status_msg=self.pipeline_status_message)

  def get_node_run_states(self) -> Dict[str, run_state_pb2.RunState]:
    """Returns a dict mapping node id to current run state."""
    result = {}
    node_states_dict = _get_node_states_dict(self.execution)
    for node in get_all_pipeline_nodes(self.pipeline):
      node_state = node_states_dict.get(node.node_info.id, NodeState())
      result[node.node_info.id] = run_state_pb2.RunState(
          state=_NODE_STATE_TO_RUN_STATE_MAP[node_state.state],
          status_msg=node_state.status_msg)
    return result


def get_orchestrator_contexts(
    mlmd_handle: metadata.Metadata) -> List[metadata_store_pb2.Context]:
  return mlmd_handle.store.get_contexts_by_type(_ORCHESTRATOR_RESERVED_ID)


def orchestrator_context_name(pipeline_uid: task_lib.PipelineUid) -> str:
  """Returns orchestrator reserved context name."""
  result = f'{pipeline_uid.pipeline_id}'
  if pipeline_uid.key:
    result = f'{result}:{pipeline_uid.key}'
  return result


def pipeline_uid_from_orchestrator_context(
    context: metadata_store_pb2.Context) -> task_lib.PipelineUid:
  """Returns pipeline uid from orchestrator reserved context."""
  splits = context.name.split(':')
  pipeline_id = splits[0]
  key = splits[1] if len(splits) > 1 else ''
  return task_lib.PipelineUid(pipeline_id=pipeline_id, key=key)


def get_all_pipeline_nodes(
    pipeline: pipeline_pb2.Pipeline) -> List[pipeline_pb2.PipelineNode]:
  """Returns all pipeline nodes in the given pipeline."""
  result = []
  for pipeline_or_node in pipeline.nodes:
    which = pipeline_or_node.WhichOneof('node')
    # TODO(goutham): Handle sub-pipelines.
    # TODO(goutham): Handle system nodes.
    if which == 'pipeline_node':
      result.append(pipeline_or_node.pipeline_node)
    else:
      raise NotImplementedError('Only pipeline nodes supported.')
  return result


def get_all_node_executions(
    pipeline: pipeline_pb2.Pipeline, mlmd_handle: metadata.Metadata
) -> Dict[str, List[metadata_store_pb2.Execution]]:
  """Returns the latest execution states of all pipeline nodes if present."""
  return {
      node.node_info.id: task_gen_utils.get_executions(mlmd_handle, node)
      for node in get_all_pipeline_nodes(pipeline)
  }


def _is_node_uid_in_pipeline(node_uid: task_lib.NodeUid,
                             pipeline: pipeline_pb2.Pipeline) -> bool:
  """Returns `True` if the `node_uid` belongs to the given pipeline."""
  for node in get_all_pipeline_nodes(pipeline):
    if task_lib.NodeUid.from_pipeline_node(pipeline, node) == node_uid:
      return True
  return False


def _get_metadata_value(
    value: Optional[metadata_store_pb2.Value]) -> Optional[types.Property]:
  if value is None:
    return None
  return data_types_utils.get_metadata_value(value)


def _get_pipeline_from_orchestrator_execution(
    execution: metadata_store_pb2.Execution) -> pipeline_pb2.Pipeline:
  pipeline_ir_b64 = data_types_utils.get_metadata_value(
      execution.properties[_PIPELINE_IR])
  return _base64_decode_pipeline(pipeline_ir_b64)


def _get_active_execution(
    pipeline_uid: task_lib.PipelineUid,
    executions: List[metadata_store_pb2.Execution]
) -> metadata_store_pb2.Execution:
  """gets a single active execution from the executions."""
  active_executions = [
      e for e in executions if execution_lib.is_execution_active(e)
  ]
  if not active_executions:
    raise status_lib.StatusNotOkError(
        code=status_lib.Code.NOT_FOUND,
        message=f'No active pipeline with uid {pipeline_uid} to load state.')
  if len(active_executions) > 1:
    raise status_lib.StatusNotOkError(
        code=status_lib.Code.INTERNAL,
        message=(
            f'Expected 1 but found {len(active_executions)} active pipeline '
            f'executions for pipeline uid: {pipeline_uid}'))
  return active_executions[0]


def _get_latest_execution(
    executions: List[metadata_store_pb2.Execution]
) -> metadata_store_pb2.Execution:
  """gets a single latest execution from the executions."""

  def _get_creation_time(execution):
    return execution.create_time_since_epoch

  return max(executions, key=_get_creation_time)


def _base64_encode_pipeline(pipeline: pipeline_pb2.Pipeline) -> str:
  return base64.b64encode(pipeline.SerializeToString()).decode('utf-8')


def _base64_decode_pipeline(pipeline_encoded: str) -> pipeline_pb2.Pipeline:
  result = pipeline_pb2.Pipeline()
  result.ParseFromString(base64.b64decode(pipeline_encoded))
  return result


def _get_node_states_dict(
    pipeline_execution: metadata_store_pb2.Execution) -> Dict[str, NodeState]:
  node_states_json = _get_metadata_value(
      pipeline_execution.custom_properties.get(_NODE_STATES))
  return json_utils.loads(node_states_json) if node_states_json else {}
