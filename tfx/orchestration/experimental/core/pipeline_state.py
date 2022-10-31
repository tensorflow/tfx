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
import copy
import dataclasses
import functools
import json
import os
import threading
import time
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple
import uuid

from absl import logging
import attr
from tfx import types
from tfx.dsl.io import fileio
from tfx.orchestration import data_types_utils
from tfx.orchestration import metadata
from tfx.orchestration import node_proto_view
from tfx.orchestration.experimental.core import env
from tfx.orchestration.experimental.core import event_observer
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

from google.protobuf import message
import ml_metadata as mlmd
from ml_metadata.proto import metadata_store_pb2

_ORCHESTRATOR_RESERVED_ID = '__ORCHESTRATOR__'
_PIPELINE_IR = 'pipeline_ir'
_STOP_INITIATED = 'stop_initiated'
_PIPELINE_RUN_ID = 'pipeline_run_id'
_PIPELINE_STATUS_CODE = 'pipeline_status_code'
_PIPELINE_STATUS_MSG = 'pipeline_status_msg'
_NODE_STATES = 'node_states'
# Denotes node states from previous run. Only applicable if a node is skipped
# in the partial run.
_PREVIOUS_NODE_STATES = 'previous_node_states'
_PIPELINE_RUN_METADATA = 'pipeline_run_metadata'
_UPDATED_PIPELINE_IR = 'updated_pipeline_ir'
_UPDATE_OPTIONS = 'update_options'
_ORCHESTRATOR_EXECUTION_TYPE = metadata_store_pb2.ExecutionType(
    name=_ORCHESTRATOR_RESERVED_ID,
    properties={_PIPELINE_IR: metadata_store_pb2.STRING})
_MAX_STATE_HISTORY_LEN = 10
_PIPELINE_EXEC_MODE = 'pipeline_exec_mode'
_PIPELINE_EXEC_MODE_SYNC = 'sync'
_PIPELINE_EXEC_MODE_ASYNC = 'async'

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


@dataclasses.dataclass
class StateRecord(json_utils.Jsonable):
  state: str
  status_code: Optional[int]
  update_time: float
  # TODO(b/242083811) Some status_msg have already been written into MLMD.
  # Keeping this field is for backward compatibility to avoid json failing to
  # parse existing status_msg. We can remove it once we are sure no status_msg
  # in MLMD is in use.
  status_msg: str = ''


# TODO(b/228198652): Stop using json_util.Jsonable. Before we do,
# this class MUST NOT be moved out of this module.
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
  # Node execution skipped due to condition not satisfied when pipeline has
  # conditionals.
  SKIPPED = 'skipped'
  # Node execution skipped due to partial run.
  SKIPPED_PARTIAL_RUN = 'skipped_partial_run'
  PAUSING = 'pausing'  # Pending work before state can change to PAUSED.
  PAUSED = 'paused'  # Node was paused and may be resumed in the future.
  FAILED = 'failed'  # Node execution failed due to errors.

  state: str = attr.ib(
      default=STARTED,
      validator=attr.validators.in_([
          STARTING, STARTED, STOPPING, STOPPED, RUNNING, COMPLETE, SKIPPED,
          SKIPPED_PARTIAL_RUN, PAUSING, PAUSED, FAILED
      ]),
      on_setattr=attr.setters.validate)
  status_code: Optional[int] = None
  status_msg: str = ''
  last_updated_time: float = attr.ib(factory=lambda: time.time())  # pylint:disable=unnecessary-lambda

  state_history: List[StateRecord] = attr.ib(default=attr.Factory(list))

  @property
  def status(self) -> Optional[status_lib.Status]:
    if self.status_code is not None:
      return status_lib.Status(code=self.status_code, message=self.status_msg)
    return None

  def update(self,
             state: str,
             status: Optional[status_lib.Status] = None) -> None:
    if self.state != state:
      self.state_history.append(
          StateRecord(
              state=self.state,
              status_code=self.status_code,
              update_time=self.last_updated_time))
      if len(self.state_history) > _MAX_STATE_HISTORY_LEN:
        self.state_history = self.state_history[-_MAX_STATE_HISTORY_LEN:]
      self.last_updated_time = time.time()

    self.state = state
    self.status_code = status.code if status is not None else None
    self.status_msg = status.message if status is not None else ''

  def is_startable(self) -> bool:
    """Returns True if the node can be started."""
    return self.state in set(
        [self.PAUSED, self.STOPPING, self.STOPPED, self.FAILED])

  def is_stoppable(self) -> bool:
    """Returns True if the node can be stopped."""
    return self.state in set(
        [self.STARTING, self.STARTED, self.RUNNING, self.PAUSED])

  def is_pausable(self) -> bool:
    """Returns True if the node can be stopped."""
    return self.state in set([self.STARTING, self.STARTED, self.RUNNING])

  def is_programmatically_skippable(self) -> bool:
    """Returns True if the node can be skipped via programmatic operation."""
    return self.state in set([self.STARTED, self.STOPPED])

  def is_success(self) -> bool:
    return is_node_state_success(self.state)

  def is_failure(self) -> bool:
    return is_node_state_failure(self.state)

  def to_run_state(self) -> run_state_pb2.RunState:
    """Returns this NodeState converted to a RunState."""
    status_code_value = None
    if self.status_code is not None:
      status_code_value = run_state_pb2.RunState.StatusCodeValue(
          value=self.status_code)
    return run_state_pb2.RunState(
        state=_NODE_STATE_TO_RUN_STATE_MAP[self.state],
        status_code=status_code_value,
        status_msg=self.status_msg,
        update_time=int(self.last_updated_time * 1000))

  def to_run_state_history(self) -> List[run_state_pb2.RunState]:
    run_state_history = []
    for state in self.state_history:
      run_state_history.append(
          NodeState(
              state=state.state,
              status_code=state.status_code,
              last_updated_time=state.update_time).to_run_state())
    return run_state_history

  # By default, json_utils.Jsonable serializes and deserializes objects using
  # obj.__dict__, which prevents attr.ib from populating default fields.
  # Overriding this function to ensure default fields are populated.
  @classmethod
  def from_json_dict(cls, dict_data: Dict[str, Any]) -> Any:
    """Convert from dictionary data to an object."""
    return cls(**dict_data)


def is_node_state_success(state: str) -> bool:
  return state in (NodeState.COMPLETE, NodeState.SKIPPED,
                   NodeState.SKIPPED_PARTIAL_RUN)


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
    NodeState.SKIPPED_PARTIAL_RUN: run_state_pb2.RunState.SKIPPED_PARTIAL_RUN,
    NodeState.PAUSING: run_state_pb2.RunState.UNKNOWN,
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


class _PipelineIRCodec:
  """A class for encoding / decoding pipeline IR."""

  _ORCHESTRATOR_METADATA_DIR = '.orchestrator'
  _PIPELINE_IRS_DIR = 'pipeline_irs'
  _PIPELINE_IR_URL_KEY = 'pipeline_ir_url'
  _obj = None
  _lock = threading.Lock()

  @classmethod
  def get(cls) -> '_PipelineIRCodec':
    with cls._lock:
      if not cls._obj:
        cls._obj = cls()
      return cls._obj

  @classmethod
  def testonly_reset(cls) -> None:
    """Reset global state, for tests only."""
    with cls._lock:
      cls._obj = None

  def __init__(self):
    self.base_dir = env.get_env().get_base_dir()
    if self.base_dir:
      self.pipeline_irs_dir = os.path.join(self.base_dir,
                                           self._ORCHESTRATOR_METADATA_DIR,
                                           self._PIPELINE_IRS_DIR)
      fileio.makedirs(self.pipeline_irs_dir)
    else:
      self.pipeline_irs_dir = None

  def encode(self, pipeline: pipeline_pb2.Pipeline) -> str:
    """Encodes pipeline IR."""
    # Attempt to store as a base64 encoded string. If base_dir is provided
    # and the length is too large, store the IR on disk and retain the URL.
    # TODO(b/248786921): Always store pipeline IR to base_dir once the
    # accessibility issue is resolved.
    pipeline_encoded = _base64_encode(pipeline)
    max_mlmd_str_value_len = env.get_env().max_mlmd_str_value_length()
    if self.base_dir and max_mlmd_str_value_len is not None and len(
        pipeline_encoded) > max_mlmd_str_value_len:
      pipeline_id = task_lib.PipelineUid.from_pipeline(pipeline).pipeline_id
      pipeline_url = os.path.join(self.pipeline_irs_dir,
                                  f'{pipeline_id}_{uuid.uuid4()}.pb')
      with fileio.open(pipeline_url, 'wb') as file:
        file.write(pipeline.SerializeToString())
      pipeline_encoded = json.dumps({self._PIPELINE_IR_URL_KEY: pipeline_url})
    return pipeline_encoded

  def decode(self, value: str) -> pipeline_pb2.Pipeline:
    """Decodes pipeline IR."""
    # Attempt to load as JSON. If it fails, fallback to decoding it as a base64
    # encoded string for backward compatibility.
    try:
      pipeline_encoded = json.loads(value)
      with fileio.open(pipeline_encoded[self._PIPELINE_IR_URL_KEY],
                       'rb') as file:
        return pipeline_pb2.Pipeline.FromString(file.read())
    except json.JSONDecodeError:
      return _base64_decode_pipeline(value)


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
    execution: The underlying execution in MLMD.
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
    self._on_commit_callbacks: List[Callable[[], None]] = []

  @classmethod
  def new(
      cls,
      mlmd_handle: metadata.Metadata,
      pipeline: pipeline_pb2.Pipeline,
      pipeline_run_metadata: Optional[Mapping[str, types.Property]] = None,
      reused_pipeline_view: Optional['PipelineView'] = None,
  ) -> 'PipelineState':
    """Creates a `PipelineState` object for a new pipeline.

    No active pipeline with the same pipeline uid should exist for the call to
    be successful.

    Args:
      mlmd_handle: A handle to the MLMD db.
      pipeline: IR of the pipeline.
      pipeline_run_metadata: Pipeline run metadata.
      reused_pipeline_view: PipelineView of the previous pipeline reused for a
        partial run.

    Returns:
      A `PipelineState` object.

    Raises:
      status_lib.StatusNotOkError: If a pipeline with same UID already exists.
    """
    pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
    context = context_lib.register_context_if_not_exists(
        mlmd_handle,
        context_type_name=_ORCHESTRATOR_RESERVED_ID,
        context_name=pipeline_uid.pipeline_id)

    executions = mlmd_handle.store.get_executions_by_context(context.id)

    def _any_active_pipeline_with_uid(
        executions: List[metadata_store_pb2.Execution],
        pipeline_uid: task_lib.PipelineUid) -> bool:
      if pipeline_uid.pipeline_run_id is None:
        return any(
            e for e in executions if execution_lib.is_execution_active(e))
      else:
        return any(
            e for e in executions if execution_lib.is_execution_active(e) and
            _get_metadata_value(e.custom_properties.get(
                _PIPELINE_RUN_ID)) == pipeline_uid.pipeline_run_id)

    if _any_active_pipeline_with_uid(executions, pipeline_uid):
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.ALREADY_EXISTS,
          message=f'Pipeline with uid {pipeline_uid} already active.')

    # TODO(b/254161062): Consider disallowing pipeline exec mode change for the
    # same pipeline id.
    if pipeline.execution_mode == pipeline_pb2.Pipeline.SYNC:
      pipeline_exec_mode = _PIPELINE_EXEC_MODE_SYNC
    elif pipeline.execution_mode == pipeline_pb2.Pipeline.ASYNC:
      pipeline_exec_mode = _PIPELINE_EXEC_MODE_ASYNC
    else:
      raise ValueError('Expected pipeline execution mode to be SYNC or ASYNC')

    exec_properties = {
        _PIPELINE_IR: _PipelineIRCodec.get().encode(pipeline),
        _PIPELINE_EXEC_MODE: pipeline_exec_mode
    }
    if pipeline_run_metadata:
      exec_properties[_PIPELINE_RUN_METADATA] = json_utils.dumps(
          pipeline_run_metadata)

    execution = execution_lib.prepare_execution(
        mlmd_handle,
        _ORCHESTRATOR_EXECUTION_TYPE,
        metadata_store_pb2.Execution.NEW,
        exec_properties=exec_properties,
        execution_name=str(uuid.uuid4()))
    if pipeline.execution_mode == pipeline_pb2.Pipeline.SYNC:
      data_types_utils.set_metadata_value(
          execution.custom_properties[_PIPELINE_RUN_ID],
          pipeline.runtime_spec.pipeline_run_id.field_value.string_value)
      _save_skipped_node_states(pipeline, reused_pipeline_view, execution)
    execution = execution_lib.put_execution(mlmd_handle, execution, [context])
    pipeline_state = cls(
        mlmd_handle=mlmd_handle, pipeline=pipeline, execution_id=execution.id)
    event_observer.notify(
        event_observer.PipelineStarted(
            pipeline_id=pipeline_uid.pipeline_id,
            pipeline_state=pipeline_state))
    record_state_change_time()
    return pipeline_state

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
    context = _get_orchestrator_context(mlmd_handle, pipeline_uid.pipeline_id)
    uids_and_states = cls._load_from_context(mlmd_handle, context, pipeline_uid)
    if not uids_and_states:
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.NOT_FOUND,
          message=f'No active pipeline with uid {pipeline_uid} to load state.')
    if len(uids_and_states) > 1:
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.INTERNAL,
          message=(
              f'Expected 1 but found {len(uids_and_states)} active pipelines '
              f'for pipeline uid: {pipeline_uid}'))
    return uids_and_states[0][1]

  @classmethod
  def load_all_active(cls,
                      mlmd_handle: metadata.Metadata) -> List['PipelineState']:
    """Loads all active pipeline states.

    Args:
      mlmd_handle: A handle to the MLMD db.

    Returns:
      List of `PipelineState` objects for all active pipelines.

    Raises:
      status_lib.StatusNotOkError: With code=INTERNAL if more than one active
      pipeline are found with the same pipeline uid.
    """
    contexts = get_orchestrator_contexts(mlmd_handle)
    active_pipeline_uids = set()
    result = []
    for context in contexts:
      uids_and_states = cls._load_from_context(mlmd_handle, context)
      for pipeline_uid, pipeline_state in uids_and_states:
        if pipeline_uid in active_pipeline_uids:
          raise status_lib.StatusNotOkError(
              code=status_lib.Code.INTERNAL,
              message=(
                  f'Found more than 1 active pipeline for pipeline uid: {pipeline_uid}'
              ))
        active_pipeline_uids.add(pipeline_uid)
        result.append(pipeline_state)
    return result

  @classmethod
  def _load_from_context(
      cls,
      mlmd_handle: metadata.Metadata,
      context: metadata_store_pb2.Context,
      matching_pipeline_uid: Optional[task_lib.PipelineUid] = None
  ) -> List[Tuple[task_lib.PipelineUid, 'PipelineState']]:
    """Loads active pipeline states associated with given orchestrator context.

    Args:
      mlmd_handle: A handle to the MLMD db.
      context: Orchestrator context.
      matching_pipeline_uid: If provided, returns only pipeline with matching
        pipeline_uid.

    Returns:
      List of active pipeline states.
    """
    pipeline_id = pipeline_id_from_orchestrator_context(context)
    # TODO(b/254578300): Use filter_query and load only the relevant executions.
    executions = mlmd_handle.store.get_executions_by_context(context.id)
    active_executions = [
        e for e in executions if execution_lib.is_execution_active(e)
    ]
    result = []
    for execution in active_executions:
      pipeline_uid = task_lib.PipelineUid.from_pipeline_id_and_run_id(
          pipeline_id,
          _get_metadata_value(
              execution.custom_properties.get(_PIPELINE_RUN_ID)))
      if matching_pipeline_uid and pipeline_uid != matching_pipeline_uid:
        continue
      pipeline = _get_pipeline_from_orchestrator_execution(execution)
      result.append(
          (pipeline_uid, PipelineState(mlmd_handle, pipeline, execution.id)))
    return result

  @property
  def pipeline_uid(self) -> task_lib.PipelineUid:
    return task_lib.PipelineUid.from_pipeline(self.pipeline)

  @property
  def pipeline_run_id(self) -> Optional[str]:
    """Returns pipeline_run_id in case of sync pipeline, `None` otherwise."""
    if self.pipeline.execution_mode == pipeline_pb2.Pipeline.SYNC:
      return self.pipeline.runtime_spec.pipeline_run_id.field_value.string_value
    return None

  @property
  def execution(self) -> metadata_store_pb2.Execution:
    self._check_context()
    return self._execution

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

  def initiate_update(
      self,
      updated_pipeline: pipeline_pb2.Pipeline,
      update_options: pipeline_pb2.UpdateOptions,
  ) -> None:
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
               list(node.downstream_nodes)) for node in get_all_nodes(pipeline)]

    if _structure(self.pipeline) != _structure(updated_pipeline):
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.INVALID_ARGUMENT,
          message=('Updated pipeline should have the same structure as the '
                   'original.'))

    data_types_utils.set_metadata_value(
        self._execution.custom_properties[_UPDATED_PIPELINE_IR],
        _PipelineIRCodec.get().encode(updated_pipeline))
    data_types_utils.set_metadata_value(
        self._execution.custom_properties[_UPDATE_OPTIONS],
        _base64_encode(update_options))

  def is_update_initiated(self) -> bool:
    self._check_context()
    return self.is_active() and self._execution.custom_properties.get(
        _UPDATED_PIPELINE_IR) is not None

  def get_update_options(self) -> pipeline_pb2.UpdateOptions:
    """Gets pipeline update option that was previously configured."""
    self._check_context()
    update_options = self._execution.custom_properties.get(_UPDATE_OPTIONS)
    if update_options is None:
      logging.warning(
          'pipeline execution missing expected custom property %s, '
          'defaulting to UpdateOptions(reload_policy=ALL)', _UPDATE_OPTIONS)
      return pipeline_pb2.UpdateOptions(
          reload_policy=pipeline_pb2.UpdateOptions.ReloadPolicy.ALL)
    return _base64_decode_update_options(_get_metadata_value(update_options))

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
    del self._execution.custom_properties[_UPDATE_OPTIONS]
    self.pipeline = _PipelineIRCodec.get().decode(updated_pipeline_ir)

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
      msg = _get_metadata_value(custom_properties.get(_PIPELINE_STATUS_MSG))
      return status_lib.Status(code=code, message=msg)
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
    old_state = copy.deepcopy(node_state)
    yield node_state
    if old_state.state != node_state.state:
      self._on_commit_callbacks.extend([
          functools.partial(_log_node_state_change, old_state.state,
                            node_state.state, node_uid),
          functools.partial(_notify_node_state_change,
                            copy.deepcopy(self._execution), node_uid,
                            self.pipeline_run_id, old_state, node_state)
      ])
    _save_node_states_dict(self._execution, node_states_dict)

  def get_node_state(self,
                     node_uid: task_lib.NodeUid,
                     state_type: Optional[str] = _NODE_STATES) -> NodeState:
    """Gets node state of a specified node."""
    self._check_context()
    if not _is_node_uid_in_pipeline(node_uid, self.pipeline):
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.INVALID_ARGUMENT,
          message=(f'Node {node_uid} does not belong to the pipeline '
                   f'{self.pipeline_uid}'))
    node_states_dict = _get_node_states_dict(self._execution, state_type)
    return node_states_dict.get(node_uid.node_id, NodeState())

  def get_node_states_dict(self) -> Dict[task_lib.NodeUid, NodeState]:
    """Gets all node states of the pipeline."""
    self._check_context()
    node_states_dict = _get_node_states_dict(self._execution, _NODE_STATES)
    result = {}
    for node in get_all_nodes(self.pipeline):
      node_uid = task_lib.NodeUid.from_node(self.pipeline, node)
      result[node_uid] = node_states_dict.get(node_uid.node_id, NodeState())
    return result

  def get_previous_node_states_dict(self) -> Dict[task_lib.NodeUid, NodeState]:
    """Gets all node states of the pipeline from previous run."""
    self._check_context()
    node_states_dict = _get_node_states_dict(self._execution,
                                             _PREVIOUS_NODE_STATES)
    result = {}
    for node in get_all_nodes(self.pipeline):
      node_uid = task_lib.NodeUid.from_node(self.pipeline, node)
      if node_uid.node_id not in node_states_dict:
        continue
      result[node_uid] = node_states_dict[node_uid.node_id]
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
      self._on_commit_callbacks.append(
          functools.partial(_log_pipeline_execution_state_change,
                            self._execution.last_known_state, state,
                            self.pipeline_uid))
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

  def __enter__(self) -> 'PipelineState':

    def _run_on_commit_callbacks():
      record_state_change_time()
      for on_commit_cb in self._on_commit_callbacks:
        on_commit_cb()

    mlmd_execution_atomic_op_context = mlmd_state.mlmd_execution_atomic_op(
        self.mlmd_handle, self.execution_id, _run_on_commit_callbacks)
    execution = mlmd_execution_atomic_op_context.__enter__()
    self._mlmd_execution_atomic_op_context = mlmd_execution_atomic_op_context
    self._execution = execution
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    mlmd_execution_atomic_op_context = self._mlmd_execution_atomic_op_context
    self._mlmd_execution_atomic_op_context = None
    self._execution = None
    try:
      mlmd_execution_atomic_op_context.__exit__(exc_type, exc_val, exc_tb)
    finally:
      self._on_commit_callbacks.clear()

  def _check_context(self) -> None:
    if self._execution is None:
      raise RuntimeError(
          'Operation must be performed within the pipeline state context.')


class PipelineView:
  """Class for reading active or inactive pipeline view."""

  def __init__(self, pipeline_id: str, context: metadata_store_pb2.Context,
               execution: metadata_store_pb2.Execution):
    self.pipeline_id = pipeline_id
    self.context = context
    self.execution = execution
    self._pipeline = None  # lazily set

  @classmethod
  def load_all(cls, mlmd_handle: metadata.Metadata, pipeline_id: str,
               **kwargs) -> List['PipelineView']:
    """Loads all pipeline views from MLMD.

    Args:
      mlmd_handle: A handle to the MLMD db.
      pipeline_id: Id of the pipeline state to load.
      **kwargs: Extra option to pass into mlmd store functions.

    Returns:
      A list of `PipelineView` objects.

    Raises:
      status_lib.StatusNotOkError: With code=NOT_FOUND if no pipeline
      with the given pipeline uid exists in MLMD.
    """
    context = _get_orchestrator_context(mlmd_handle, pipeline_id, **kwargs)
    list_options = mlmd.ListOptions(
        order_by=mlmd.OrderByField.CREATE_TIME, is_asc=True)
    executions = mlmd_handle.store.get_executions_by_context(
        context.id, list_options=list_options, **kwargs)
    return [cls(pipeline_id, context, execution) for execution in executions]

  @classmethod
  def load(cls,
           mlmd_handle: metadata.Metadata,
           pipeline_id: str,
           pipeline_run_id: Optional[str] = None,
           **kwargs) -> 'PipelineView':
    """Loads pipeline view from MLMD.

    Args:
      mlmd_handle: A handle to the MLMD db.
      pipeline_id: Id of the pipeline state to load.
      pipeline_run_id: Run id of the pipeline for the synchronous pipeline.
      **kwargs: Extra option to pass into mlmd store functions.

    Returns:
      A `PipelineView` object.

    Raises:
      status_lib.StatusNotOkError: With code=NOT_FOUND if no pipeline
      with the given pipeline uid exists in MLMD.
    """
    context = _get_orchestrator_context(mlmd_handle, pipeline_id, **kwargs)
    executions = mlmd_handle.store.get_executions_by_context(
        context.id, **kwargs)

    if pipeline_run_id is None and executions:
      execution = _get_latest_execution(executions)
      return cls(pipeline_id, context, execution)

    for execution in executions:
      if execution.custom_properties[
          _PIPELINE_RUN_ID].string_value == pipeline_run_id:
        return cls(pipeline_id, context, execution)
    raise status_lib.StatusNotOkError(
        code=status_lib.Code.NOT_FOUND,
        message=f'No pipeline with run_id {pipeline_run_id} found.')

  @property
  def pipeline(self) -> pipeline_pb2.Pipeline:
    if not self._pipeline:
      self._pipeline = _get_pipeline_from_orchestrator_execution(self.execution)
    return self._pipeline

  @property
  def pipeline_execution_mode(self) -> pipeline_pb2.Pipeline.ExecutionMode:
    return _retrieve_pipeline_exec_mode(self.execution)

  @property
  def pipeline_run_id(self) -> str:
    if _PIPELINE_RUN_ID in self.execution.custom_properties:
      return self.execution.custom_properties[_PIPELINE_RUN_ID].string_value
    return self.pipeline.runtime_spec.pipeline_run_id.field_value.string_value

  @property
  def pipeline_status_code(
      self) -> Optional[run_state_pb2.RunState.StatusCodeValue]:
    if _PIPELINE_STATUS_CODE in self.execution.custom_properties:
      return run_state_pb2.RunState.StatusCodeValue(
          value=self.execution.custom_properties[_PIPELINE_STATUS_CODE]
          .int_value)
    return None

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
    state = run_state_pb2.RunState.UNKNOWN
    if self.execution.last_known_state in _EXECUTION_STATE_TO_RUN_STATE_MAP:
      state = _EXECUTION_STATE_TO_RUN_STATE_MAP[self.execution.last_known_state]
    return run_state_pb2.RunState(
        state=state,
        status_code=self.pipeline_status_code,
        status_msg=self.pipeline_status_message,
        update_time=self.execution.last_update_time_since_epoch)

  def get_node_run_states(self) -> Dict[str, run_state_pb2.RunState]:
    """Returns a dict mapping node id to current run state."""
    result = {}
    node_states_dict = _get_node_states_dict(self.execution, _NODE_STATES)
    for node in get_all_nodes(self.pipeline):
      node_state = node_states_dict.get(node.node_info.id, NodeState())
      result[node.node_info.id] = node_state.to_run_state()
    return result

  def get_node_run_states_history(
      self) -> Dict[str, List[run_state_pb2.RunState]]:
    """Returns the history of node run states and timestamps."""
    node_states_dict = _get_node_states_dict(self.execution, _NODE_STATES)
    result = {}
    for node in get_all_nodes(self.pipeline):
      node_state = node_states_dict.get(node.node_info.id, NodeState())
      result[node.node_info.id] = node_state.to_run_state_history()
    return result

  def get_previous_node_run_states(self) -> Dict[str, run_state_pb2.RunState]:
    """Returns a dict mapping node id to previous run state."""
    result = {}
    node_states_dict = _get_node_states_dict(self.execution,
                                             _PREVIOUS_NODE_STATES)
    for node in get_all_nodes(self.pipeline):
      if node.node_info.id not in node_states_dict:
        continue
      node_state = node_states_dict[node.node_info.id]
      result[node.node_info.id] = node_state.to_run_state()
    return result

  def get_previous_node_run_states_history(
      self) -> Dict[str, List[run_state_pb2.RunState]]:
    """Returns a dict mapping node id to previous run state and timestamps."""
    prev_node_states_dict = _get_node_states_dict(self.execution,
                                                  _PREVIOUS_NODE_STATES)
    result = {}
    for node in get_all_nodes(self.pipeline):
      if node.node_info.id not in prev_node_states_dict:
        continue
      node_state = prev_node_states_dict[node.node_info.id]
      result[node.node_info.id] = node_state.to_run_state_history()
    return result

  def get_property(self, property_key: str) -> Optional[types.Property]:
    """Returns custom property value from the pipeline execution."""
    return _get_metadata_value(
        self.execution.custom_properties.get(property_key))

  def get_node_states_dict(self) -> Dict[str, NodeState]:
    """Returns a dict mapping node id to node state."""
    result = {}
    node_states_dict = _get_node_states_dict(self.execution, _NODE_STATES)
    for node in get_all_nodes(self.pipeline):
      result[node.node_info.id] = node_states_dict.get(node.node_info.id,
                                                       NodeState())
    return result

  def get_previous_node_states_dict(self) -> Dict[str, NodeState]:
    """Returns a dict mapping node id to node state in previous run."""
    result = {}
    node_states_dict = _get_node_states_dict(self.execution,
                                             _PREVIOUS_NODE_STATES)
    for node in get_all_nodes(self.pipeline):
      if node.node_info.id not in node_states_dict:
        continue
      result[node.node_info.id] = node_states_dict[node.node_info.id]
    return result


def get_orchestrator_contexts(mlmd_handle: metadata.Metadata,
                              **kwargs) -> List[metadata_store_pb2.Context]:
  """Returns all of the orchestrator contexts."""
  return mlmd_handle.store.get_contexts_by_type(_ORCHESTRATOR_RESERVED_ID,
                                                **kwargs)


def pipeline_id_from_orchestrator_context(
    context: metadata_store_pb2.Context) -> str:
  """Returns pipeline id from orchestrator reserved context."""
  return context.name


def get_all_nodes(
    pipeline: pipeline_pb2.Pipeline) -> List[node_proto_view.NodeProtoView]:
  """Returns the views of nodes or inner pipelines in the given pipeline."""
  # TODO(goutham): Handle system nodes.
  return [
      node_proto_view.get_view(pipeline_or_node)
      for pipeline_or_node in pipeline.nodes
  ]


def get_all_node_executions(
    pipeline: pipeline_pb2.Pipeline, mlmd_handle: metadata.Metadata
) -> Dict[str, List[metadata_store_pb2.Execution]]:
  """Returns all executions of all pipeline nodes if present."""
  return {
      node.node_info.id: task_gen_utils.get_executions(mlmd_handle, node)
      for node in get_all_nodes(pipeline)
  }


def get_all_node_artifacts(
    pipeline: pipeline_pb2.Pipeline, mlmd_handle: metadata.Metadata
) -> Dict[str, Dict[int, Dict[str, List[metadata_store_pb2.Artifact]]]]:
  """Returns all output artifacts of all pipeline nodes if present.

  Args:
    pipeline: Pipeline proto associated with a `PipelineState` object.
    mlmd_handle: Handle to MLMD db.

  Returns:
    Dict of node id to Dict of execution id to Dict of key to output artifact
    list.
  """
  executions_dict = get_all_node_executions(pipeline, mlmd_handle)
  result = {}
  for node_id, executions in executions_dict.items():
    node_artifacts = {}
    for execution in executions:
      execution_artifacts = {}
      for key, artifacts in execution_lib.get_artifacts_dict(
          mlmd_handle, execution.id, [
              metadata_store_pb2.Event.OUTPUT,
              metadata_store_pb2.Event.DECLARED_OUTPUT
          ]).items():
        execution_artifacts[key] = [
            artifact.mlmd_artifact for artifact in artifacts
        ]
      node_artifacts[execution.id] = execution_artifacts
    result[node_id] = node_artifacts
  return result


def _is_node_uid_in_pipeline(node_uid: task_lib.NodeUid,
                             pipeline: pipeline_pb2.Pipeline) -> bool:
  """Returns `True` if the `node_uid` belongs to the given pipeline."""
  for node in get_all_nodes(pipeline):
    if task_lib.NodeUid.from_node(pipeline, node) == node_uid:
      return True
  return False


def _get_metadata_value(
    value: Optional[metadata_store_pb2.Value]) -> Optional[types.Property]:
  if value is None:
    return None
  return data_types_utils.get_metadata_value(value)


def _get_pipeline_from_orchestrator_execution(
    execution: metadata_store_pb2.Execution) -> pipeline_pb2.Pipeline:
  pipeline_ir = data_types_utils.get_metadata_value(
      execution.properties[_PIPELINE_IR])
  return _PipelineIRCodec.get().decode(pipeline_ir)


def _get_latest_execution(
    executions: List[metadata_store_pb2.Execution]
) -> metadata_store_pb2.Execution:
  """gets a single latest execution from the executions."""

  def _get_creation_time(execution):
    return execution.create_time_since_epoch

  return max(executions, key=_get_creation_time)


def _get_orchestrator_context(mlmd_handle: metadata.Metadata, pipeline_id: str,
                              **kwargs) -> metadata_store_pb2.Context:
  """Returns the orchestrator context of a particular pipeline."""
  context = mlmd_handle.store.get_context_by_type_and_name(
      type_name=_ORCHESTRATOR_RESERVED_ID, context_name=pipeline_id, **kwargs)
  if not context:
    raise status_lib.StatusNotOkError(
        code=status_lib.Code.NOT_FOUND,
        message=f'No pipeline with id {pipeline_id} found.')
  return context


def _base64_encode(msg: message.Message) -> str:
  return base64.b64encode(msg.SerializeToString()).decode('utf-8')


def _base64_decode_pipeline(pipeline_encoded: str) -> pipeline_pb2.Pipeline:
  result = pipeline_pb2.Pipeline()
  result.ParseFromString(base64.b64decode(pipeline_encoded))
  return result


def _base64_decode_update_options(
    update_options_encoded: str) -> pipeline_pb2.UpdateOptions:
  result = pipeline_pb2.UpdateOptions()
  result.ParseFromString(base64.b64decode(update_options_encoded))
  return result


def _get_node_states_dict(
    pipeline_execution: metadata_store_pb2.Execution,
    state_type: Optional[str] = _NODE_STATES) -> Dict[str, NodeState]:
  """Gets node states dict from pipeline execution with specified type."""
  if state_type not in [_NODE_STATES, _PREVIOUS_NODE_STATES]:
    raise status_lib.StatusNotOkError(
        code=status_lib.Code.INVALID_ARGUMENT,
        message=f'Expected state_type is {_NODE_STATES} or {_PREVIOUS_NODE_STATES}, got {state_type}.'
    )
  node_states_json = _get_metadata_value(
      pipeline_execution.custom_properties.get(state_type))
  return json_utils.loads(node_states_json) if node_states_json else {}


def _save_node_states_dict(pipeline_execution: metadata_store_pb2.Execution,
                           node_states: Dict[str, NodeState],
                           state_type: Optional[str] = _NODE_STATES) -> None:
  """Saves node states dict to pipeline execution with specified type."""
  data_types_utils.set_metadata_value(
      pipeline_execution.custom_properties[state_type],
      json_utils.dumps(node_states))


def _save_skipped_node_states(pipeline: pipeline_pb2.Pipeline,
                              reused_pipeline_view: PipelineView,
                              execution: metadata_store_pb2.Execution) -> None:
  """Records (previous) node states for nodes that are skipped in partial run.
  """
  # Set the node state to SKIPPED_PARTIAL_RUN for any nodes that are marked
  # to be skipped in a partial pipeline run.
  node_states_dict = {}
  previous_node_states_dict = {}
  reused_pipeline_node_states_dict = reused_pipeline_view.get_node_states_dict(
  ) if reused_pipeline_view else {}
  reused_pipeline_previous_node_states_dict = reused_pipeline_view.get_previous_node_states_dict(
  ) if reused_pipeline_view else {}
  for node in get_all_nodes(pipeline):
    node_id = node.node_info.id
    if node.execution_options.HasField('skip'):
      logging.info('Node %s is skipped in this partial run.', node_id)
      node_states_dict[node_id] = NodeState(state=NodeState.SKIPPED_PARTIAL_RUN)
      if node_id in reused_pipeline_node_states_dict:
        # Indicates a node's in any base run when skipped. If a user makes
        # a chain of partial runs, we record the latest time when the
        # skipped node has a different state.
        reused_node_state = reused_pipeline_node_states_dict[node_id]
        if reused_node_state.state == NodeState.SKIPPED_PARTIAL_RUN:
          previous_node_states_dict[
              node_id] = reused_pipeline_previous_node_states_dict.get(
                  node_id, NodeState())
        else:
          previous_node_states_dict[node_id] = reused_node_state
  if node_states_dict:
    _save_node_states_dict(execution, node_states_dict, _NODE_STATES)
  if previous_node_states_dict:
    _save_node_states_dict(execution, previous_node_states_dict,
                           _PREVIOUS_NODE_STATES)


def _retrieve_pipeline_exec_mode(
    execution: metadata_store_pb2.Execution
) -> pipeline_pb2.Pipeline.ExecutionMode:
  """Returns pipeline execution mode given pipeline-level execution."""
  pipeline_exec_mode = _get_metadata_value(
      execution.custom_properties.get(_PIPELINE_EXEC_MODE))
  if pipeline_exec_mode is None:
    # Retrieve execution mode from pipeline IR for backward compatibility (this
    # is more expensive and requires parsing the proto).
    return _get_pipeline_from_orchestrator_execution(execution).execution_mode
  elif pipeline_exec_mode == _PIPELINE_EXEC_MODE_SYNC:
    return pipeline_pb2.Pipeline.SYNC
  elif pipeline_exec_mode == _PIPELINE_EXEC_MODE_ASYNC:
    return pipeline_pb2.Pipeline.ASYNC
  else:
    raise RuntimeError(
        f'Unable to determine pipeline execution mode from pipeline execution {execution}'
    )


def _log_pipeline_execution_state_change(
    old_state: metadata_store_pb2.Execution.State,
    new_state: metadata_store_pb2.Execution.State,
    pipeline_uid: task_lib.PipelineUid) -> None:
  logging.info('Changed pipeline execution state: %s -> %s; pipeline uid: %s',
               metadata_store_pb2.Execution.State.Name(old_state),
               metadata_store_pb2.Execution.State.Name(new_state), pipeline_uid)


def _log_node_state_change(old_state: str, new_state: str,
                           node_uid: task_lib.NodeUid) -> None:
  logging.info('Changed node state: %s -> %s; node uid: %s', old_state,
               new_state, node_uid)


def _notify_node_state_change(execution: metadata_store_pb2.Execution,
                              node_uid: task_lib.NodeUid, pipeline_run_id: str,
                              old_state: NodeState,
                              new_state: NodeState) -> None:
  event_observer.notify(
      event_observer.NodeStateChange(
          execution=execution,
          pipeline_id=node_uid.pipeline_uid.pipeline_id,
          pipeline_run=pipeline_run_id,
          node_id=node_uid.node_id,
          old_state=old_state,
          new_state=new_state))
