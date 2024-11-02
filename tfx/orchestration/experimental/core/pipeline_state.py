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
import collections
import contextlib
import copy
import dataclasses
import functools
import threading
import time
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Set, Tuple, cast
import uuid

from absl import logging
import attr
from tfx import types
from tfx.orchestration import data_types_utils
from tfx.orchestration import metadata
from tfx.orchestration import node_proto_view
from tfx.orchestration.experimental.core import env
from tfx.orchestration.experimental.core import event_observer
from tfx.orchestration.experimental.core import mlmd_state
from tfx.orchestration.experimental.core import orchestration_options
from tfx.orchestration.experimental.core import pipeline_ir_codec
from tfx.utils import metrics_utils
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import task_gen_utils
from tfx.orchestration.portable.mlmd import context_lib
from tfx.orchestration.portable.mlmd import execution_lib
from tfx.proto.orchestration import metadata_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.proto.orchestration import run_state_pb2
from tfx.utils import json_utils
from tfx.utils import status as status_lib

from tfx.utils import telemetry_utils
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
  backfill_token: str
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
  FAILED = 'failed'  # Node execution failed due to errors.

  state: str = attr.ib(
      default=STARTED,
      validator=attr.validators.in_([
          STARTED,
          STOPPING,
          STOPPED,
          RUNNING,
          COMPLETE,
          SKIPPED,
          SKIPPED_PARTIAL_RUN,
          FAILED,
      ]),
      on_setattr=attr.setters.validate,
  )
  backfill_token: str = ''
  status_code: Optional[int] = None
  status_msg: str = ''
  last_updated_time: float = attr.ib(factory=lambda: time.time())  # pylint:disable=unnecessary-lambda

  state_history: List[StateRecord] = attr.ib(default=attr.Factory(list))

  @property
  def status(self) -> Optional[status_lib.Status]:
    if self.status_code is not None:
      return status_lib.Status(code=self.status_code, message=self.status_msg)
    return None

  def update(
      self,
      state: str,
      status: Optional[status_lib.Status] = None,
      backfill_token: str = '',
  ) -> None:
    if self.state != state:
      self.state_history.append(
          StateRecord(
              state=self.state,
              backfill_token=self.backfill_token,
              status_code=self.status_code,
              update_time=self.last_updated_time,
          )
      )
      if len(self.state_history) > _MAX_STATE_HISTORY_LEN:
        self.state_history = self.state_history[-_MAX_STATE_HISTORY_LEN:]
      self.last_updated_time = time.time()

    self.state = state
    self.backfill_token = backfill_token
    self.status_code = status.code if status is not None else None
    self.status_msg = (status.message or '') if status is not None else ''

  def is_startable(self) -> bool:
    """Returns True if the node can be started."""
    return self.state in set([self.STOPPING, self.STOPPED, self.FAILED])

  def is_stoppable(self) -> bool:
    """Returns True if the node can be stopped."""
    return self.state in set([self.STARTED, self.RUNNING])

  def is_backfillable(self) -> bool:
    """Returns True if the node can be backfilled."""
    return self.state in set([self.STOPPED, self.FAILED])

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
        state=_NODE_STATE_TO_RUN_STATE_MAP.get(
            self.state, run_state_pb2.RunState.UNKNOWN
        ),
        status_code=status_code_value,
        status_msg=self.status_msg,
        update_time=int(self.last_updated_time * 1000),
    )

  def to_run_state_history(self) -> List[run_state_pb2.RunState]:
    run_state_history = []
    for state in self.state_history:
      # STARTING, PAUSING and PAUSED has been deprecated but may still be
      # present in state_history.
      if (
          state.state == 'starting'
          or state.state == 'pausing'
          or state.state == 'paused'
      ):
        continue
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

  def latest_predicate_time_s(self, predicate: Callable[[StateRecord], bool],
                              include_current_state: bool) -> Optional[int]:
    """Returns the latest time the StateRecord satisfies the given predicate.

    Args:
      predicate: Predicate that takes the state string.
      include_current_state: Whether to include the current node state when
        checking the node state history (the node state history doesn't include
        the current node state).

    Returns:
      The latest time (in the state history) the StateRecord satisfies the given
      predicate, or None if the predicate is never satisfied.
    """
    if include_current_state:
      current_record = StateRecord(
          state=self.state,
          backfill_token=self.backfill_token,
          status_code=self.status_code,
          update_time=self.last_updated_time,
      )
      if predicate(current_record):
        return int(current_record.update_time)

    for s in reversed(self.state_history):
      if predicate(s):
        return int(s.update_time)
    return None

  def latest_running_time_s(self) -> Optional[int]:
    """Returns the latest time the node entered a RUNNING state.

    Returns:
      The latest time (in the state history) the node entered a RUNNING
      state, or None if the node never entered a RUNNING state.
    """
    return self.latest_predicate_time_s(
        lambda s: is_node_state_running(s.state), include_current_state=True)


class _NodeStatesProxy:
  """Proxy for reading and updating deserialized NodeState dicts from Execution.

  This proxy contains an internal write-back cache. Changes are not saved back
  to the `Execution` until `save()` is called; cache would not be updated if
  changes were made outside of the proxy, either. This is primarily used to
  reduce JSON serialization/deserialization overhead for getting node state
  execution property from pipeline execution.
  """

  def __init__(self, execution: metadata_store_pb2.Execution):
    self._custom_properties = execution.custom_properties
    self._deserialized_cache: Dict[str, Dict[str, NodeState]] = {}
    self._changed_state_types: Set[str] = set()

  def get(self, state_type: str = _NODE_STATES) -> Dict[str, NodeState]:
    """Gets node states dict from pipeline execution with the specified type."""
    if state_type not in [_NODE_STATES, _PREVIOUS_NODE_STATES]:
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.INVALID_ARGUMENT,
          message=(
              f'Expected state_type is {_NODE_STATES} or'
              f' {_PREVIOUS_NODE_STATES}, got {state_type}.'
          ),
      )
    if state_type not in self._deserialized_cache:
      node_states_json = _get_metadata_value(
          self._custom_properties.get(state_type)
      )
      self._deserialized_cache[state_type] = (
          json_utils.loads(node_states_json) if node_states_json else {}
      )
    return self._deserialized_cache[state_type]

  def set(
      self, node_states: Dict[str, NodeState], state_type: str = _NODE_STATES
  ) -> None:
    """Sets node states dict with the specified type."""
    self._deserialized_cache[state_type] = node_states
    self._changed_state_types.add(state_type)

  def save(self) -> None:
    """Saves all changed node states dicts to pipeline execution."""
    max_mlmd_str_value_len = env.get_env().max_mlmd_str_value_length()

    for state_type in self._changed_state_types:
      node_states = self._deserialized_cache[state_type]
      node_states_json = json_utils.dumps(node_states)

      # Removes state history from node states if it's too large to avoid
      # hitting MLMD limit.
      if (
          max_mlmd_str_value_len
          and len(node_states_json) > max_mlmd_str_value_len
      ):
        logging.info(
            'Node states length %d is too large (> %d); Removing state history'
            ' from it.',
            len(node_states_json),
            max_mlmd_str_value_len,
        )
        node_states_no_history = {}
        for node, old_state in node_states.items():
          new_state = copy.deepcopy(old_state)
          new_state.state_history.clear()
          node_states_no_history[node] = new_state
        node_states_json = json_utils.dumps(node_states_no_history)
        logging.info(
            'Node states length after removing state history: %d',
            len(node_states_json),
        )

      data_types_utils.set_metadata_value(
          self._custom_properties[state_type], node_states_json
      )


def is_node_state_success(state: str) -> bool:
  return state in (NodeState.COMPLETE, NodeState.SKIPPED,
                   NodeState.SKIPPED_PARTIAL_RUN)


def is_node_state_failure(state: str) -> bool:
  return state == NodeState.FAILED


def is_node_state_running(state: str) -> bool:
  return state == NodeState.RUNNING


_NODE_STATE_TO_RUN_STATE_MAP = {
    NodeState.STARTED: run_state_pb2.RunState.READY,
    NodeState.STOPPING: run_state_pb2.RunState.UNKNOWN,
    NodeState.STOPPED: run_state_pb2.RunState.STOPPED,
    NodeState.RUNNING: run_state_pb2.RunState.RUNNING,
    NodeState.COMPLETE: run_state_pb2.RunState.COMPLETE,
    NodeState.SKIPPED: run_state_pb2.RunState.SKIPPED,
    NodeState.SKIPPED_PARTIAL_RUN: run_state_pb2.RunState.SKIPPED_PARTIAL_RUN,
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


# Signal to record whether there are active pipelines, this is an optimization
# to avoid generating too many RPC calls getting contexts/executions during
# idle time. Everytime when the pipeline state is updated to active (eg. start,
# resume a pipeline), this variable must be toggled to True. Default as True as
# well to make sure latest executions and contexts are checked when
# orchestrator starts or gets preempted.
# Note from sharded orchestrator: this flag ONLY ACCOUNTS FOR the active
# pipeline states of THIS orchestrator shard. Active pipelines for other
# orchestrator shards MUST NOT affect this.
_active_owned_pipelines_exist = True
# Lock to serialize the functions changing the _active_own_pipeline_exist
# status.
_active_pipelines_lock = threading.Lock()


def _synchronized(f):
  @functools.wraps(f)
  def wrapper(*args, **kwargs):
    with _active_pipelines_lock:
      return f(*args, **kwargs)

  return wrapper


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
    pipeline_decode_error: If not None, we failed to decode the pipeline proto
      from the MLMD execution.
    execution: The underlying execution in MLMD.
    execution_id: Id of the underlying execution in MLMD.
    pipeline_uid: Unique id of the pipeline.
    pipeline_run_id: pipeline_run_id in case of sync pipeline, `None` otherwise.
  """

  def __init__(
      self,
      mlmd_handle: metadata.Metadata,
      execution: metadata_store_pb2.Execution,
      pipeline_id: str,
  ):
    """Constructor. Use one of the factory methods to initialize."""
    self.mlmd_handle = mlmd_handle
    # TODO(b/201294315): Fix self.pipeline going out of sync with the actual
    # pipeline proto stored in the underlying MLMD execution in some cases.
    try:
      self.pipeline = _get_pipeline_from_orchestrator_execution(execution)  # pytype: disable=name-error
      self.pipeline_decode_error = None
    except Exception as e:  # pylint: disable=broad-except
      logging.exception('Failed to load pipeline IR')
      self.pipeline = pipeline_pb2.Pipeline()
      self.pipeline_decode_error = e
    self.execution_id = execution.id
    self.pipeline_run_id = None
    if _PIPELINE_RUN_ID in execution.custom_properties:
      self.pipeline_run_id = execution.custom_properties[
          _PIPELINE_RUN_ID
      ].string_value
    self.pipeline_uid = task_lib.PipelineUid.from_pipeline_id_and_run_id(
        pipeline_id, self.pipeline_run_id
    )

    # Only set within the pipeline state context.
    self._mlmd_execution_atomic_op_context = None
    self._execution: Optional[metadata_store_pb2.Execution] = None
    self._on_commit_callbacks: List[Callable[[], None]] = []
    # The note state proxy is assumed to be initialized before being used.
    self._node_states_proxy: _NodeStatesProxy = cast(_NodeStatesProxy, None)

  @classmethod
  @telemetry_utils.noop_telemetry(metrics_utils.no_op_metrics)
  @_synchronized
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
    num_subpipelines = 0
    to_process = collections.deque([pipeline])
    while to_process:
      p = to_process.popleft()
      for node in p.nodes:
        if node.WhichOneof('node') == 'sub_pipeline':
          num_subpipelines += 1
          to_process.append(node.sub_pipeline)
    # If the number of active task schedulers is less than the maximum number of
    # active task schedulers, subpipelines may not work.
    # This is because when scheduling the subpipeline, the start node
    # and end node will be scheduled immediately, potentially causing contention
    # where the end node is waiting on some intermediary node to finish, but the
    # intermediary node cannot be scheduled as the end node is running.
    # Note that this number is an overestimate - in reality if subpipelines are
    # dependent on each other we may not need so many task schedulers.
    max_task_schedulers = env.get_env().maximum_active_task_schedulers()
    if max_task_schedulers < num_subpipelines:
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.FAILED_PRECONDITION,
          message=(
              f'The maximum number of task schedulers ({max_task_schedulers})'
              f' is less than the number of subpipelines ({num_subpipelines}).'
              ' Please set the maximum number of task schedulers to at least'
              f' {num_subpipelines} in'
              ' OrchestrationOptions.max_running_components.'
          ),
      )
    pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
    context = context_lib.register_context_if_not_exists(
        mlmd_handle,
        context_type_name=_ORCHESTRATOR_RESERVED_ID,
        context_name=pipeline_uid.pipeline_id)

    active_pipeline_executions = mlmd_handle.store.get_executions_by_context(
        context.id,
        list_options=mlmd.ListOptions(
            filter_query='last_known_state = NEW OR last_known_state = RUNNING'
        ),
    )
    assert all(
        execution_lib.is_execution_active(e) for e in active_pipeline_executions
    )
    active_async_pipeline_executions = [
        e for e in active_pipeline_executions
        if _retrieve_pipeline_exec_mode(e) == pipeline_pb2.Pipeline.ASYNC
    ]

    # Disallow running concurrent async pipelines regardless of whether
    # concurrent pipeline runs are enabled.
    if (
        pipeline.execution_mode == pipeline_pb2.Pipeline.ASYNC
        and active_pipeline_executions
    ):
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.ALREADY_EXISTS,
          message=(
              'Cannot run an async pipeline concurrently when another '
              f'pipeline with id {pipeline_uid.pipeline_id} is active.'
          ),
      )

    if env.get_env().concurrent_pipeline_runs_enabled(pipeline):
      # If concurrent runs are enabled, we should still prohibit interference
      # with any active async pipelines so disallow starting a sync pipeline.
      if active_async_pipeline_executions:
        raise status_lib.StatusNotOkError(
            code=status_lib.Code.ALREADY_EXISTS,
            message=(
                'Cannot run a sync pipeline concurrently when an async '
                f'pipeline with id {pipeline_uid.pipeline_id} is active.'
            ),
        )
      # If concurrent runs are enabled, before starting a sync pipeline run,
      # ensure there isn't another active sync pipeline that shares the run id.
      if pipeline.execution_mode == pipeline_pb2.Pipeline.SYNC:
        assert pipeline_uid.pipeline_run_id is not None
        for e in active_pipeline_executions:
          if _get_metadata_value(e.custom_properties.get(
              _PIPELINE_RUN_ID)) == pipeline_uid.pipeline_run_id:
            raise status_lib.StatusNotOkError(
                code=status_lib.Code.ALREADY_EXISTS,
                message=(
                    'Another pipeline run having pipeline id'
                    f' {pipeline_uid.pipeline_id} and run id'
                    f' {pipeline_uid.pipeline_run_id} is already active.'
                ),
            )
    else:
      if active_pipeline_executions:
        raise status_lib.StatusNotOkError(
            code=status_lib.Code.ALREADY_EXISTS,
            message=(
                'Another pipeline run having pipeline id '
                f'{pipeline_uid.pipeline_id} is already active.'
            ),
        )

    # TODO(b/254161062): Consider disallowing pipeline exec mode change for the
    # same pipeline id.
    if pipeline.execution_mode == pipeline_pb2.Pipeline.SYNC:
      pipeline_exec_mode = _PIPELINE_EXEC_MODE_SYNC
    elif pipeline.execution_mode == pipeline_pb2.Pipeline.ASYNC:
      pipeline_exec_mode = _PIPELINE_EXEC_MODE_ASYNC
    else:
      raise ValueError('Expected pipeline execution mode to be SYNC or ASYNC')

    exec_properties = {
        _PIPELINE_IR: pipeline_ir_codec.PipelineIRCodec.get().encode(pipeline),
        _PIPELINE_EXEC_MODE: pipeline_exec_mode,
    }
    pipeline_run_metadata_json = None
    if pipeline_run_metadata:
      pipeline_run_metadata_json = json_utils.dumps(pipeline_run_metadata)
      exec_properties[_PIPELINE_RUN_METADATA] = pipeline_run_metadata_json

    execution = execution_lib.prepare_execution(
        mlmd_handle,
        _ORCHESTRATOR_EXECUTION_TYPE,
        metadata_store_pb2.Execution.NEW,
        exec_properties=exec_properties,
        execution_name=str(uuid.uuid4()),
    )
    if pipeline.execution_mode == pipeline_pb2.Pipeline.SYNC:
      data_types_utils.set_metadata_value(
          execution.custom_properties[_PIPELINE_RUN_ID],
          pipeline.runtime_spec.pipeline_run_id.field_value.string_value,
      )
      _save_skipped_node_states(pipeline, reused_pipeline_view, execution)

    # Find any normal pipeline node (possibly in a subpipeline) and prepare the
    # contexts, which will register the associated pipeline contexts and
    # pipeline run ID context.
    #
    # We do this so the pipeline contexts and pipeline run ID context are
    # created immediately when the pipeline is started, so we can immediately
    # associate extra information with them, rather than having to wait
    # until the orchestrator generates tasks for a node in the pipeline for
    # the contexts to be registered.
    #
    # If there are no normal nodes then no contexts are prepared.
    def _prepare_pipeline_node_contexts(
        pipeline: pipeline_pb2.Pipeline,
    ) -> bool:
      """Prepares contexts for any pipeline node in any sub pipeline layer."""
      for node in pipeline.nodes:
        if node.WhichOneof('node') == 'pipeline_node':
          context_lib.prepare_contexts(mlmd_handle, node.pipeline_node.contexts)
          return True
        elif node.WhichOneof('node') == 'sub_pipeline':
          if _prepare_pipeline_node_contexts(node.sub_pipeline):
            return True
      return False

    _prepare_pipeline_node_contexts(pipeline)

    # update _active_owned_pipelines_exist to be True so orchestrator will keep
    # fetching the latest contexts and execution when orchestrating the pipeline
    # run.
    global _active_owned_pipelines_exist
    _active_owned_pipelines_exist = True
    logging.info('Pipeline start, set active_pipelines_exist=True.')
    # Skip dual logging if MLMD backend does not have pipeline-asset support.
    pipeline_asset = mlmd_handle.store.pipeline_asset
    if pipeline_asset:
      env.get_env().create_sync_or_upsert_async_pipeline_run(
          pipeline_asset.owner,
          pipeline_asset.name,
          execution,
          pipeline,
          pipeline_run_metadata_json,
          reused_pipeline_view.pipeline_run_id
          if reused_pipeline_view
          else None,
      )
    execution = execution_lib.put_execution(mlmd_handle, execution, [context])
    pipeline_state = cls(mlmd_handle, execution, pipeline_uid.pipeline_id)
    event_observer.notify(
        event_observer.PipelineStarted(
            pipeline_uid=pipeline_uid, pipeline_state=pipeline_state
        )
    )
    record_state_change_time()
    return pipeline_state

  @classmethod
  @telemetry_utils.noop_telemetry(metrics_utils.no_op_metrics)
  def load(
      cls, mlmd_handle: metadata.Metadata, pipeline_uid: task_lib.PipelineUid
  ) -> 'PipelineState':
    """Loads pipeline state from MLMD.

    Args:
      mlmd_handle: A handle to the MLMD db.
      pipeline_uid: Uid of the pipeline state to load.

    Returns:
      A `PipelineState` object.

    Raises:
      status_lib.StatusNotOkError: With code=NOT_FOUND if no active pipeline
      with the given pipeline uid exists in MLMD. With code=FAILED_PRECONDITION
      if more than 1 active execution exists for given pipeline uid.
    """
    context = _get_orchestrator_context(mlmd_handle, pipeline_uid.pipeline_id)
    uids_and_states = cls._load_from_context(mlmd_handle, context, pipeline_uid)
    if not uids_and_states:
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.NOT_FOUND,
          message=f'No active pipeline with uid {pipeline_uid} to load state.')
    if len(uids_and_states) > 1:
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.FAILED_PRECONDITION,
          message=(
              f'Expected 1 but found {len(uids_and_states)} active pipelines '
              f'for pipeline uid: {pipeline_uid}'))
    return uids_and_states[0][1]

  @classmethod
  @telemetry_utils.noop_telemetry(metrics_utils.no_op_metrics)
  @_synchronized
  def load_all_active_and_owned(
      cls,
      mlmd_handle: metadata.Metadata,
  ) -> list['PipelineState']:
    """Loads all active pipeline states that the current orchestrator owns.

    Whether the pipeline state is owned by the current orchestrator or not is
    determined by the Env.should_orchestrate(). For example, whether the
    orchestrator is for the lightning mode or not, or for sharded orchestrator
    if the pipeline state belongs to the current shard.

    Args:
      mlmd_handle: A handle to the MLMD db.

    Returns:
      List of `PipelineState` objects for all active pipelines.

    Raises:
      status_lib.StatusNotOkError: With code=FAILED_PRECONDITION if more than
      one active pipeline are found with the same pipeline uid.
    """
    result: list['PipelineState'] = []
    global _active_owned_pipelines_exist
    if _active_owned_pipelines_exist:
      logging.info('Checking active pipelines.')
      contexts = get_orchestrator_contexts(mlmd_handle)
      active_pipeline_uids = set()
      for context in contexts:
        uids_and_states = cls._load_from_context(mlmd_handle, context)
        for pipeline_uid, pipeline_state in uids_and_states:
          if pipeline_uid in active_pipeline_uids:
            raise status_lib.StatusNotOkError(
                code=status_lib.Code.FAILED_PRECONDITION,
                message=(
                    'Found more than 1 active pipeline for pipeline uid:'
                    f' {pipeline_uid}'
                ),
            )
          active_pipeline_uids.add(pipeline_uid)
          result.append(pipeline_state)

    result = [
        ps for ps in result if env.get_env().should_orchestrate(ps.pipeline)
    ]
    if not result:
      _active_owned_pipelines_exist = False
      logging.info(
          'No active pipelines, set _active_owned_pipelines_exist=False.'
      )
    return result

  @classmethod
  def load_run(
      cls,
      mlmd_handle: metadata.Metadata,
      pipeline_id: str,
      run_id: str,
  ) -> 'PipelineState':
    """Loads pipeline state for a specific run from MLMD.

    Args:
      mlmd_handle: A handle to the MLMD db.
      pipeline_id: Id of the pipeline state to load.
      run_id: The run_id of the pipeline to load.

    Returns:
      A `PipelineState` object.

    Raises:
      status_lib.StatusNotOkError: With code=NOT_FOUND if no active pipeline
      with the given pipeline uid exists in MLMD. With code=INVALID_ARGUMENT if
      there is not exactly 1 active execution for given pipeline uid.
    """
    context = _get_orchestrator_context(mlmd_handle, pipeline_id)
    query = f'custom_properties.pipeline_run_id.string_value = "{run_id}"'
    executions = mlmd_handle.store.get_executions_by_context(
        context.id,
        list_options=mlmd.ListOptions(filter_query=query),
    )

    if len(executions) != 1:
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.FAILED_PRECONDITION,
          message=(
              f'Expected 1 but found {len(executions)} pipeline runs '
              f'for pipeline id: {pipeline_id} with run_id {run_id}'
          ),
      )

    return cls(
        mlmd_handle,
        executions[0],
        pipeline_id,
    )

  @classmethod
  def _load_from_context(
      cls,
      mlmd_handle: metadata.Metadata,
      context: metadata_store_pb2.Context,
      matching_pipeline_uid: Optional[task_lib.PipelineUid] = None,
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
    active_executions = mlmd_handle.store.get_executions_by_context(
        context.id,
        list_options=mlmd.ListOptions(
            filter_query='last_known_state = NEW OR last_known_state = RUNNING'
        ),
    )
    assert all(execution_lib.is_execution_active(e) for e in active_executions)
    result = []
    for execution in active_executions:
      pipeline_uid = task_lib.PipelineUid.from_pipeline_id_and_run_id(
          pipeline_id,
          _get_metadata_value(
              execution.custom_properties.get(_PIPELINE_RUN_ID)))
      if matching_pipeline_uid and pipeline_uid != matching_pipeline_uid:
        continue
      result.append(
          (pipeline_uid, PipelineState(mlmd_handle, execution, pipeline_id))
      )
    return result

  @property
  def execution(self) -> metadata_store_pb2.Execution:
    if self._execution is None:
      raise RuntimeError(
          'Operation must be performed within the pipeline state context.'
      )
    return self._execution

  def is_active(self) -> bool:
    """Returns `True` if pipeline is active."""
    return execution_lib.is_execution_active(self.execution)

  def initiate_stop(self, status: status_lib.Status) -> None:
    """Updates pipeline state to signal stopping pipeline execution."""
    data_types_utils.set_metadata_value(
        self.execution.custom_properties[_STOP_INITIATED], 1
    )
    data_types_utils.set_metadata_value(
        self.execution.custom_properties[_PIPELINE_STATUS_CODE],
        int(status.code),
    )
    if status.message:
      data_types_utils.set_metadata_value(
          self.execution.custom_properties[_PIPELINE_STATUS_MSG], status.message
      )

  @_synchronized
  def initiate_resume(self) -> None:
    global _active_owned_pipelines_exist
    _active_owned_pipelines_exist = True
    self._check_context()
    self.remove_property(_STOP_INITIATED)
    self.remove_property(_PIPELINE_STATUS_CODE)
    self.remove_property(_PIPELINE_STATUS_MSG)

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
      return [
          (
              node.node_info.id,
              list(node.upstream_nodes),
              list(node.downstream_nodes),
          )
          for node in node_proto_view.get_view_for_all_in(pipeline)
      ]

    if _structure(self.pipeline) != _structure(updated_pipeline):
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.INVALID_ARGUMENT,
          message=(
              'Updated pipeline should have the same structure as the original.'
          ),
      )

    env.get_env().prepare_orchestrator_for_pipeline_run(updated_pipeline)
    data_types_utils.set_metadata_value(
        self.execution.custom_properties[_UPDATED_PIPELINE_IR],
        pipeline_ir_codec.PipelineIRCodec.get().encode(updated_pipeline),
    )
    data_types_utils.set_metadata_value(
        self.execution.custom_properties[_UPDATE_OPTIONS],
        _base64_encode(update_options),
    )

  def is_update_initiated(self) -> bool:
    return (
        self.is_active()
        and self.execution.custom_properties.get(_UPDATED_PIPELINE_IR)
        is not None
    )

  def get_update_options(self) -> pipeline_pb2.UpdateOptions:
    """Gets pipeline update option that was previously configured."""
    update_options = self.execution.custom_properties.get(_UPDATE_OPTIONS)
    if update_options is None:
      logging.warning(
          'pipeline execution missing expected custom property %s, '
          'defaulting to UpdateOptions(reload_policy=ALL)', _UPDATE_OPTIONS)
      return pipeline_pb2.UpdateOptions(
          reload_policy=pipeline_pb2.UpdateOptions.ReloadPolicy.ALL)
    return _base64_decode_update_options(_get_metadata_value(update_options))

  def apply_pipeline_update(self) -> None:
    """Applies pipeline update that was previously initiated."""
    updated_pipeline_ir = _get_metadata_value(
        self.execution.custom_properties.get(_UPDATED_PIPELINE_IR)
    )
    if not updated_pipeline_ir:
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.INVALID_ARGUMENT,
          message='No updated pipeline IR to apply')
    data_types_utils.set_metadata_value(
        self.execution.properties[_PIPELINE_IR], updated_pipeline_ir
    )
    del self.execution.custom_properties[_UPDATED_PIPELINE_IR]
    del self.execution.custom_properties[_UPDATE_OPTIONS]
    self.pipeline = pipeline_ir_codec.PipelineIRCodec.get().decode(
        updated_pipeline_ir
    )

  def is_stop_initiated(self) -> bool:
    self._check_context()
    return self.stop_initiated_reason() is not None

  def stop_initiated_reason(self) -> Optional[status_lib.Status]:
    """Returns status object if stop initiated, `None` otherwise."""
    custom_properties = self.execution.custom_properties
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
    node_states_dict = self._node_states_proxy.get()
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
    if old_state != node_state:
      self._node_states_proxy.set(node_states_dict)

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
    node_states_dict = self._node_states_proxy.get(state_type)
    return node_states_dict.get(node_uid.node_id, NodeState())

  def get_node_states_dict(self) -> Dict[task_lib.NodeUid, NodeState]:
    """Gets all node states of the pipeline."""
    self._check_context()
    node_states_dict = self._node_states_proxy.get()
    result = {}
    for node in node_proto_view.get_view_for_all_in(self.pipeline):
      node_uid = task_lib.NodeUid.from_node(self.pipeline, node)
      result[node_uid] = node_states_dict.get(node_uid.node_id, NodeState())
    return result

  def get_previous_node_states_dict(self) -> Dict[task_lib.NodeUid, NodeState]:
    """Gets all node states of the pipeline from previous run."""
    self._check_context()
    node_states_dict = self._node_states_proxy.get(_PREVIOUS_NODE_STATES)
    result = {}
    for node in node_proto_view.get_view_for_all_in(self.pipeline):
      node_uid = task_lib.NodeUid.from_node(self.pipeline, node)
      if node_uid.node_id not in node_states_dict:
        continue
      result[node_uid] = node_states_dict[node_uid.node_id]
    return result

  def get_pipeline_execution_state(self) -> metadata_store_pb2.Execution.State:
    """Returns state of underlying pipeline execution."""
    return self.execution.last_known_state

  def set_pipeline_execution_state(
      self, state: metadata_store_pb2.Execution.State) -> None:
    """Sets state of underlying pipeline execution."""
    if self.execution.last_known_state != state:
      self._on_commit_callbacks.append(
          functools.partial(
              _log_pipeline_execution_state_change,
              self.execution.last_known_state,
              state,
              self.pipeline_uid,
          )
      )
      self.execution.last_known_state = state

  def get_property(self, property_key: str) -> Optional[types.Property]:
    """Returns custom property value from the pipeline execution."""
    return _get_metadata_value(
        self.execution.custom_properties.get(property_key)
    )

  def save_property(
      self, property_key: str, property_value: types.Property
  ) -> None:
    data_types_utils.set_metadata_value(
        self.execution.custom_properties[property_key], property_value
    )

  def remove_property(self, property_key: str) -> None:
    """Removes a custom property of the pipeline execution if exists."""
    if self.execution.custom_properties.get(property_key):
      del self.execution.custom_properties[property_key]

  def pipeline_creation_time_secs_since_epoch(self) -> int:
    """Returns the pipeline creation time as seconds since epoch."""
    # Convert from milliseconds to seconds.
    return self.execution.create_time_since_epoch // 1000

  def get_orchestration_options(
      self) -> orchestration_options.OrchestrationOptions:
    self._check_context()
    return env.get_env().get_orchestration_options(self.pipeline)

  def __enter__(self) -> 'PipelineState':

    def _pre_commit(original_execution, modified_execution):
      pipeline_asset = self.mlmd_handle.store.pipeline_asset
      if not pipeline_asset:
        logging.warning('Pipeline asset not found.')
        return
      env.get_env().update_pipeline_run_status(
          pipeline_asset.owner,
          pipeline_asset.name,
          self.pipeline,
          original_execution,
          modified_execution,
          _get_sub_pipeline_ids_from_pipeline_info(self.pipeline.pipeline_info),
      )

    def _run_on_commit_callbacks(pre_commit_execution, post_commit_execution):
      del pre_commit_execution
      del post_commit_execution
      record_state_change_time()
      for on_commit_cb in self._on_commit_callbacks:
        on_commit_cb()

    mlmd_execution_atomic_op_context = mlmd_state.mlmd_execution_atomic_op(
        self.mlmd_handle,
        self.execution_id,
        _run_on_commit_callbacks,
        _pre_commit,
    )
    execution = mlmd_execution_atomic_op_context.__enter__()
    self._mlmd_execution_atomic_op_context = mlmd_execution_atomic_op_context
    self._execution = execution
    self._node_states_proxy = _NodeStatesProxy(execution)
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self._node_states_proxy.save()
    mlmd_execution_atomic_op_context = self._mlmd_execution_atomic_op_context
    self._mlmd_execution_atomic_op_context = None
    self._execution = None
    try:
      assert mlmd_execution_atomic_op_context is not None
      mlmd_execution_atomic_op_context.__exit__(exc_type, exc_val, exc_tb)
    finally:
      self._on_commit_callbacks.clear()

  def _check_context(self) -> None:
    if self._execution is None:
      raise RuntimeError(
          'Operation must be performed within the pipeline state context.')


class PipelineView:
  """Class for reading active or inactive pipeline view."""

  def __init__(self, pipeline_id: str, execution: metadata_store_pb2.Execution):
    self.pipeline_id = pipeline_id
    self.execution = execution
    self._node_states_proxy = _NodeStatesProxy(execution)
    self.pipeline_run_id = None
    if _PIPELINE_RUN_ID in execution.custom_properties:
      self.pipeline_run_id = execution.custom_properties[
          _PIPELINE_RUN_ID
      ].string_value
    self.pipeline_uid = task_lib.PipelineUid.from_pipeline_id_and_run_id(
        self.pipeline_id, self.pipeline_run_id
    )
    self._pipeline = None  # lazily set

  @classmethod
  def load_all(
      cls,
      mlmd_handle: metadata.Metadata,
      pipeline_id: str,
      list_options: Optional[mlmd.ListOptions] = None,
      **kwargs,
  ) -> List['PipelineView']:
    """Loads all pipeline views from MLMD.

    Args:
      mlmd_handle: A handle to the MLMD db.
      pipeline_id: Id of the pipeline state to load.
      list_options: List options to customize the query for getting executions.
      **kwargs: Extra option to pass into mlmd store functions.

    Returns:
      A list of `PipelineView` objects.

    Raises:
      status_lib.StatusNotOkError: With code=NOT_FOUND if no pipeline
      with the given pipeline uid exists in MLMD.
    """
    context = _get_orchestrator_context(mlmd_handle, pipeline_id, **kwargs)
    # TODO(b/279798582):
    # Uncomment the following when the slow sorting MLMD query is fixed.
    # list_options = mlmd.ListOptions(
    #    order_by=mlmd.OrderByField.CREATE_TIME, is_asc=True)
    executions = mlmd_handle.store.get_executions_by_context(
        context.id, list_options=list_options, **kwargs
    )
    executions = sorted(executions, key=lambda x: x.create_time_since_epoch)
    return [cls(pipeline_id, execution) for execution in executions]

  @classmethod
  def load(cls,
           mlmd_handle: metadata.Metadata,
           pipeline_id: str,
           pipeline_run_id: Optional[str] = None,
           non_active_only: Optional[bool] = False,
           **kwargs) -> 'PipelineView':
    """Loads pipeline view from MLMD.

    Args:
      mlmd_handle: A handle to the MLMD db.
      pipeline_id: Id of the pipeline state to load.
      pipeline_run_id: Run id of the pipeline for the synchronous pipeline.
      non_active_only: Whether to only load from a non-active pipeline.
      **kwargs: Extra option to pass into mlmd store functions.

    Returns:
      A `PipelineView` object.

    Raises:
      status_lib.StatusNotOkError: With code=NOT_FOUND if no pipeline
      with the given pipeline uid exists in MLMD.
    """
    context = _get_orchestrator_context(mlmd_handle, pipeline_id, **kwargs)
    filter_query = ''
    if non_active_only:
      filter_query = 'last_known_state != RUNNING AND last_known_state != NEW'
    list_options = mlmd.ListOptions(
        order_by=mlmd.OrderByField.CREATE_TIME,
        is_asc=False,
        filter_query=filter_query,
        limit=1,
    )
    if pipeline_run_id:
      # Note(b/281478984):
      # This optimization is done for requests with pipeline run id
      # by specifying which pipeline run is queried.
      # Order by with this filter query is slow with large # of runs.
      list_options = mlmd.ListOptions(
          filter_query=(
              'custom_properties.pipeline_run_id.string_value ='
              f' "{pipeline_run_id}"'
          )
      )
    executions = mlmd_handle.store.get_executions_by_context(
        context.id, list_options=list_options, **kwargs
    )

    non_active_msg = 'non active ' if non_active_only else ''
    if executions:
      if len(executions) != 1:
        raise status_lib.StatusNotOkError(
            code=status_lib.Code.FAILED_PRECONDITION,
            message=(
                'Expected 1 but found'
                f' {len(executions)} {non_active_msg}'
                f' runs for pipeline id: {pipeline_id} with run_id'
                f' {pipeline_run_id}'
            ),
        )
      return cls(pipeline_id, executions[0])

    raise status_lib.StatusNotOkError(
        code=status_lib.Code.NOT_FOUND,
        message=(
            f'No {non_active_msg} pipeline with run_id {pipeline_run_id} found.'
        ),
    )

  @property
  def pipeline(self) -> pipeline_pb2.Pipeline:
    if self._pipeline is None:
      try:
        self._pipeline = _get_pipeline_from_orchestrator_execution(
            self.execution
        )
      except Exception:  # pylint: disable=broad-except
        logging.exception('Failed to load pipeline IR for %s', self.pipeline_id)
        self._pipeline = pipeline_pb2.Pipeline()
    return self._pipeline

  @property
  def pipeline_execution_mode(self) -> pipeline_pb2.Pipeline.ExecutionMode:
    return _retrieve_pipeline_exec_mode(self.execution)

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
    node_states_dict = self._node_states_proxy.get()
    for node in node_proto_view.get_view_for_all_in(self.pipeline):
      node_state = node_states_dict.get(node.node_info.id, NodeState())
      result[node.node_info.id] = node_state.to_run_state()
    return result

  def get_node_run_states_history(
      self) -> Dict[str, List[run_state_pb2.RunState]]:
    """Returns the history of node run states and timestamps."""
    node_states_dict = self._node_states_proxy.get()
    result = {}
    for node in node_proto_view.get_view_for_all_in(self.pipeline):
      node_state = node_states_dict.get(node.node_info.id, NodeState())
      result[node.node_info.id] = node_state.to_run_state_history()
    return result

  def get_previous_node_run_states(self) -> Dict[str, run_state_pb2.RunState]:
    """Returns a dict mapping node id to previous run state."""
    result = {}
    node_states_dict = self._node_states_proxy.get(_PREVIOUS_NODE_STATES)
    for node in node_proto_view.get_view_for_all_in(self.pipeline):
      if node.node_info.id not in node_states_dict:
        continue
      node_state = node_states_dict[node.node_info.id]
      result[node.node_info.id] = node_state.to_run_state()
    return result

  def get_previous_node_run_states_history(
      self) -> Dict[str, List[run_state_pb2.RunState]]:
    """Returns a dict mapping node id to previous run state and timestamps."""
    prev_node_states_dict = self._node_states_proxy.get(_PREVIOUS_NODE_STATES)
    result = {}
    for node in node_proto_view.get_view_for_all_in(self.pipeline):
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
    node_states_dict = self._node_states_proxy.get()
    for node in node_proto_view.get_view_for_all_in(self.pipeline):
      result[node.node_info.id] = node_states_dict.get(node.node_info.id,
                                                       NodeState())
    return result

  def get_previous_node_states_dict(self) -> Dict[str, NodeState]:
    """Returns a dict mapping node id to node state in previous run."""
    result = {}
    node_states_dict = self._node_states_proxy.get(_PREVIOUS_NODE_STATES)
    for node in node_proto_view.get_view_for_all_in(self.pipeline):
      if node.node_info.id not in node_states_dict:
        continue
      result[node.node_info.id] = node_states_dict[node.node_info.id]
    return result

  def get_node_state(self, node_uid: task_lib.NodeUid) -> NodeState:
    """Gets node state of a specified node."""
    if not _is_node_uid_in_pipeline(node_uid, self.pipeline):
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.INVALID_ARGUMENT,
          message=(
              f'Node {node_uid} does not belong to the pipeline '
              f'{self.pipeline_uid}'
          ),
      )
    node_states_dict = self.get_node_states_dict()
    return node_states_dict.get(node_uid.node_id, NodeState())


def get_orchestrator_contexts(mlmd_handle: metadata.Metadata,
                              **kwargs) -> List[metadata_store_pb2.Context]:
  """Returns all of the orchestrator contexts."""
  return mlmd_handle.store.get_contexts_by_type(_ORCHESTRATOR_RESERVED_ID,
                                                **kwargs)


def pipeline_id_from_orchestrator_context(
    context: metadata_store_pb2.Context) -> str:
  """Returns pipeline id from orchestrator reserved context."""
  return context.name


@telemetry_utils.noop_telemetry(metrics_utils.no_op_metrics)
def get_all_node_executions(
    pipeline: pipeline_pb2.Pipeline,
    mlmd_handle: metadata.Metadata,
    node_filter_options: Optional[metadata_pb2.NodeFilterOptions] = None,
) -> Dict[str, List[metadata_store_pb2.Execution]]:
  """Returns all executions of all pipeline nodes if present."""
  # TODO(b/310712984): Make use of Tflex MLMD filter query builder once
  # developed.
  additional_filters = None
  if node_filter_options is not None:
    additional_filters = []
    if node_filter_options.max_create_time.ToMilliseconds() > 0:
      additional_filters.append(
          'create_time_since_epoch <='
          f' {node_filter_options.max_create_time.ToMilliseconds()}'
      )
    if node_filter_options.min_create_time.ToMilliseconds() > 0:
      additional_filters.append(
          'create_time_since_epoch >='
          f' {node_filter_options.min_create_time.ToMilliseconds()}'
      )
    if node_filter_options.types:
      type_filter_query = '","'.join(node_filter_options.types)
      additional_filters.append(f'type IN ("{type_filter_query}")')
  return {
      node.node_info.id: task_gen_utils.get_executions(
          mlmd_handle, node, additional_filters=additional_filters
      )
      for node in node_proto_view.get_view_for_all_in(pipeline)
  }


@telemetry_utils.noop_telemetry(metrics_utils.no_op_metrics)
def get_all_node_artifacts(
    pipeline: pipeline_pb2.Pipeline,
    mlmd_handle: metadata.Metadata,
    execution_filter_options: Optional[metadata_pb2.NodeFilterOptions] = None,
) -> Dict[str, Dict[int, Dict[str, List[metadata_store_pb2.Artifact]]]]:
  """Returns all output artifacts of all pipeline nodes if present.

  Args:
    pipeline: Pipeline proto associated with a `PipelineState` object.
    mlmd_handle: Handle to MLMD db.
    execution_filter_options: Filter options on executions from which the output
      artifacts are created.

  Returns:
    Dict of node id to Dict of execution id to Dict of key to output artifact
    list.
  """

  executions_dict = get_all_node_executions(
      pipeline, mlmd_handle, node_filter_options=execution_filter_options
  )
  result = {}
  for node_id, executions in executions_dict.items():
    node_artifacts = {}
    for execution in executions:
      execution_artifacts = {}
      for key, artifacts in execution_lib.get_output_artifacts(
          mlmd_handle, execution.id).items():
        execution_artifacts[key] = [
            artifact.mlmd_artifact for artifact in artifacts
        ]
      node_artifacts[execution.id] = execution_artifacts
    result[node_id] = node_artifacts
  return result


def _is_node_uid_in_pipeline(node_uid: task_lib.NodeUid,
                             pipeline: pipeline_pb2.Pipeline) -> bool:
  """Returns `True` if the `node_uid` belongs to the given pipeline."""
  for node in node_proto_view.get_view_for_all_in(pipeline):
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
  return pipeline_ir_codec.PipelineIRCodec.get().decode(pipeline_ir)


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


def _base64_decode_update_options(
    update_options_encoded: str) -> pipeline_pb2.UpdateOptions:
  result = pipeline_pb2.UpdateOptions()
  result.ParseFromString(base64.b64decode(update_options_encoded))
  return result


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
  reused_pipeline_previous_node_states_dict = (
      reused_pipeline_view.get_previous_node_states_dict()
      if reused_pipeline_view
      else {}
  )
  for node in node_proto_view.get_view_for_all_in(pipeline):
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
  node_states_proxy = _NodeStatesProxy(execution)
  if node_states_dict:
    node_states_proxy.set(node_states_dict, _NODE_STATES)
  if previous_node_states_dict:
    node_states_proxy.set(previous_node_states_dict, _PREVIOUS_NODE_STATES)
  node_states_proxy.save()


def _retrieve_pipeline_exec_mode(
    execution: metadata_store_pb2.Execution
) -> pipeline_pb2.Pipeline.ExecutionMode:
  """Returns pipeline execution mode given pipeline-level execution."""
  pipeline_exec_mode = _get_metadata_value(
      execution.custom_properties.get(_PIPELINE_EXEC_MODE))
  if pipeline_exec_mode == _PIPELINE_EXEC_MODE_SYNC:
    return pipeline_pb2.Pipeline.SYNC
  elif pipeline_exec_mode == _PIPELINE_EXEC_MODE_ASYNC:
    return pipeline_pb2.Pipeline.ASYNC
  else:
    return pipeline_pb2.Pipeline.EXECUTION_MODE_UNSPECIFIED


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
          pipeline_uid=node_uid.pipeline_uid,
          pipeline_run=pipeline_run_id,
          node_id=node_uid.node_id,
          old_state=old_state,
          new_state=new_state))


def _get_sub_pipeline_ids_from_pipeline_info(
    pipeline_info: pipeline_pb2.PipelineInfo,
) -> Optional[List[str]]:
  """Returns sub pipeline ids from pipeline info if parent_ids exists."""
  sub_pipeline_ids = None
  if pipeline_info.parent_ids:
    sub_pipeline_ids = pipeline_info.parent_ids[1:]
    sub_pipeline_ids.append(pipeline_info.id)
  return sub_pipeline_ids


def get_pipeline_and_node(
    mlmd_handle: metadata.Metadata,
    node_uid: task_lib.NodeUid,
    pipeline_run_id: str,
) -> tuple[pipeline_pb2.Pipeline, node_proto_view.PipelineNodeProtoView]:
  """Gets the pipeline and node for the node_uid.

  This function is experimental, and should only be used when publishing
  external and intermediate artifacts.

  Args:
      mlmd_handle: A handle to the MLMD db.
      node_uid: Node uid of the node to get.
      pipeline_run_id: Run id of the pipeline for the synchronous pipeline.

  Returns:
  A tuple with the pipeline and node proto view for the node_uid.
  """
  with PipelineState.load(mlmd_handle, node_uid.pipeline_uid) as pipeline_state:
    if (
        pipeline_run_id or pipeline_state.pipeline_run_id
    ) and pipeline_run_id != pipeline_state.pipeline_run_id:
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.NOT_FOUND,
          message=(
              'Unable to find an active pipeline run for pipeline_run_id: '
              f'{pipeline_run_id}'
          ),
      )
    nodes = node_proto_view.get_view_for_all_in(pipeline_state.pipeline)
    filtered_nodes = [n for n in nodes if n.node_info.id == node_uid.node_id]
    if len(filtered_nodes) != 1:
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.NOT_FOUND,
          message=f'unable to find node: {node_uid}',
      )
    node = filtered_nodes[0]
    if not isinstance(node, node_proto_view.PipelineNodeProtoView):
      raise ValueError(
          f'Unexpected type for node {node.node_info.id}. Only '
          'pipeline nodes are supported for external executions.'
      )
    return (pipeline_state.pipeline, node)
