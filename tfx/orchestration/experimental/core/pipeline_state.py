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
import threading
import time
from typing import List, Optional

from tfx import types
from tfx.orchestration import data_types_utils
from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import mlmd_state
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.portable.mlmd import context_lib
from tfx.orchestration.portable.mlmd import execution_lib
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import status as status_lib

from ml_metadata.proto import metadata_store_pb2

_ORCHESTRATOR_RESERVED_ID = '__ORCHESTRATOR__'
_PIPELINE_IR = 'pipeline_ir'
_STOP_INITIATED = 'stop_initiated'
_PIPELINE_RUN_ID = 'pipeline_run_id'
_PIPELINE_STATUS_CODE = 'pipeline_status_code'
_PIPELINE_STATUS_MSG = 'pipeline_status_msg'
_NODE_STOP_INITIATED_PREFIX = 'node_stop_initiated_'
_NODE_STATUS_CODE_PREFIX = 'node_status_code_'
_NODE_STATUS_MSG_PREFIX = 'node_status_msg_'
_ORCHESTRATOR_EXECUTION_TYPE = metadata_store_pb2.ExecutionType(
    name=_ORCHESTRATOR_RESERVED_ID,
    properties={_PIPELINE_IR: metadata_store_pb2.STRING})

_last_state_change_time_secs = -1.0
_state_change_time_lock = threading.Lock()


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
  """Class for dealing with pipeline state.

  Can be used as a context manager.

  Methods must be invoked inside the pipeline state context for thread safety
  and ensuring that in-memory state is kept in sync with corresponding state in
  MLMD. If the underlying pipeline execution is mutated, it is automatically
  committed when exiting the context so no separate commit operation is needed.

  Attributes:
    mlmd_handle: Handle to MLMD db.
    pipeline: The pipeline proto associated with this `PipelineState` object.
    execution_id: Id of the underlying execution in MLMD.
    pipeline_uid: Unique id of the pipeline.
  """

  def __init__(self, mlmd_handle: metadata.Metadata,
               pipeline: pipeline_pb2.Pipeline, execution_id: int):
    """Constructor. Use one of the factory methods to initialize."""
    self.mlmd_handle = mlmd_handle
    self.pipeline = pipeline
    self.execution_id = execution_id
    self.pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)

    # Only set within the pipeline state context.
    self._mlmd_execution_atomic_op_context = None
    self._execution = None

  @classmethod
  def new(cls, mlmd_handle: metadata.Metadata,
          pipeline: pipeline_pb2.Pipeline) -> 'PipelineState':
    """Creates a `PipelineState` object for a new pipeline.

    No active pipeline with the same pipeline uid should exist for the call to
    be successful.

    Args:
      mlmd_handle: A handle to the MLMD db.
      pipeline: IR of the pipeline.

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

    execution = execution_lib.prepare_execution(
        mlmd_handle,
        _ORCHESTRATOR_EXECUTION_TYPE,
        metadata_store_pb2.Execution.NEW,
        exec_properties={
            _PIPELINE_IR:
                base64.b64encode(pipeline.SerializeToString()).decode('utf-8')
        },
    )
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
    record_state_change_time()

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

  def initiate_node_start(self, node_uid: task_lib.NodeUid) -> None:
    """Updates pipeline state to signal that a node should be started."""
    self._check_context()
    if self.pipeline.execution_mode != pipeline_pb2.Pipeline.ASYNC:
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.UNIMPLEMENTED,
          message='Node can be started only for async pipelines.')
    if not _is_node_uid_in_pipeline(node_uid, self.pipeline):
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.INVALID_ARGUMENT,
          message=(f'Node given by uid {node_uid} does not belong to pipeline '
                   f'given by uid {self.pipeline_uid}'))
    if self._execution.custom_properties.pop(
        _node_stop_initiated_property(node_uid), None) is not None:
      self._execution.custom_properties.pop(
          _node_status_code_property(node_uid), None)
      self._execution.custom_properties.pop(
          _node_status_msg_property(node_uid), None)
    record_state_change_time()

  def initiate_node_stop(self, node_uid: task_lib.NodeUid,
                         status: status_lib.Status) -> None:
    """Updates pipeline state to signal that a node should be stopped."""
    self._check_context()
    if self.pipeline.execution_mode != pipeline_pb2.Pipeline.ASYNC:
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.UNIMPLEMENTED,
          message='Node can be stopped only for async pipelines.')
    if not _is_node_uid_in_pipeline(node_uid, self.pipeline):
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.INVALID_ARGUMENT,
          message=(f'Node given by uid {node_uid} does not belong to pipeline '
                   f'given by uid {self.pipeline_uid}'))
    data_types_utils.set_metadata_value(
        self._execution.custom_properties[_node_stop_initiated_property(
            node_uid)], 1)
    data_types_utils.set_metadata_value(
        self._execution.custom_properties[_node_status_code_property(node_uid)],
        int(status.code))
    if status.message:
      data_types_utils.set_metadata_value(
          self._execution.custom_properties[_node_status_msg_property(
              node_uid)], status.message)
    record_state_change_time()

  def node_stop_initiated_reason(
      self, node_uid: task_lib.NodeUid) -> Optional[status_lib.Status]:
    """Returns status object if node stop initiated, `None` otherwise."""
    self._check_context()
    if node_uid.pipeline_uid != self.pipeline_uid:
      raise RuntimeError(
          f'Node given by uid {node_uid} does not belong to pipeline given '
          f'by uid {self.pipeline_uid}')
    custom_properties = self._execution.custom_properties
    if _get_metadata_value(
        custom_properties.get(_node_stop_initiated_property(node_uid))) == 1:
      code = _get_metadata_value(
          custom_properties.get(_node_status_code_property(node_uid)))
      if code is None:
        code = status_lib.Code.UNKNOWN
      message = _get_metadata_value(
          custom_properties.get(_node_status_msg_property(node_uid)))
      return status_lib.Status(code=code, message=message)
    else:
      return None

  def get_pipeline_execution_state(self) -> metadata_store_pb2.Execution.State:
    """Returns state of underlying pipeline execution."""
    self._check_context()
    return self._execution.last_known_state

  def set_pipeline_execution_state(
      self, state: metadata_store_pb2.Execution.State) -> None:
    """Sets state of underlying pipeline execution."""
    self._check_context()
    self._execution.last_known_state = state

  def set_pipeline_execution_state_from_status(
      self, status: status_lib.Status) -> None:
    """Sets state of underlying pipeline execution derived from input status."""
    self._check_context()
    self._execution.last_known_state = _mlmd_execution_code(status)

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

  def __enter__(self) -> 'PipelineState':
    mlmd_execution_atomic_op_context = mlmd_state.mlmd_execution_atomic_op(
        self.mlmd_handle, self.execution_id)
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


def _node_stop_initiated_property(node_uid: task_lib.NodeUid) -> str:
  return f'{_NODE_STOP_INITIATED_PREFIX}{node_uid.node_id}'


def _node_status_code_property(node_uid: task_lib.NodeUid) -> str:
  return f'{_NODE_STATUS_CODE_PREFIX}{node_uid.node_id}'


def _node_status_msg_property(node_uid: task_lib.NodeUid) -> str:
  return f'{_NODE_STATUS_MSG_PREFIX}{node_uid.node_id}'


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


def _mlmd_execution_code(
    status: status_lib.Status) -> metadata_store_pb2.Execution.State:
  if status.code == status_lib.Code.OK:
    return metadata_store_pb2.Execution.COMPLETE
  elif status.code == status_lib.Code.CANCELLED:
    return metadata_store_pb2.Execution.CANCELED
  return metadata_store_pb2.Execution.FAILED


def _get_pipeline_from_orchestrator_execution(
    execution: metadata_store_pb2.Execution) -> pipeline_pb2.Pipeline:
  pipeline_ir_b64 = data_types_utils.get_metadata_value(
      execution.properties[_PIPELINE_IR])
  pipeline = pipeline_pb2.Pipeline()
  pipeline.ParseFromString(base64.b64decode(pipeline_ir_b64))
  return pipeline


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
