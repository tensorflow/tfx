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
"""Pipeline-level operations."""

import copy
import functools
import threading
import time
import typing
from typing import Callable, List, Mapping, Optional

from absl import logging
import attr
from tfx import types
from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import async_pipeline_task_gen
from tfx.orchestration.experimental.core import constants
from tfx.orchestration.experimental.core import mlmd_state
from tfx.orchestration.experimental.core import pipeline_state as pstate
from tfx.orchestration.experimental.core import service_jobs
from tfx.orchestration.experimental.core import sync_pipeline_task_gen
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import task_gen_utils
from tfx.orchestration.experimental.core import task_queue as tq
from tfx.orchestration.experimental.core.task_schedulers import manual_task_scheduler
from tfx.orchestration.portable.mlmd import execution_lib
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import status as status_lib

from ml_metadata.proto import metadata_store_pb2

# A coarse grained lock is used to ensure serialization of pipeline operations
# since there isn't a suitable MLMD transaction API.
_PIPELINE_OPS_LOCK = threading.RLock()


def _pipeline_ops_lock(fn):
  """Decorator to run `fn` within `_PIPELINE_OPS_LOCK` context."""

  @functools.wraps(fn)
  def _wrapper(*args, **kwargs):
    with _PIPELINE_OPS_LOCK:
      return fn(*args, **kwargs)

  return _wrapper


def _to_status_not_ok_error(fn):
  """Decorator to catch exceptions and re-raise a `status_lib.StatusNotOkError`."""

  @functools.wraps(fn)
  def _wrapper(*args, **kwargs):
    try:
      return fn(*args, **kwargs)
    except Exception as e:  # pylint: disable=broad-except
      logging.exception('Error raised by `%s`:', fn.__name__)
      if isinstance(e, status_lib.StatusNotOkError):
        raise
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.UNKNOWN,
          message=f'`{fn.__name__}` error: {str(e)}')

  return _wrapper


@_to_status_not_ok_error
@_pipeline_ops_lock
def initiate_pipeline_start(
    mlmd_handle: metadata.Metadata,
    pipeline: pipeline_pb2.Pipeline,
    pipeline_run_metadata: Optional[Mapping[str, types.Property]] = None
) -> pstate.PipelineState:
  """Initiates a pipeline start operation.

  Upon success, MLMD is updated to signal that the pipeline must be started.

  Args:
    mlmd_handle: A handle to the MLMD db.
    pipeline: IR of the pipeline to start.
    pipeline_run_metadata: Pipeline run metadata.

  Returns:
    The `PipelineState` object upon success.

  Raises:
    status_lib.StatusNotOkError: Failure to initiate pipeline start. With code
      `INVALILD_ARGUMENT` if it's a sync pipeline without `pipeline_run_id`
      provided.
  """
  logging.info('Received request to start pipeline; pipeline uid: %s',
               task_lib.PipelineUid.from_pipeline(pipeline))
  pipeline = copy.deepcopy(pipeline)
  if pipeline.execution_mode == pipeline_pb2.Pipeline.SYNC and not (
      pipeline.runtime_spec.pipeline_run_id.HasField('field_value') and
      pipeline.runtime_spec.pipeline_run_id.field_value.string_value):
    raise status_lib.StatusNotOkError(
        code=status_lib.Code.INVALID_ARGUMENT,
        message='Sync pipeline IR must specify pipeline_run_id.')

  return pstate.PipelineState.new(mlmd_handle, pipeline, pipeline_run_metadata)


@_to_status_not_ok_error
def stop_pipeline(mlmd_handle: metadata.Metadata,
                  pipeline_uid: task_lib.PipelineUid,
                  timeout_secs: Optional[float] = None) -> None:
  """Stops a pipeline.

  Initiates a pipeline stop operation and waits for the pipeline execution to be
  gracefully stopped in the orchestration loop.

  Args:
    mlmd_handle: A handle to the MLMD db.
    pipeline_uid: Uid of the pipeline to be stopped.
    timeout_secs: Amount of time in seconds to wait for pipeline to stop. If
      `None`, waits indefinitely.

  Raises:
    status_lib.StatusNotOkError: Failure to initiate pipeline stop.
  """
  logging.info('Received request to stop pipeline; pipeline uid: %s',
               pipeline_uid)
  with _PIPELINE_OPS_LOCK:
    with pstate.PipelineState.load(mlmd_handle, pipeline_uid) as pipeline_state:
      pipeline_state.initiate_stop(
          status_lib.Status(
              code=status_lib.Code.CANCELLED,
              message='Cancellation requested by client.'))
  logging.info('Waiting for pipeline to be stopped; pipeline uid: %s',
               pipeline_uid)
  _wait_for_inactivation(
      mlmd_handle, pipeline_state.execution_id, timeout_secs=timeout_secs)
  logging.info('Done waiting for pipeline to be stopped; pipeline uid: %s',
               pipeline_uid)


@_to_status_not_ok_error
@_pipeline_ops_lock
def initiate_node_start(mlmd_handle: metadata.Metadata,
                        node_uid: task_lib.NodeUid) -> pstate.PipelineState:
  """Initiates a node start operation for a pipeline node.

  Args:
    mlmd_handle: A handle to the MLMD db.
    node_uid: Uid of the node to be started.

  Returns:
    The `PipelineState` object upon success.

  Raises:
    status_lib.StatusNotOkError: Failure to initiate node start operation.
  """
  logging.info('Received request to start node; node uid: %s', node_uid)
  with pstate.PipelineState.load(mlmd_handle,
                                 node_uid.pipeline_uid) as pipeline_state:
    with pipeline_state.node_state_update_context(node_uid) as node_state:
      if node_state.is_startable():
        node_state.update(pstate.NodeState.STARTING)
  return pipeline_state


@_to_status_not_ok_error
def stop_node(mlmd_handle: metadata.Metadata,
              node_uid: task_lib.NodeUid,
              timeout_secs: Optional[float] = None) -> None:
  """Stops a node.

  Initiates a node stop operation and waits for the node execution to become
  inactive.

  Args:
    mlmd_handle: A handle to the MLMD db.
    node_uid: Uid of the node to be stopped.
    timeout_secs: Amount of time in seconds to wait for node to stop. If `None`,
      waits indefinitely.

  Raises:
    status_lib.StatusNotOkError: Failure to stop the node.
  """
  logging.info('Received request to stop node; node uid: %s', node_uid)
  with _PIPELINE_OPS_LOCK:
    with pstate.PipelineState.load(mlmd_handle,
                                   node_uid.pipeline_uid) as pipeline_state:
      nodes = pstate.get_all_pipeline_nodes(pipeline_state.pipeline)
      filtered_nodes = [n for n in nodes if n.node_info.id == node_uid.node_id]
      if len(filtered_nodes) != 1:
        raise status_lib.StatusNotOkError(
            code=status_lib.Code.INTERNAL,
            message=(
                f'`stop_node` operation failed, unable to find node to stop: '
                f'{node_uid}'))
      node = filtered_nodes[0]
      with pipeline_state.node_state_update_context(node_uid) as node_state:
        if node_state.is_stoppable():
          node_state.update(
              pstate.NodeState.STOPPING,
              status_lib.Status(
                  code=status_lib.Code.CANCELLED,
                  message='Cancellation requested by client.'))

  # Wait until the node is stopped or time out.
  _wait_for_node_inactivation(
      pipeline_state, node_uid, timeout_secs=timeout_secs)


@_to_status_not_ok_error
@_pipeline_ops_lock
def resume_manual_node(mlmd_handle: metadata.Metadata,
                       node_uid: task_lib.NodeUid) -> None:
  """Resumes a manual node.

  Args:
    mlmd_handle: A handle to the MLMD db.
    node_uid: Uid of the manual node to be resumed.

  Raises:
    status_lib.StatusNotOkError: Failure to resume a manual node.
  """
  logging.info('Received request to resume manual node; node uid: %s', node_uid)
  with pstate.PipelineState.load(mlmd_handle,
                                 node_uid.pipeline_uid) as pipeline_state:
    nodes = pstate.get_all_pipeline_nodes(pipeline_state.pipeline)
    filtered_nodes = [n for n in nodes if n.node_info.id == node_uid.node_id]
    if len(filtered_nodes) != 1:
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.NOT_FOUND,
          message=(f'Unable to find manual node to resume: {node_uid}'))
    node = filtered_nodes[0]
    node_type = node.node_info.type.name
    if node_type != constants.MANUAL_NODE_TYPE:
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.INVALID_ARGUMENT,
          message=('Unable to resume a non-manual node. '
                   f'Got non-manual node id: {node_uid}'))

  executions = task_gen_utils.get_executions(mlmd_handle, node)
  active_executions = [
      e for e in executions if execution_lib.is_execution_active(e)
  ]
  if not active_executions:
    raise status_lib.StatusNotOkError(
        code=status_lib.Code.NOT_FOUND,
        message=(f'Unable to find active manual node to resume: {node_uid}'))
  if len(active_executions) > 1:
    raise status_lib.StatusNotOkError(
        code=status_lib.Code.INTERNAL,
        message=(f'Unexpected multiple active executions for manual node: '
                 f'{node_uid}'))
  with mlmd_state.mlmd_execution_atomic_op(
      mlmd_handle=mlmd_handle,
      execution_id=active_executions[0].id) as execution:
    completed_state = manual_task_scheduler.ManualNodeState(
        state=manual_task_scheduler.ManualNodeState.COMPLETED)
    completed_state.set_mlmd_value(
        execution.custom_properties.get_or_create(
            manual_task_scheduler.NODE_STATE_PROPERTY_KEY))


@_to_status_not_ok_error
@_pipeline_ops_lock
def _initiate_pipeline_update(
    mlmd_handle: metadata.Metadata,
    pipeline: pipeline_pb2.Pipeline) -> pstate.PipelineState:
  """Initiates pipeline update."""
  pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
  with pstate.PipelineState.load(mlmd_handle, pipeline_uid) as pipeline_state:
    pipeline_state.initiate_update(pipeline)
  return pipeline_state


@_to_status_not_ok_error
def update_pipeline(mlmd_handle: metadata.Metadata,
                    pipeline: pipeline_pb2.Pipeline,
                    timeout_secs: Optional[float] = None) -> None:
  """Updates an active pipeline with a new pipeline IR.

  Initiates a pipeline update operation and waits for it to finish.

  Args:
    mlmd_handle: A handle to the MLMD db.
    pipeline: New pipeline IR to be applied.
    timeout_secs: Timeout in seconds to wait for the update to finish. If
      `None`, waits indefinitely.

  Raises:
    status_lib.StatusNotOkError: Failure to update the pipeline.
  """
  pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
  logging.info('Received request to update pipeline; pipeline uid: %s',
               pipeline_uid)
  pipeline_state = _initiate_pipeline_update(mlmd_handle, pipeline)

  def _is_update_applied() -> bool:
    with pipeline_state:
      if pipeline_state.is_active():
        return not pipeline_state.is_update_initiated()
      # If the pipeline is no longer active, whether or not the update is
      # applied is irrelevant.
      return True

  logging.info('Waiting for pipeline update; pipeline uid: %s', pipeline_uid)
  _wait_for_predicate(_is_update_applied, 'pipeline update', timeout_secs)
  logging.info('Done waiting for pipeline update; pipeline uid: %s',
               pipeline_uid)


def _wait_for_inactivation(mlmd_handle: metadata.Metadata,
                           execution_id: metadata_store_pb2.Execution,
                           timeout_secs: Optional[float]) -> None:
  """Waits for the given execution to become inactive.

  Args:
    mlmd_handle: A handle to the MLMD db.
    execution_id: Id of the execution whose inactivation is awaited.
    timeout_secs: Amount of time in seconds to wait. If `None`, waits
      indefinitely.

  Raises:
    StatusNotOkError: With error code `DEADLINE_EXCEEDED` if execution is not
      inactive after waiting approx. `timeout_secs`.
  """

  def _is_inactivated() -> bool:
    [execution] = mlmd_handle.store.get_executions_by_id([execution_id])
    return not execution_lib.is_execution_active(execution)

  return _wait_for_predicate(_is_inactivated, 'execution inactivation',
                             timeout_secs)


def _wait_for_node_inactivation(pipeline_state: pstate.PipelineState,
                                node_uid: task_lib.NodeUid,
                                timeout_secs: Optional[float]) -> None:
  """Waits for the given node to become inactive.

  Args:
    pipeline_state: Pipeline state.
    node_uid: Uid of the node whose inactivation is awaited.
    timeout_secs: Amount of time in seconds to wait. If `None`, waits
      indefinitely.

  Raises:
    StatusNotOkError: With error code `DEADLINE_EXCEEDED` if node is not
      inactive after waiting approx. `timeout_secs`.
  """

  def _is_inactivated() -> bool:
    with pipeline_state:
      node_state = pipeline_state.get_node_state(node_uid)
      return node_state.state in (pstate.NodeState.COMPLETE,
                                  pstate.NodeState.FAILED,
                                  pstate.NodeState.SKIPPED,
                                  pstate.NodeState.STOPPED)

  return _wait_for_predicate(_is_inactivated, 'node inactivation', timeout_secs)


_POLLING_INTERVAL_SECS = 10.0


def _wait_for_predicate(predicate_fn: Callable[[], bool], waiting_for_desc: str,
                        timeout_secs: Optional[float]) -> None:
  """Waits for `predicate_fn` to return `True` or until timeout seconds elapse."""
  if timeout_secs is None:
    while not predicate_fn():
      time.sleep(_POLLING_INTERVAL_SECS)
    return
  polling_interval_secs = min(_POLLING_INTERVAL_SECS, timeout_secs / 4)
  end_time = time.time() + timeout_secs
  while end_time - time.time() > 0:
    if predicate_fn():
      return
    time.sleep(max(0, min(polling_interval_secs, end_time - time.time())))
  raise status_lib.StatusNotOkError(
      code=status_lib.Code.DEADLINE_EXCEEDED,
      message=(
          f'Timed out ({timeout_secs} secs) waiting for {waiting_for_desc}.'))


@_to_status_not_ok_error
@_pipeline_ops_lock
def orchestrate(mlmd_handle: metadata.Metadata, task_queue: tq.TaskQueue,
                service_job_manager: service_jobs.ServiceJobManager) -> None:
  """Performs a single iteration of the orchestration loop.

  Embodies the core functionality of the main orchestration loop that scans MLMD
  pipeline execution states, generates and enqueues the tasks to be performed.

  Args:
    mlmd_handle: A handle to the MLMD db.
    task_queue: A `TaskQueue` instance into which any tasks will be enqueued.
    service_job_manager: A `ServiceJobManager` instance for handling service
      jobs.

  Raises:
    status_lib.StatusNotOkError: If error generating tasks.
  """
  pipeline_states = _get_pipeline_states(mlmd_handle)
  if not pipeline_states:
    logging.info('No active pipelines to run.')
    return

  active_pipeline_states = []
  stop_initiated_pipeline_states = []
  update_initiated_pipeline_states = []
  for pipeline_state in pipeline_states:
    with pipeline_state:
      if pipeline_state.is_stop_initiated():
        stop_initiated_pipeline_states.append(pipeline_state)
      elif pipeline_state.is_update_initiated():
        update_initiated_pipeline_states.append(pipeline_state)
      elif pipeline_state.is_active():
        active_pipeline_states.append(pipeline_state)
      else:
        raise status_lib.StatusNotOkError(
            code=status_lib.Code.INTERNAL,
            message=(f'Found pipeline (uid: {pipeline_state.pipeline_uid}) '
                     f'which is neither active nor stop-initiated.'))

  for pipeline_state in stop_initiated_pipeline_states:
    logging.info('Orchestrating stop-initiated pipeline: %s',
                 pipeline_state.pipeline_uid)
    _orchestrate_stop_initiated_pipeline(mlmd_handle, task_queue,
                                         service_job_manager, pipeline_state)

  for pipeline_state in update_initiated_pipeline_states:
    logging.info('Orchestrating update-initiated pipeline: %s',
                 pipeline_state.pipeline_uid)
    _orchestrate_update_initiated_pipeline(mlmd_handle, task_queue,
                                           service_job_manager, pipeline_state)

  for pipeline_state in active_pipeline_states:
    logging.info('Orchestrating pipeline: %s', pipeline_state.pipeline_uid)
    _orchestrate_active_pipeline(mlmd_handle, task_queue, service_job_manager,
                                 pipeline_state)


def _get_pipeline_states(
    mlmd_handle: metadata.Metadata) -> List[pstate.PipelineState]:
  """Scans MLMD and returns pipeline states."""
  contexts = pstate.get_orchestrator_contexts(mlmd_handle)
  result = []
  for context in contexts:
    try:
      pipeline_state = pstate.PipelineState.load_from_orchestrator_context(
          mlmd_handle, context)
    except status_lib.StatusNotOkError as e:
      if e.code == status_lib.Code.NOT_FOUND:
        # Ignore any old contexts with no associated active pipelines.
        logging.info(e.message)
        continue
      else:
        raise
    result.append(pipeline_state)
  return result


def _cancel_nodes(mlmd_handle: metadata.Metadata, task_queue: tq.TaskQueue,
                  service_job_manager: service_jobs.ServiceJobManager,
                  pipeline_state: pstate.PipelineState, pause: bool) -> bool:
  """Cancels pipeline nodes and returns `True` if any node is currently active."""
  pipeline = pipeline_state.pipeline
  is_active = False
  for node in pstate.get_all_pipeline_nodes(pipeline):
    if service_job_manager.is_pure_service_node(pipeline_state,
                                                node.node_info.id):
      if not service_job_manager.stop_node_services(pipeline_state,
                                                    node.node_info.id):
        is_active = True
    elif _maybe_enqueue_cancellation_task(
        mlmd_handle, pipeline, node, task_queue, pause=pause):
      is_active = True
    elif service_job_manager.is_mixed_service_node(pipeline_state,
                                                   node.node_info.id):
      if not service_job_manager.stop_node_services(pipeline_state,
                                                    node.node_info.id):
        is_active = True
  return is_active


def _orchestrate_stop_initiated_pipeline(
    mlmd_handle: metadata.Metadata, task_queue: tq.TaskQueue,
    service_job_manager: service_jobs.ServiceJobManager,
    pipeline_state: pstate.PipelineState) -> None:
  """Orchestrates stop initiated pipeline."""
  with pipeline_state:
    stop_reason = pipeline_state.stop_initiated_reason()
  assert stop_reason is not None
  is_active = _cancel_nodes(
      mlmd_handle, task_queue, service_job_manager, pipeline_state, pause=False)
  if not is_active:
    with pipeline_state:
      # Update pipeline execution state in MLMD.
      pipeline_state.set_pipeline_execution_state(
          _mlmd_execution_code(stop_reason))


def _orchestrate_update_initiated_pipeline(
    mlmd_handle: metadata.Metadata, task_queue: tq.TaskQueue,
    service_job_manager: service_jobs.ServiceJobManager,
    pipeline_state: pstate.PipelineState) -> None:
  """Orchestrates an update-initiated pipeline."""
  is_active = _cancel_nodes(
      mlmd_handle, task_queue, service_job_manager, pipeline_state, pause=True)
  if not is_active:
    with pipeline_state:
      pipeline_state.apply_pipeline_update()


@attr.s(auto_attribs=True, kw_only=True)
class _NodeInfo:
  """A convenience container of pipeline node and its state."""
  node: pipeline_pb2.PipelineNode
  state: pstate.NodeState


def _orchestrate_active_pipeline(
    mlmd_handle: metadata.Metadata, task_queue: tq.TaskQueue,
    service_job_manager: service_jobs.ServiceJobManager,
    pipeline_state: pstate.PipelineState) -> None:
  """Orchestrates active pipeline."""
  pipeline = pipeline_state.pipeline
  with pipeline_state:
    assert pipeline_state.is_active()
    if pipeline_state.get_pipeline_execution_state() != (
        metadata_store_pb2.Execution.RUNNING):
      pipeline_state.set_pipeline_execution_state(
          metadata_store_pb2.Execution.RUNNING)
    orchestration_options = pipeline_state.get_orchestration_options()
    logging.info('Orchestration options: %s', orchestration_options)
    deadline_secs = orchestration_options.deadline_secs
    if (pipeline.execution_mode == pipeline_pb2.Pipeline.SYNC and
        deadline_secs > 0 and
        time.time() - pipeline_state.pipeline_creation_time_secs_since_epoch() >
        deadline_secs):
      logging.error(
          'Aborting pipeline due to exceeding deadline (%s secs); '
          'pipeline uid: %s', deadline_secs, pipeline_state.pipeline_uid)
      pipeline_state.initiate_stop(
          status_lib.Status(
              code=status_lib.Code.DEADLINE_EXCEEDED,
              message=('Pipeline aborted due to exceeding deadline '
                       f'({deadline_secs} secs)')))
      return

  def _filter_by_state(node_infos: List[_NodeInfo],
                       state_str: str) -> List[_NodeInfo]:
    return [n for n in node_infos if n.state.state == state_str]

  node_infos = _get_node_infos(pipeline_state)
  stopping_node_infos = _filter_by_state(node_infos, pstate.NodeState.STOPPING)

  # Tracks nodes stopped in the current iteration.
  stopped_node_infos: List[_NodeInfo] = []

  # Create cancellation tasks for nodes in state STOPPING.
  for node_info in stopping_node_infos:
    if service_job_manager.is_pure_service_node(pipeline_state,
                                                node_info.node.node_info.id):
      if service_job_manager.stop_node_services(pipeline_state,
                                                node_info.node.node_info.id):
        stopped_node_infos.append(node_info)
    elif _maybe_enqueue_cancellation_task(mlmd_handle, pipeline, node_info.node,
                                          task_queue):
      pass
    elif service_job_manager.is_mixed_service_node(pipeline_state,
                                                   node_info.node.node_info.id):
      if service_job_manager.stop_node_services(pipeline_state,
                                                node_info.node.node_info.id):
        stopped_node_infos.append(node_info)
    else:
      stopped_node_infos.append(node_info)

  # Change the state of stopped nodes from STOPPING to STOPPED.
  if stopped_node_infos:
    with pipeline_state:
      for node_info in stopped_node_infos:
        node_uid = task_lib.NodeUid.from_pipeline_node(pipeline, node_info.node)
        with pipeline_state.node_state_update_context(node_uid) as node_state:
          node_state.update(pstate.NodeState.STOPPED, node_state.status)

  # Initialize task generator for the pipeline.
  if pipeline.execution_mode == pipeline_pb2.Pipeline.SYNC:
    generator = sync_pipeline_task_gen.SyncPipelineTaskGenerator(
        mlmd_handle,
        task_queue.contains_task_id,
        service_job_manager,
        fail_fast=orchestration_options.fail_fast)
  elif pipeline.execution_mode == pipeline_pb2.Pipeline.ASYNC:
    generator = async_pipeline_task_gen.AsyncPipelineTaskGenerator(
        mlmd_handle, task_queue.contains_task_id, service_job_manager)
  else:
    raise status_lib.StatusNotOkError(
        code=status_lib.Code.FAILED_PRECONDITION,
        message=(
            f'Only SYNC and ASYNC pipeline execution modes supported; '
            f'found pipeline with execution mode: {pipeline.execution_mode}'))

  tasks = generator.generate(pipeline_state)

  with pipeline_state:
    # Handle all the UpdateNodeStateTasks by updating node states.
    for task in tasks:
      if task_lib.is_update_node_state_task(task):
        task = typing.cast(task_lib.UpdateNodeStateTask, task)
        with pipeline_state.node_state_update_context(
            task.node_uid) as node_state:
          node_state.update(task.state, task.status)

    tasks = [t for t in tasks if not task_lib.is_update_node_state_task(t)]

    # If there are still nodes in state STARTING, change them to STARTED.
    for node in pstate.get_all_pipeline_nodes(pipeline_state.pipeline):
      node_uid = task_lib.NodeUid.from_pipeline_node(pipeline_state.pipeline,
                                                     node)
      with pipeline_state.node_state_update_context(node_uid) as node_state:
        if node_state.state == pstate.NodeState.STARTING:
          node_state.update(pstate.NodeState.STARTED)

    for task in tasks:
      if task_lib.is_exec_node_task(task):
        task = typing.cast(task_lib.ExecNodeTask, task)
        task_queue.enqueue(task)
      else:
        assert task_lib.is_finalize_pipeline_task(task)
        assert pipeline.execution_mode == pipeline_pb2.Pipeline.SYNC
        assert len(tasks) == 1
        task = typing.cast(task_lib.FinalizePipelineTask, task)
        if task.status.code == status_lib.Code.OK:
          logging.info('Pipeline run successful; pipeline uid: %s',
                       pipeline_state.pipeline_uid)
        else:
          logging.info('Pipeline run failed; pipeline uid: %s',
                       pipeline_state.pipeline_uid)
        pipeline_state.initiate_stop(task.status)


def _get_node_infos(pipeline_state: pstate.PipelineState) -> List[_NodeInfo]:
  """Returns a list of `_NodeInfo` object for each node in the pipeline."""
  nodes = pstate.get_all_pipeline_nodes(pipeline_state.pipeline)
  result: List[_NodeInfo] = []
  with pipeline_state:
    for node in nodes:
      node_uid = task_lib.NodeUid.from_pipeline_node(pipeline_state.pipeline,
                                                     node)
      result.append(
          _NodeInfo(node=node, state=pipeline_state.get_node_state(node_uid)))
  return result


def _maybe_enqueue_cancellation_task(mlmd_handle: metadata.Metadata,
                                     pipeline: pipeline_pb2.Pipeline,
                                     node: pipeline_pb2.PipelineNode,
                                     task_queue: tq.TaskQueue,
                                     pause: bool = False) -> bool:
  """Enqueues a node cancellation task if not already stopped.

  If the node has an ExecNodeTask in the task queue, issue a cancellation.
  Otherwise, when pause=False, if the node has an active execution in MLMD but
  no ExecNodeTask enqueued, it may be due to orchestrator restart after stopping
  was initiated but before the schedulers could finish. So, enqueue an
  ExecNodeTask with is_cancelled set to give a chance for the scheduler to
  finish gracefully.

  Args:
    mlmd_handle: A handle to the MLMD db.
    pipeline: The pipeline containing the node to cancel.
    node: The node to cancel.
    task_queue: A `TaskQueue` instance into which any cancellation tasks will be
      enqueued.
    pause: Whether the cancellation is to pause the node rather than cancelling
      the execution.

  Returns:
    `True` if a cancellation task was enqueued. `False` if node is already
    stopped or no cancellation was required.
  """
  exec_node_task_id = task_lib.exec_node_task_id_from_pipeline_node(
      pipeline, node)
  if task_queue.contains_task_id(exec_node_task_id):
    task_queue.enqueue(
        task_lib.CancelNodeTask(
            node_uid=task_lib.NodeUid.from_pipeline_node(pipeline, node),
            pause=pause))
    return True
  if not pause:
    executions = task_gen_utils.get_executions(mlmd_handle, node)
    exec_node_task = task_gen_utils.generate_task_from_active_execution(
        mlmd_handle, pipeline, node, executions, is_cancelled=True)
    if exec_node_task:
      task_queue.enqueue(exec_node_task)
      return True
  return False


def _mlmd_execution_code(
    status: status_lib.Status) -> metadata_store_pb2.Execution.State:
  if status.code == status_lib.Code.OK:
    return metadata_store_pb2.Execution.COMPLETE
  elif status.code == status_lib.Code.CANCELLED:
    return metadata_store_pb2.Execution.CANCELED
  return metadata_store_pb2.Execution.FAILED
