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

import base64
import copy
import functools
import threading
import time
from typing import List, Optional, Sequence, Text

from absl import logging
import attr
from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import async_pipeline_task_gen
from tfx.orchestration.experimental.core import status as status_lib
from tfx.orchestration.experimental.core import sync_pipeline_task_gen
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import task_gen
from tfx.orchestration.experimental.core import task_gen_utils
from tfx.orchestration.experimental.core import task_queue as tq
from tfx.orchestration.portable.mlmd import common_utils
from tfx.orchestration.portable.mlmd import context_lib
from tfx.orchestration.portable.mlmd import execution_lib
from tfx.proto.orchestration import pipeline_pb2

from ml_metadata.proto import metadata_store_pb2

_ORCHESTRATOR_RESERVED_ID = '__ORCHESTRATOR__'
_PIPELINE_IR = 'pipeline_ir'
_STOP_INITIATED = 'stop_initiated'
_ORCHESTRATOR_EXECUTION_TYPE = metadata_store_pb2.ExecutionType(
    name=_ORCHESTRATOR_RESERVED_ID,
    properties={_PIPELINE_IR: metadata_store_pb2.STRING})

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
    pipeline: pipeline_pb2.Pipeline) -> metadata_store_pb2.Execution:
  """Initiates a pipeline start operation.

  Upon success, MLMD is updated to signal that the given pipeline must be
  started.

  Args:
    mlmd_handle: A handle to the MLMD db.
    pipeline: IR of the pipeline to start.

  Returns:
    The pipeline-level MLMD execution proto upon success.

  Raises:
    status_lib.StatusNotOkError: Failure to initiate pipeline start or if
      execution is not inactive after waiting `timeout_secs`.
  """
  pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
  context = context_lib.register_context_if_not_exists(
      mlmd_handle,
      context_type_name=_ORCHESTRATOR_RESERVED_ID,
      context_name=_orchestrator_context_name(pipeline_uid))

  executions = mlmd_handle.store.get_executions_by_context(context.id)
  if any(e for e in executions if execution_lib.is_execution_active(e)):
    raise status_lib.StatusNotOkError(
        code=status_lib.Code.ALREADY_EXISTS,
        message=f'Pipeline with uid {pipeline_uid} already started.')

  execution = execution_lib.prepare_execution(
      mlmd_handle,
      _ORCHESTRATOR_EXECUTION_TYPE,
      metadata_store_pb2.Execution.NEW,
      exec_properties={
          _PIPELINE_IR:
              base64.b64encode(pipeline.SerializeToString()).decode('utf-8')
      })
  execution = execution_lib.put_execution(mlmd_handle, execution, [context])
  logging.info('Registered execution (id: %s) for the pipeline with uid: %s',
               execution.id, pipeline_uid)
  return execution


DEFAULT_WAIT_FOR_INACTIVATION_TIMEOUT_SECS = 120.0


def stop_pipeline(
    mlmd_handle: metadata.Metadata,
    pipeline_uid: task_lib.PipelineUid,
    timeout_secs: float = DEFAULT_WAIT_FOR_INACTIVATION_TIMEOUT_SECS) -> None:
  """Stops a pipeline.

  Initiates a pipeline stop operation and waits for the pipeline execution to be
  gracefully stopped in the orchestration loop.

  Args:
    mlmd_handle: A handle to the MLMD db.
    pipeline_uid: Uid of the pipeline to be stopped.
    timeout_secs: Amount of time in seconds to wait for pipeline to stop.

  Raises:
    status_lib.StatusNotOkError: Failure to initiate pipeline stop.
  """
  execution = _initiate_pipeline_stop(mlmd_handle, pipeline_uid)
  _wait_for_inactivation(mlmd_handle, execution, timeout_secs=timeout_secs)


@_to_status_not_ok_error
@_pipeline_ops_lock
def _initiate_pipeline_stop(
    mlmd_handle: metadata.Metadata,
    pipeline_uid: task_lib.PipelineUid) -> metadata_store_pb2.Execution:
  """Initiates a pipeline stop operation.

  Upon success, MLMD is updated to signal that the pipeline given by
  `pipeline_uid` must be stopped.

  Args:
    mlmd_handle: A handle to the MLMD db.
    pipeline_uid: Uid of the pipeline to be stopped.

  Returns:
    The pipeline-level MLMD execution proto upon success.

  Raises:
    status_lib.StatusNotOkError: Failure to initiate pipeline stop.
  """
  context = mlmd_handle.store.get_context_by_type_and_name(
      type_name=_ORCHESTRATOR_RESERVED_ID,
      context_name=_orchestrator_context_name(pipeline_uid))
  if not context:
    raise status_lib.StatusNotOkError(
        code=status_lib.Code.NOT_FOUND,
        message=f'No active pipeline with uid {pipeline_uid} to stop.')

  executions = mlmd_handle.store.get_executions_by_context(context.id)
  active_executions = [
      e for e in executions if execution_lib.is_execution_active(e)
  ]
  if not active_executions:
    if executions:
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.ALREADY_EXISTS,
          message=f'Pipeline with uid {pipeline_uid} already stopped.')
    else:
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.NOT_FOUND,
          message=f'No active pipeline with uid {pipeline_uid} to stop.')
  if len(active_executions) > 1:
    raise status_lib.StatusNotOkError(
        code=status_lib.Code.INTERNAL,
        message=(f'Error while stopping pipeline uid {pipeline_uid}: found '
                 f'{len(active_executions)} active executions, expected 1.'))
  result = active_executions[0]
  _set_stop_initiated_property(result)
  mlmd_handle.store.put_executions([result])
  return result


@_to_status_not_ok_error
def _wait_for_inactivation(
    mlmd_handle: metadata.Metadata,
    execution: metadata_store_pb2.Execution,
    timeout_secs: float = DEFAULT_WAIT_FOR_INACTIVATION_TIMEOUT_SECS) -> None:
  """Waits for the given execution to become inactive.

  Args:
    mlmd_handle: A handle to the MLMD db.
    execution: Execution whose inactivation is waited.
    timeout_secs: Amount of time in seconds to wait.

  Raises:
    StatusNotOkError: With error code `DEADLINE_EXCEEDED` if execution is not
      inactive after waiting approx. `timeout_secs`.
  """
  polling_interval_secs = min(10.0, timeout_secs / 4)
  end_time = time.time() + timeout_secs
  while end_time - time.time() > 0:
    updated_executions = mlmd_handle.store.get_executions_by_id(
        [execution.id])
    if not execution_lib.is_execution_active(updated_executions[0]):
      return
    time.sleep(max(0, min(polling_interval_secs, end_time - time.time())))
  raise status_lib.StatusNotOkError(
      code=status_lib.Code.DEADLINE_EXCEEDED,
      message=(f'Timed out ({timeout_secs} secs) waiting for execution '
               f'inactivation.'))


@attr.s
class _PipelineDetail:
  """Helper data class for `generate_tasks`."""
  context = attr.ib(type=metadata_store_pb2.Context)
  execution = attr.ib(type=metadata_store_pb2.Execution)
  pipeline = attr.ib(type=pipeline_pb2.Pipeline)
  pipeline_uid = attr.ib(type=task_lib.PipelineUid)
  stop_initiated = attr.ib(type=bool, default=False)
  generator = attr.ib(type=Optional[task_gen.TaskGenerator], default=None)


@_to_status_not_ok_error
@_pipeline_ops_lock
def generate_tasks(mlmd_handle: metadata.Metadata,
                   task_queue: tq.TaskQueue) -> None:
  """Generates and enqueues tasks to be performed.

  Embodies the core functionality of the main orchestration loop that scans MLMD
  pipeline execution states, generates and enqueues the tasks to be performed.

  Args:
    mlmd_handle: A handle to the MLMD db.
    task_queue: A `TaskQueue` instance into which any tasks will be enqueued.

  Raises:
    status_lib.StatusNotOkError: If error generating tasks.
  """
  pipeline_details = _get_pipeline_details(mlmd_handle, task_queue)
  if not pipeline_details:
    logging.info('No active pipelines to run.')
    return

  active_pipeline_details = []
  stop_initiated_pipeline_details = []
  for detail in pipeline_details:
    if detail.stop_initiated:
      stop_initiated_pipeline_details.append(detail)
    elif execution_lib.is_execution_active(detail.execution):
      active_pipeline_details.append(detail)
    else:
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.INTERNAL,
          message=(f'Found pipeline (uid: {detail.pipeline_uid}) which is '
                   f'neither active nor stop-initiated.'))

  if stop_initiated_pipeline_details:
    logging.info(
        'Stop-initiated pipeline uids:\n%s', '\n'.join(
            str(detail.pipeline_uid)
            for detail in stop_initiated_pipeline_details))
    _process_stop_initiated_pipelines(mlmd_handle, task_queue,
                                      stop_initiated_pipeline_details)
  if active_pipeline_details:
    logging.info(
        'Active (excluding stop-initiated) pipeline uids:\n%s', '\n'.join(
            str(detail.pipeline_uid) for detail in active_pipeline_details))
    _process_active_pipelines(mlmd_handle, task_queue, active_pipeline_details)


def _get_pipeline_details(mlmd_handle: metadata.Metadata,
                          task_queue: tq.TaskQueue) -> List[_PipelineDetail]:
  """Scans MLMD and returns pipeline details."""
  result = []

  contexts = mlmd_handle.store.get_contexts_by_type(_ORCHESTRATOR_RESERVED_ID)

  for context in contexts:
    active_executions = [
        e for e in mlmd_handle.store.get_executions_by_context(context.id)
        if execution_lib.is_execution_active(e)
    ]
    if len(active_executions) > 1:
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.INTERNAL,
          message=(f'Expected 1 but found {len(active_executions)} active '
                   f'executions for context named: {context.name}'))
    if not active_executions:
      continue
    execution = active_executions[0]

    # TODO(goutham): Instead of parsing the pipeline IR each time, we could
    # cache the parsed pipeline IR in `initiate_pipeline_start` and reuse it.
    pipeline_ir_b64 = common_utils.get_metadata_value(
        execution.properties[_PIPELINE_IR])
    pipeline = pipeline_pb2.Pipeline()
    pipeline.ParseFromString(base64.b64decode(pipeline_ir_b64))

    stop_initiated = _is_stop_initiated(execution)

    if stop_initiated:
      generator = None
    else:
      if pipeline.execution_mode == pipeline_pb2.Pipeline.SYNC:
        generator = sync_pipeline_task_gen.SyncPipelineTaskGenerator(
            mlmd_handle, pipeline, task_queue.contains_task_id)
      elif pipeline.execution_mode == pipeline_pb2.Pipeline.ASYNC:
        generator = async_pipeline_task_gen.AsyncPipelineTaskGenerator(
            mlmd_handle, pipeline, task_queue.contains_task_id)
      else:
        raise status_lib.StatusNotOkError(
            code=status_lib.Code.FAILED_PRECONDITION,
            message=(
                f'Only SYNC and ASYNC pipeline execution modes supported; '
                f'found pipeline with execution mode: {pipeline.execution_mode}'
            ))

    result.append(
        _PipelineDetail(
            context=context,
            execution=execution,
            pipeline=pipeline,
            pipeline_uid=task_lib.PipelineUid.from_pipeline(pipeline),
            stop_initiated=stop_initiated,
            generator=generator))

  return result


def _process_stop_initiated_pipelines(
    mlmd_handle: metadata.Metadata, task_queue: tq.TaskQueue,
    pipeline_details: Sequence[_PipelineDetail]) -> None:
  """Processes stop initiated pipelines."""
  for detail in pipeline_details:
    has_active_executions = False
    for node in _get_all_pipeline_nodes(detail.pipeline):
      # If the node has an ExecNodeTask in the task queue, issue a cancellation.
      # Otherwise, if the node has an active execution in MLMD but no
      # ExecNodeTask enqueued, it may be due to orchestrator restart after
      # pipeline stop was initiated but before the schedulers could finish. So,
      # enqueue an ExecNodeTask with is_cancelled set to give a chance for the
      # scheduler to finish gracefully.
      exec_node_task_id = task_lib.exec_node_task_id_from_pipeline_node(
          detail.pipeline, node)
      if task_queue.contains_task_id(exec_node_task_id):
        task_queue.enqueue(
            task_lib.CancelNodeTask(
                node_uid=task_lib.NodeUid.from_pipeline_node(
                    detail.pipeline, node)))
        has_active_executions = True
      else:
        executions = task_gen_utils.get_executions(mlmd_handle, node)
        exec_node_task = task_gen_utils.generate_task_from_active_execution(
            mlmd_handle, detail.pipeline, node, executions, is_cancelled=True)
        if exec_node_task:
          task_queue.enqueue(exec_node_task)
          has_active_executions = True
    if not has_active_executions:
      updated_execution = copy.deepcopy(detail.execution)
      updated_execution.last_known_state = metadata_store_pb2.Execution.CANCELED
      mlmd_handle.store.put_executions([updated_execution])


def _process_active_pipelines(
    mlmd_handle: metadata.Metadata, task_queue: tq.TaskQueue,
    pipeline_details: Sequence[_PipelineDetail]) -> None:
  """Processes active pipelines."""
  for detail in pipeline_details:
    assert detail.execution.last_known_state in (
        metadata_store_pb2.Execution.NEW, metadata_store_pb2.Execution.RUNNING)
    if (detail.execution.last_known_state !=
        metadata_store_pb2.Execution.RUNNING):
      updated_execution = copy.deepcopy(detail.execution)
      updated_execution.last_known_state = metadata_store_pb2.Execution.RUNNING
      mlmd_handle.store.put_executions([updated_execution])

    # TODO(goutham): Consider concurrent task generation.
    tasks = detail.generator.generate()
    for task in tasks:
      task_queue.enqueue(task)


# TODO(goutham): Handle sync pipelines.
def _orchestrator_context_name(pipeline_uid: task_lib.PipelineUid) -> Text:
  """Returns orchestrator reserved context name."""
  return f'{_ORCHESTRATOR_RESERVED_ID}_{pipeline_uid.pipeline_id}'


# TODO(goutham): Handle sync pipelines.
def _pipeline_uid_from_context(
    context: metadata_store_pb2.Context) -> task_lib.PipelineUid:
  """Returns pipeline uid from orchestrator reserved context."""
  pipeline_id = context.name.split(_ORCHESTRATOR_RESERVED_ID + '_')[1]
  return task_lib.PipelineUid(pipeline_id=pipeline_id, pipeline_run_id=None)


def _set_stop_initiated_property(
    execution: metadata_store_pb2.Execution) -> None:
  common_utils.set_metadata_value(execution.custom_properties[_STOP_INITIATED],
                                  1)


def _is_stop_initiated(execution: metadata_store_pb2.Execution) -> bool:
  if _STOP_INITIATED in execution.custom_properties:
    return common_utils.get_metadata_value(
        execution.custom_properties[_STOP_INITIATED]) == 1
  return False


def _get_all_pipeline_nodes(
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
