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
"""TaskManager manages the execution and cancellation of tasks."""

from concurrent import futures
import sys
import threading
import traceback
import typing
from typing import Dict, Optional

from absl import logging
from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import post_execution_utils
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import task_queue as tq
from tfx.orchestration.experimental.core import task_scheduler as ts

from tfx.utils import status as status_lib


_MAX_DEQUEUE_WAIT_SECS = 5.0


class Error(Exception):
  """Top-level error for current module."""


class TasksProcessingError(Error):
  """Error that accumulates other errors raised during processing tasks."""

  def __init__(self, errors):
    err_msg = '\n'.join(str(e) for e in errors)
    super().__init__(err_msg)
    self.errors = errors


class _SchedulerWrapper:
  """Wraps a TaskScheduler to store additional details."""

  def __init__(self, task_scheduler: ts.TaskScheduler):
    self._task_scheduler = task_scheduler
    self.pause = False

  def schedule(self) -> ts.TaskSchedulerResult:
    return self._task_scheduler.schedule()

  def cancel(self, cancel_task: task_lib.CancelNodeTask) -> None:
    self.pause = cancel_task.cancel_type == task_lib.NodeCancelType.PAUSE_EXEC
    self._task_scheduler.cancel(cancel_task=cancel_task)


class TaskManager:
  """TaskManager acts on the tasks fetched from the task queues.

  TaskManager instance can be used as a context manager:
  """

  def __init__(self,
               mlmd_handle: metadata.Metadata,
               task_queue: tq.TaskQueue,
               max_active_task_schedulers: int,
               max_dequeue_wait_secs: float = _MAX_DEQUEUE_WAIT_SECS,
               process_all_queued_tasks_before_exit: bool = False):
    """Constructs `TaskManager`.

    Args:
      mlmd_handle: ML metadata db connection.
      task_queue: Task queue.
      max_active_task_schedulers: Maximum number of task schedulers that can be
        active at once.
      max_dequeue_wait_secs: Maximum time to wait when dequeuing if the queue is
        empty.
      process_all_queued_tasks_before_exit: All existing items in the queues are
        processed before exiting the context manager. This is useful for
        deterministic behavior in tests.
    """
    self._mlmd_handle = mlmd_handle
    self._task_queue = task_queue
    self._max_dequeue_wait_secs = max_dequeue_wait_secs
    self._process_all_queued_tasks_before_exit = (
        process_all_queued_tasks_before_exit)

    self._tm_lock = threading.Lock()
    self._stop_event = threading.Event()
    self._scheduler_by_node_uid: Dict[task_lib.NodeUid, _SchedulerWrapper] = {}

    # Async executor for the main task management thread.
    self._main_executor = futures.ThreadPoolExecutor(max_workers=1)
    self._main_future = None

    # Async executor for task schedulers.
    self._ts_executor = futures.ThreadPoolExecutor(
        max_workers=max_active_task_schedulers)
    self._ts_futures = set()

  def __enter__(self):
    if self._main_future is not None:
      raise RuntimeError('TaskManager already started.')
    self._main_future = self._main_executor.submit(self._main)
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    if self._main_future is None:
      raise RuntimeError('TaskManager not started.')
    self._stop_event.set()
    self._main_executor.shutdown()

  def done(self) -> bool:
    """Returns `True` if the main task management thread has exited.

    Raises:
      RuntimeError: If `done` called without entering the task manager context.
    """
    if self._main_future is None:
      raise RuntimeError('Task manager context not entered.')
    return self._main_future.done()

  def exception(self) -> Optional[BaseException]:
    """Returns exception raised by the main task management thread (if any).

    Raises:
      RuntimeError: If `exception` called without entering the task manager
        context or if the main thread is not done (`done` returns `False`).
    """
    if self._main_future is None:
      raise RuntimeError('Task manager context not entered.')
    if not self._main_future.done():
      raise RuntimeError('Task manager main thread not done; call should be '
                         'conditioned on `done` returning `True`.')
    return self._main_future.exception()

  def _main(self) -> None:
    """Runs the main task management loop."""
    try:
      while not self._stop_event.is_set():
        self._cleanup()
        task = self._task_queue.dequeue(self._max_dequeue_wait_secs)
        if task is None:
          continue
        self._handle_task(task)
    finally:
      if self._process_all_queued_tasks_before_exit:
        # Process any remaining tasks from the queue before exiting. This is
        # mainly to make tests deterministic.
        while True:
          task = self._task_queue.dequeue()
          if task is None:
            break
          self._handle_task(task)

      # Final cleanup before exiting. Any exceptions raised here are
      # automatically chained with any raised in the try block.
      self._cleanup(True)

  def _handle_task(self, task: task_lib.Task) -> None:
    """Dispatches task to the task specific handler."""
    if isinstance(task, task_lib.ExecNodeTask):
      self._handle_exec_node_task(task)
    elif isinstance(task, task_lib.CancelNodeTask):
      self._handle_cancel_node_task(task)
    else:
      raise RuntimeError('Cannot dispatch bad task: {}'.format(task))

  def _handle_exec_node_task(self, task: task_lib.ExecNodeTask) -> None:
    """Handles `ExecNodeTask`."""
    logging.info('Handling ExecNodeTask, task-id: %s', task.task_id)
    node_uid = task.node_uid
    with self._tm_lock:
      if node_uid in self._scheduler_by_node_uid:
        raise RuntimeError(
            'Cannot create multiple task schedulers for the same task; '
            'task_id: {}'.format(task.task_id))
      scheduler = _SchedulerWrapper(
          typing.cast(
              ts.TaskScheduler[task_lib.ExecNodeTask],
              ts.TaskSchedulerRegistry.create_task_scheduler(
                  self._mlmd_handle, task.pipeline, task)))
      if task.cancel_type == task_lib.NodeCancelType.PAUSE_EXEC:
        scheduler.pause = True
      self._scheduler_by_node_uid[node_uid] = scheduler
      self._ts_futures.add(
          self._ts_executor.submit(self._process_exec_node_task, scheduler,
                                   task))

  def _handle_cancel_node_task(self, task: task_lib.CancelNodeTask) -> None:
    """Handles `CancelNodeTask`."""
    logging.info('Handling CancelNodeTask, task-id: %s', task.task_id)
    node_uid = task.node_uid
    with self._tm_lock:
      scheduler = self._scheduler_by_node_uid.get(node_uid)
      if scheduler is None:
        logging.info(
            'No task scheduled for node uid: %s. The task might have already '
            'completed before it could be cancelled.', task.node_uid)
      else:
        scheduler.cancel(cancel_task=task)
      self._task_queue.task_done(task)

  def _process_exec_node_task(self, scheduler: _SchedulerWrapper,
                              task: task_lib.ExecNodeTask) -> None:
    """Processes an `ExecNodeTask` using the given task scheduler."""
    # This is a blocking call to the scheduler which can take a long time to
    # complete for some types of task schedulers. The scheduler is expected to
    # handle any internal errors gracefully and return the result with an error
    # status. But in case the scheduler raises an exception, it is considered
    # a failed execution and MLMD is updated accordingly.
    try:
      result = scheduler.schedule()
    except Exception:  # pylint: disable=broad-except
      logging.exception('Exception raised by task scheduler; node uid: %s',
                        task.node_uid)
      result = ts.TaskSchedulerResult(
          status=status_lib.Status(
              code=status_lib.Code.ABORTED,
              message=''.join(traceback.format_exception(*sys.exc_info()))))
    logging.info('For ExecNodeTask id: %s, task-scheduler result status: %s',
                 task.task_id, result.status)
    # If the node was paused, we do not complete the execution as it is expected
    # that a new ExecNodeTask would be issued for resuming the execution.
    if not (scheduler.pause and
            result.status.code == status_lib.Code.CANCELLED):
      post_execution_utils.publish_execution_results_for_task(
          mlmd_handle=self._mlmd_handle, task=task, result=result)
    with self._tm_lock:
      del self._scheduler_by_node_uid[task.node_uid]
      self._task_queue.task_done(task)

  def _cleanup(self, final: bool = False) -> None:
    """Cleans up any remnant effects."""
    if final:
      # Waits for all pending task scheduler futures to complete.
      self._ts_executor.shutdown()
    done_futures = set(fut for fut in self._ts_futures if fut.done())
    self._ts_futures -= done_futures
    exceptions = [fut.exception() for fut in done_futures if fut.exception()]
    if exceptions:
      logging.error('Exception(s) occurred during the pipeline run.')
      for i, e in enumerate(exceptions, start=1):
        logging.error(
            'Exception %d (out of %d):',
            i,
            len(exceptions),
            exc_info=(type(e), e, e.__traceback__))
      raise TasksProcessingError(exceptions)


