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
import threading
import typing

from absl import logging
from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import task_queue as tq
from tfx.orchestration.experimental.core import task_scheduler as ts
from tfx.proto.orchestration import pipeline_pb2

_MAX_DEQUEUE_WAIT_SECS = 5.0


class TaskManager:
  """TaskManager acts on the tasks fetched from the task queues.

  TaskManager instance can be used as a context manager:
  """

  def __init__(self,
               mlmd_handle: metadata.Metadata,
               pipeline: pipeline_pb2.Pipeline,
               task_queue: tq.TaskQueue,
               max_active_task_schedulers: int,
               max_dequeue_wait_secs: float = _MAX_DEQUEUE_WAIT_SECS,
               process_all_queued_tasks_before_exit: bool = False):
    """Constructs `TaskManager`.

    Args:
      mlmd_handle: ML metadata db connection.
      pipeline: A pipeline IR proto.
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
    self._pipeline = pipeline
    self._task_queue = task_queue
    self._max_dequeue_wait_secs = max_dequeue_wait_secs
    self._process_all_queued_tasks_before_exit = (
        process_all_queued_tasks_before_exit)

    self._tm_lock = threading.Lock()
    self._tm_thread = None  # Initialized when entering context.
    self._stop_event = threading.Event()
    self._ts_executor = futures.ThreadPoolExecutor(
        max_workers=max_active_task_schedulers)
    self._scheduler_by_node_uid = {}

  def __enter__(self):
    if self._tm_thread is not None:
      raise RuntimeError('TaskManager already started.')
    self._ts_executor.__enter__()
    self._tm_thread = threading.Thread(target=self._process_tasks)
    self._tm_thread.start()

  def __exit__(self, exc_type, exc_val, exc_tb):
    if self._tm_thread is None:
      raise RuntimeError('TaskManager not started.')
    self._stop_event.set()
    self._tm_thread.join()
    self._ts_executor.__exit__(exc_type, exc_val, exc_tb)

  def _process_tasks(self) -> None:
    """Processes tasks from the multiplexed queue."""
    while not self._stop_event.is_set():
      task = self._task_queue.dequeue(self._max_dequeue_wait_secs)
      if task is None:
        continue
      self._dispatch_task(task)
    if self._process_all_queued_tasks_before_exit:
      # Process any remaining tasks from the queue before exiting. This is
      # mainly to make tests deterministic.
      while True:
        task = self._task_queue.dequeue()
        if task is None:
          break
        self._dispatch_task(task)

  def _dispatch_task(self, task: task_lib.Task) -> None:
    """Dispatches task to the right handler."""
    if task_lib.is_exec_node_task(task):
      self._handle_exec(typing.cast(task_lib.ExecNodeTask, task))
    elif task_lib.is_cancel_node_task(task):
      self._handle_cancel(typing.cast(task_lib.CancelNodeTask, task))
    else:
      raise RuntimeError('Cannot dispatch bad task: {}'.format(task))

  def _handle_exec(self, task: task_lib.ExecNodeTask) -> None:
    """Handles execution task."""
    node_uid = task.node_uid
    with self._tm_lock:
      if node_uid in self._scheduler_by_node_uid:
        raise RuntimeError(
            'Cannot create multiple task schedulers for the same task; '
            'task_id: {}'.format(task.task_id))
      scheduler = ts.TaskSchedulerRegistry.create_task_scheduler(
          self._mlmd_handle, self._pipeline, task)
      self._scheduler_by_node_uid[node_uid] = scheduler
      self._ts_executor.submit(self._schedule, scheduler, task)

  def _handle_cancel(self, task: task_lib.CancelNodeTask) -> None:
    """Handles cancellation task."""
    node_uid = task.node_uid
    with self._tm_lock:
      scheduler = self._scheduler_by_node_uid.get(node_uid)
      if scheduler is None:
        logging.info(
            'No task scheduled for task id: %s. The task might have '
            'already completed before it could be cancelled.', task.task_id)
        return
      scheduler.cancel()
      self._task_queue.task_done(task)

  def _schedule(self, scheduler: ts.TaskScheduler,
                task: task_lib.ExecNodeTask) -> None:
    """Schedules task execution using the given task scheduler."""
    # This is a blocking call to the scheduler which can take a long time to
    # complete for some types of task schedulers.
    # TODO(goutham): Handle any exceptions raised by the scheduler.
    result = scheduler.schedule()
    _publish_execution_results(self._mlmd_handle, self._pipeline, task, result)
    with self._tm_lock:
      del self._scheduler_by_node_uid[task.node_uid]
      self._task_queue.task_done(task)


def _publish_execution_results(mlmd_handle: metadata.Metadata,
                               pipeline: pipeline_pb2.Pipeline,
                               task: task_lib.ExecNodeTask,
                               result: ts.TaskSchedulerResult) -> None:
  # TODO(goutham): Implement this.
  del mlmd_handle, pipeline, task, result
