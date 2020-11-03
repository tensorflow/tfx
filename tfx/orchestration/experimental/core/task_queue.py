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
"""Task queue."""

import queue
import threading
from typing import Optional

from tfx.orchestration.experimental.core import task as task_lib


class TaskQueue:
  """A thread-safe task queue with duplicate detection.

  The life-cycle of a task starts with producers calling `enqueue`. Consumers
  call `dequeue` to obtain the tasks in FIFO order. When processing is complete,
  consumers must release the tasks by calling `task_done`.
  """

  def __init__(self):
    self._lock = threading.Lock()
    self._task_ids = set()
    # Note: the TaskQueue implementation relies on the queue being unbounded.
    # This must not change without revising the implementation.
    self._queue = queue.Queue()
    self._pending_tasks_by_id = {}

  def enqueue(self, task: task_lib.Task) -> bool:
    """Enqueues the given task if no prior task with the same id exists.

    Args:
      task: A `Task` object.

    Returns:
      `True` if the task could be enqueued. `False` if a task with the same id
      already exists.
    """
    task_id = task.task_id
    with self._lock:
      if task_id in self._task_ids:
        return False
      self._task_ids.add(task_id)
      self._queue.put((task_id, task))
    return True

  def dequeue(self,
              max_wait_secs: Optional[float] = None) -> Optional[task_lib.Task]:
    """Removes and returns a task from the queue.

    Once the processing is complete, queue consumers must call `task_done`.

    Args:
      max_wait_secs: If not `None`, waits a maximum of `max_wait_secs` when the
        queue is empty for a task to be enqueued. If no task is present in the
        queue after the wait, `None` is returned. If `max_wait_secs` is `None`
        (default), returns `None` without waiting when the queue is empty.

    Returns:
      A `Task` or `None` if the queue is empty.
    """
    try:
      task_id, task = self._queue.get(
          block=max_wait_secs is not None, timeout=max_wait_secs)
    except queue.Empty:
      return None
    with self._lock:
      self._pending_tasks_by_id[task_id] = task
    return task

  def task_done(self, task: task_lib.Task) -> None:
    """Marks the processing of a task as done.

    Consumers should call this method after the task is processed.

    Args:
      task: A `Task` object.

    Raises:
      RuntimeError: If attempt is made to mark a non-existent or non-dequeued
      task as done.
    """
    task_id = task.task_id
    with self._lock:
      if task_id not in self._pending_tasks_by_id:
        if task_id in self._task_ids:
          raise RuntimeError(
              'Must call `dequeue` before calling `task_done`; task id: {}'
              .format(task_id))
        else:
          raise RuntimeError(
              'Task not present in the queue; task id: {}'.format(task_id))
      self._pending_tasks_by_id.pop(task_id)
      self._task_ids.remove(task_id)

  def contains_task_id(self, task_id: task_lib.TaskId) -> bool:
    """Returns `True` if the task queue contains a task with the given `task_id`.

    Args:
      task_id: A task id.

    Returns:
      `True` if a task with `task_id` was enqueued but `task_done` has not been
      invoked yet.
    """
    with self._lock:
      return task_id in self._task_ids

  def is_empty(self) -> bool:
    """Returns `True` if the task queue is empty.

    Queue is considered empty only if any enqueued tasks have been dequeued and
    `task_done` invoked on them.
    """
    with self._lock:
      return not self._task_ids
