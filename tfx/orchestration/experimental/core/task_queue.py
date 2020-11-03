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

import collections
import threading
from typing import Optional

from tfx.orchestration.experimental.core import task as task_lib


class TaskQueue:
  """A thread-safe task queue.

  The life-cycle of a task starts with producers calling `enqueue`. Consumers
  call `dequeue` to obtain the tasks in FIFO order. When processing is complete,
  consumers must release the tasks by calling `task_done`.
  """

  def __init__(self):
    self._lock = threading.Lock()
    self._task_ids = set()
    self._task_queue = collections.deque()
    self._pending_tasks_by_id = {}

  def enqueue(self, task: task_lib.Task) -> bool:
    """Enqueues the given task if no prior task with the same id exists.

    Args:
      task: A `Task` proto.

    Returns:
      `True` if the task could be enqueued. `False` if a task with the same id
      already exists.
    """
    with self._lock:
      task_id = task.task_id
      if task_id in self._task_ids:
        return False
      self._task_ids.add(task_id)
      self._task_queue.append((task_id, task))
      return True

  def dequeue(self) -> Optional[task_lib.Task]:
    """Removes and returns a task from the queue.

    Once the processing is complete, queue consumers must call `task_done`.

    Returns:
      A `Task` or `None` if the queue is empty.
    """
    with self._lock:
      if not self._task_queue:
        return None
      task_id, task = self._task_queue.popleft()
      self._pending_tasks_by_id[task_id] = task
      return task

  def task_done(self, task: task_lib.Task) -> None:
    """Marks a task as done.

    Consumers should call this method after the task is processed.

    Args:
      task: A `Task` proto.

    Raises:
      RuntimeError: If attempt is made to mark a non-existent or non-dequeued
      task as done.
    """
    with self._lock:
      task_id = task.task_id
      if task_id not in self._pending_tasks_by_id:
        if task_id in self._task_ids:
          raise RuntimeError(
              'Must call `dequeue` before calling `task_done`; task: {}'.format(
                  task))
        else:
          raise RuntimeError(
              'Task not tracked by task queue; task: {}'.format(task))
      self._pending_tasks_by_id.pop(task_id)
      self._task_ids.remove(task_id)

  def is_task_id_tracked(self, task_id: task_lib.TaskId) -> bool:
    """Returns `True` if a task with given `task_id` is tracked.

    The task is considered "tracked" if it has been `enqueue`d, probably
    `dequeue`d but `task_done` has not been called.

    Args:
      task_id: An instance of `TaskId` representing the task to be checked.

    Returns:
      `True` if task with given `task_id` is tracked.
    """
    with self._lock:
      return task_id in self._task_ids

  def is_empty(self) -> bool:
    """Returns `True` if the task queue is empty."""
    return not self._task_ids
