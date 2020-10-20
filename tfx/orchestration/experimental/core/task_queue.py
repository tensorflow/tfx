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
from typing import Optional, Text, Type, TypeVar

import attr
from tfx.orchestration.experimental.core.proto import task_pb2
from tfx.proto.orchestration import pipeline_pb2

_TASK_TYPE = 'task_type'
_EXEC_TASK = 'exec_task'
_NODE_OR_SUBP_ID = 'node_or_sub_pipeline_id'
_NODE_ID = 'node_id'

_T = TypeVar('_T', bound='_ExecTaskId')


@attr.s(frozen=True)
class _ExecTaskId:
  """Unique identifier for an `ExecTask`."""
  node_id = attr.ib(type=Text)
  pipeline_id = attr.ib(type=Text)
  pipeline_run_id = attr.ib(type=Optional[Text])

  @classmethod
  def from_exec_task(cls: Type[_T], exec_task: task_pb2.ExecTask) -> _T:
    """Creates an instance from `ExecTask`."""
    if exec_task.WhichOneof(_NODE_OR_SUBP_ID) != _NODE_ID:
      raise ValueError('Supported exec task id type: {}'.format(_NODE_ID))
    return cls(
        node_id=exec_task.node_id,
        pipeline_id=exec_task.pipeline_id,
        pipeline_run_id=exec_task.pipeline_run_id or None)

  @classmethod
  def from_pipeline_node(cls: Type[_T], pipeline: pipeline_pb2.Pipeline,
                         node: pipeline_pb2.PipelineNode) -> _T:
    """Creates an instance from pipeline and node definitions."""
    pipeline_run_id = (
        pipeline.runtime_spec.pipeline_run_id.field_value.string_value
        if pipeline.runtime_spec.HasField('pipeline_run_id') else None)
    return cls(
        node_id=node.node_info.id,
        pipeline_id=pipeline.pipeline_info.id,
        pipeline_run_id=pipeline_run_id)


_U = TypeVar('_U', bound='TaskId')


@attr.s(frozen=True)
class TaskId:
  """Unique identifier for a `Task`."""
  exec_task_id = attr.ib(type=Optional[_ExecTaskId])

  @classmethod
  def from_task(cls: Type[_U], task: task_pb2.Task) -> _U:
    """Creates an instance from `Task`."""
    if task.WhichOneof(_TASK_TYPE) != _EXEC_TASK:
      raise ValueError('Task type supported: `{}`'.format(_EXEC_TASK))
    return cls(exec_task_id=_ExecTaskId.from_exec_task(task.exec_task))

  @classmethod
  def from_pipeline_node(cls: Type[_U], pipeline: pipeline_pb2.Pipeline,
                         node: pipeline_pb2.PipelineNode) -> _U:
    """Creates an instance from pipeline and node definitions."""
    return cls(exec_task_id=_ExecTaskId.from_pipeline_node(pipeline, node))


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

  def enqueue(self, task: task_pb2.Task) -> bool:
    """Enqueues the given task if no prior task with the same id exists.

    Args:
      task: A `Task` proto.

    Returns:
      `True` if the task could be enqueued. `False` if a task with the same id
      already exists.
    """
    with self._lock:
      task_id = TaskId.from_task(task)
      if task_id in self._task_ids:
        return False
      self._task_ids.add(task_id)
      self._task_queue.append((task_id, task))
      return True

  def dequeue(self) -> Optional[task_pb2.Task]:
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

  def task_done(self, task: task_pb2.Task) -> None:
    """Marks a task as done.

    Consumers should call this method after the task is processed.

    Args:
      task: A `Task` proto.

    Raises:
      RuntimeError: If attempt is made to mark a non-existent or non-dequeued
      task as done.
    """
    with self._lock:
      task_id = TaskId.from_task(task)
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

  def is_task_id_tracked(self, task_id: TaskId) -> bool:
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
