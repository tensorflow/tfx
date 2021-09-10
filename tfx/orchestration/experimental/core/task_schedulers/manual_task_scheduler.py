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
"""A task scheduler for Manual system node."""

import threading
import typing
from typing import Optional

import attr
from tfx.orchestration import data_types_utils
from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import mlmd_state
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import task_scheduler
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import json_utils
from tfx.utils import status as status_lib

from ml_metadata.proto import metadata_store_pb2

NODE_STATE_PROPERTY_KEY = '__manual_node_state__'
_POLLING_INTERVAL_SECS = 30


@attr.s(auto_attribs=True, kw_only=True)
class ManualNodeState(json_utils.Jsonable):
  """Manual node's internal state.

  Attributes:
    state: Current state of the manual node.
  """

  # This state indicates that the manual node is waiting for the manual step to
  # be completed.
  WAITING = 'waiting'

  # This state indicates that the manual step has been completed.
  COMPLETED = 'completed'

  state: str = attr.ib(
      default=WAITING, validator=attr.validators.in_([WAITING, COMPLETED]))

  @classmethod
  def from_mlmd_value(
      cls,
      value: Optional[metadata_store_pb2.Value] = None) -> 'ManualNodeState':
    if not value:
      return ManualNodeState()
    node_state_json = data_types_utils.get_metadata_value(value)
    if not node_state_json:
      return ManualNodeState()
    return json_utils.loads(node_state_json)

  def set_mlmd_value(
      self, value: metadata_store_pb2.Value) -> metadata_store_pb2.Value:
    data_types_utils.set_metadata_value(value, json_utils.dumps(self))
    return value


class ManualTaskScheduler(task_scheduler.TaskScheduler):
  """A task scheduler for Manual system node."""

  def __init__(self, mlmd_handle: metadata.Metadata,
               pipeline: pipeline_pb2.Pipeline, task: task_lib.ExecNodeTask):
    super().__init__(mlmd_handle, pipeline, task)
    self._cancel = threading.Event()
    if task.is_cancelled:
      self._cancel.set()

  def schedule(self) -> task_scheduler.TaskSchedulerResult:
    task = typing.cast(task_lib.ExecNodeTask, self.task)
    while not self._cancel.wait(_POLLING_INTERVAL_SECS):
      with mlmd_state.mlmd_execution_atomic_op(
          mlmd_handle=self.mlmd_handle,
          execution_id=task.execution_id) as execution:
        node_state_mlmd_value = execution.custom_properties.get(
            NODE_STATE_PROPERTY_KEY)
        node_state = ManualNodeState.from_mlmd_value(node_state_mlmd_value)
        if node_state.state == ManualNodeState.COMPLETED:
          return task_scheduler.TaskSchedulerResult(
              status=status_lib.Status(code=status_lib.Code.OK),
              output=task_scheduler.ExecutorNodeOutput())

    return task_scheduler.TaskSchedulerResult(
        status=status_lib.Status(code=status_lib.Code.CANCELLED),
        output=task_scheduler.ExecutorNodeOutput())

  def cancel(self) -> None:
    self._cancel.set()
