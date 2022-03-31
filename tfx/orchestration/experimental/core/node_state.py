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
"""Node state."""

from typing import Optional

import attr
from tfx.proto.orchestration import run_state_pb2
from tfx.utils import json_utils
from tfx.utils import status as status_lib


@attr.s(auto_attribs=True, kw_only=True)
class NodeState(json_utils.Jsonable):
  """Records node state.

  Attributes:
    state: Current state of the node.
    status: Status of the node in state STOPPING or STOPPED.
  """

  STARTING = 'starting'  # Pending work before state can change to STARTED.
  STARTED = 'started'  # Node is ready for execution.
  STOPPING = 'stopping'  # Pending work before state can change to STOPPED.
  STOPPED = 'stopped'  # Node execution is stopped.
  RUNNING = 'running'  # Node is under active execution (i.e. triggered).
  COMPLETE = 'complete'  # Node execution completed successfully.
  SKIPPED = 'skipped'  # Node execution skipped due to conditional.
  # Node execution skipped due to partial run.
  SKIPPED_PARTIAL_RUN = 'skipped_partial_run'
  PAUSING = 'pausing'  # Pending work before state can change to PAUSED.
  PAUSED = 'paused'  # Node was paused and may be resumed in the future.
  FAILED = 'failed'  # Node execution failed due to errors.

  state: str = attr.ib(
      default=STARTED,
      validator=attr.validators.in_([
          STARTING, STARTED, STOPPING, STOPPED, RUNNING, COMPLETE, SKIPPED,
          SKIPPED_PARTIAL_RUN, PAUSING, PAUSED, FAILED
      ]),
      on_setattr=attr.setters.validate)
  status_code: Optional[int] = None
  status_msg: str = ''

  @property
  def status(self) -> Optional[status_lib.Status]:
    if self.status_code is not None:
      return status_lib.Status(code=self.status_code, message=self.status_msg)
    return None

  def update(self,
             state: str,
             status: Optional[status_lib.Status] = None) -> None:
    self.state = state
    if status is not None:
      self.status_code = status.code
      self.status_msg = status.message
    else:
      self.status_code = None
      self.status_msg = ''

  def is_startable(self) -> bool:
    """Returns True if the node can be started."""
    return self.state in set(
        [self.PAUSED, self.STOPPING, self.STOPPED, self.FAILED])

  def is_stoppable(self) -> bool:
    """Returns True if the node can be stopped."""
    return self.state in set(
        [self.STARTING, self.STARTED, self.RUNNING, self.PAUSED])

  def is_pausable(self) -> bool:
    """Returns True if the node can be stopped."""
    return self.state in set([self.STARTING, self.STARTED, self.RUNNING])

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
        state=_NODE_STATE_TO_RUN_STATE_MAP[self.state],
        status_code=status_code_value,
        status_msg=self.status_msg)


def is_node_state_success(state: str) -> bool:
  return state in (NodeState.COMPLETE, NodeState.SKIPPED,
                   NodeState.SKIPPED_PARTIAL_RUN)


def is_node_state_failure(state: str) -> bool:
  return state == NodeState.FAILED


_NODE_STATE_TO_RUN_STATE_MAP = {
    NodeState.STARTING: run_state_pb2.RunState.UNKNOWN,
    NodeState.STARTED: run_state_pb2.RunState.READY,
    NodeState.STOPPING: run_state_pb2.RunState.UNKNOWN,
    NodeState.STOPPED: run_state_pb2.RunState.STOPPED,
    NodeState.RUNNING: run_state_pb2.RunState.RUNNING,
    NodeState.COMPLETE: run_state_pb2.RunState.COMPLETE,
    NodeState.SKIPPED: run_state_pb2.RunState.SKIPPED,
    NodeState.SKIPPED_PARTIAL_RUN: run_state_pb2.RunState.SKIPPED_PARTIAL_RUN,
    NodeState.PAUSING: run_state_pb2.RunState.UNKNOWN,
    NodeState.PAUSED: run_state_pb2.RunState.PAUSED,
    NodeState.FAILED: run_state_pb2.RunState.FAILED
}
