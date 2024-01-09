# Copyright 2022 Google LLC. All Rights Reserved.
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
"""Utils for user facing SYNC pipeline node execution options.

This is only used for the experimental orchestrator.
"""
import dataclasses
from typing import Optional

from tfx.proto.orchestration import pipeline_pb2


@dataclasses.dataclass
class NodeExecutionOptions:
  """Component Node Execution Options.

  Currently only apply in experimental orchestrator.
  """
  trigger_strategy: pipeline_pb2.NodeExecutionOptions.TriggerStrategy = (
      pipeline_pb2.NodeExecutionOptions.TRIGGER_STRATEGY_UNSPECIFIED)
  success_optional: bool = False
  max_execution_retries: Optional[int] = None
  execution_timeout_sec: int = 0
  # This feature enable users to choose if they want to reuse or reset the
  # stateful working dir from the previously failed execution. If set False
  # (which is the default), previous stateful working dir (if exists) will be
  # reused. If set True, previous stateful working dir will NOT be reused and a
  # new stateful working dir will be created for every new execution.
  reset_stateful_working_dir: bool = False

  # This is an experimental feature to enable "end nodes" in a pipeline to
  # support resource lifetimes. If this field is set then the node which this
  # NodeExecutionOptions belongs to will run during pipeline finalization if the
  # "lifetime_start" has run succesfully.
  # Pipeline finalization happens when:
  # 1. All nodes in the pipeline completed, this is the "happy path".
  # 2. A user requests for the pipeline to stop
  # 3. A node fails in the pipeline and it cannot continue executing.
  # This should be the id of the node "starting" a lifetime.
  # If you want to use this feature please contact kmonte@ first.
  lifetime_start: Optional[str] = None

  def __post_init__(self):
    if self.max_execution_retries is not None:
      self.max_execution_retries = max(self.max_execution_retries, 0)
