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

from tfx.proto.orchestration import pipeline_pb2


@dataclasses.dataclass
class NodeExecutionOptions:
  """Component Node Execution Options.

  Currently only apply in experimental orchestrator.
  """
  trigger_strategy: pipeline_pb2.NodeExecutionOptions.TriggerStrategy = (
      pipeline_pb2.NodeExecutionOptions.TRIGGER_STRATEGY_UNSPECIFIED)
  success_optional: bool = False
  max_execution_retries: int = 0
  execution_timeout_sec: int = 0

  def __post_init__(self):
    self.max_execution_retries = max(self.max_execution_retries, 0)
