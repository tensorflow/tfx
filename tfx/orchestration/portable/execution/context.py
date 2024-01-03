# Copyright 2023 Google LLC. All Rights Reserved.
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
"""Runtime context dataclass for Python executables."""
from typing import Optional
import attr

from tfx.orchestration.portable import data_types


@attr.define(init=False)
class ExecutionContext:
  """Dataclass that stores public execution context.

  This is the single public context class to access ExecutionInvocation and
  runtime information of a node execution. This contains similar fields from
  `ph.execution_invocation()` or `ph.runtime_info()`, but is meant to be used
  during the component execution, not from the pipeline authoring DSL.

  An ExecutionContext instance is typically dependency injected from the
  runtime, and should not be directly instantiated by the users.
  """
  # LINT.IfChange
  # The Execution id that is registered in MLMD.
  execution_id: Optional[int] = None
  # Stateful working dir will be deterministic given pipeline, node and run_id.
  # The typical usecase is to restore long running executor's state after
  # eviction. For examples, a Trainer can use this directory to store
  # checkpoints. This dir is undefined when Launcher.launch() is done.
  stateful_working_dir: Optional[str] = None
  # A tempory dir for executions and it is expected to be cleared up at the end
  # of executions in both success and failure cases. This dir is undefined when
  # Launcher.launch() is done.
  tmp_dir: Optional[str] = None
  # The config of this Node.
  node_id: Optional[str] = None
  # The config of the pipeline that this node is running in.
  pipeline_id: Optional[str] = None
  # The id of the pipeline run that this execution is in.
  pipeline_run_id: Optional[str] = None
  # The id of the pipeline run for the top-level pipeline in this execution. If
  # the top-level pipeline is ASYNC then this will be the empty string.
  top_level_pipeline_run_id: Optional[str] = None
  # URL to the Tflex frontend for this pipeline/run.
  frontend_url: Optional[str] = None
  # LINT.ThenChange(../../../proto/orchestration/execution_invocation.proto)

  def __init__(self, exec_info: data_types.ExecutionInfo, **unused_kwargs):
    del unused_kwargs
    self.execution_id = exec_info.execution_id
    self.stateful_working_dir = exec_info.stateful_working_dir
    self.tmp_dir = exec_info.tmp_dir
    self.node_id = (
        exec_info.pipeline_node.node_info.id
        if exec_info.pipeline_node is not None
        else None
    )
    self.pipeline_id = (
        exec_info.pipeline_info.id
        if exec_info.pipeline_info is not None
        else None
    )
    self.pipeline_run_id = exec_info.pipeline_run_id
    self.top_level_pipeline_run_id = exec_info.top_level_pipeline_run_id
    self.frontend_url = exec_info.frontend_url
