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
"""Data types shared for orchestration."""
from typing import Any, Dict, List

import attr
from tfx import types
from tfx.proto.orchestration import pipeline_pb2

from ml_metadata.proto import metadata_store_pb2


# TODO(b/150979622): We should introduce an id that is not changed across
# retires of the same component run and pass it to executor operators for
# human-readability purpose.
# TODO(b/165359991): Restore 'auto_attribs=True' once we drop Python3.5 support.
# LINT.IfChange
# TODO(ruoyu): Move it out to a common place.
@attr.s
class ExecutionInfo:
  """A struct to store information for an execution."""
  # The metadata of this execution that is registered in MLMD.
  execution_metadata = attr.ib(type=metadata_store_pb2.Execution, default=None)
  # The input map to feed to execution
  input_dict = attr.ib(type=Dict[str, List[types.Artifact]], default=None)
  # The output map to feed to execution
  output_dict = attr.ib(type=Dict[str, List[types.Artifact]], default=None)
  # The exec_properties to feed to execution
  exec_properties = attr.ib(type=Dict[str, Any], default=None)
  # The uri to execution result, note that the drivers or executors and
  # Launchers may not run in the same process, so they should use this uri to
  # "return" execution result to the launcher.
  execution_output_uri = attr.ib(type=str, default=None)
  # Stateful working dir will be deterministic given pipeline, node and run_id.
  # The typical usecase is to restore long running executor's state after
  # eviction. For examples, a Trainer can use this directory to store
  # checkpoints.
  stateful_working_dir = attr.ib(type=str, default=None)
  # A tempory dir for executions and it is expected to be cleared up at the end
  # of executions in both success and failure cases.
  tmp_dir = attr.ib(type=str, default=None)
  # The config of this Node.
  pipeline_node = attr.ib(type=pipeline_pb2.PipelineNode, default=None)
  # The config of the pipeline that this node is running in.
  pipeline_info = attr.ib(type=pipeline_pb2.PipelineInfo, default=None)


# LINT.ThenChange(../../proto/orchestration/executor_invocation.proto)
