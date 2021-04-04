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
from tfx.orchestration import data_types_utils
from tfx.proto.orchestration import execution_invocation_pb2
from tfx.proto.orchestration import pipeline_pb2


# TODO(b/150979622): We should introduce an id that is not changed across
# retires of the same component run and pass it to executor operators for
# human-readability purpose.
# TODO(b/165359991): Restore 'auto_attribs=True' once we drop Python3.5 support.
@attr.s
class ExecutionInfo:
  """A struct to store information for an execution."""
  # LINT.IfChange
  # The Execution id that is registered in MLMD.
  execution_id = attr.ib(type=int, default=None)
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
  # checkpoints. This dir is undefined when Launcher.launch() is done.
  stateful_working_dir = attr.ib(type=str, default=None)
  # A tempory dir for executions and it is expected to be cleared up at the end
  # of executions in both success and failure cases. This dir is undefined when
  # Launcher.launch() is done.
  tmp_dir = attr.ib(type=str, default=None)
  # The config of this Node.
  pipeline_node = attr.ib(type=pipeline_pb2.PipelineNode, default=None)
  # The config of the pipeline that this node is running in.
  pipeline_info = attr.ib(type=pipeline_pb2.PipelineInfo, default=None)
  # The id of the pipeline run that this execution is in.
  pipeline_run_id = attr.ib(type=str, default=None)
  # LINT.ThenChange(../../proto/orchestration/execution_invocation.proto)

  def to_proto(self) -> execution_invocation_pb2.ExecutionInvocation:
    return execution_invocation_pb2.ExecutionInvocation(
        execution_id=self.execution_id,
        input_dict=data_types_utils.build_artifact_struct_dict(self.input_dict),
        output_dict=data_types_utils.build_artifact_struct_dict(
            self.output_dict),
        execution_properties=data_types_utils.build_metadata_value_dict(
            self.exec_properties),
        output_metadata_uri=self.execution_output_uri,
        stateful_working_dir=self.stateful_working_dir,
        tmp_dir=self.tmp_dir,
        pipeline_node=self.pipeline_node,
        pipeline_info=self.pipeline_info,
        pipeline_run_id=self.pipeline_run_id)

  @classmethod
  def from_proto(
      cls, execution_invocation: execution_invocation_pb2.ExecutionInvocation
  ) -> 'ExecutionInfo':
    return cls(
        execution_id=execution_invocation.execution_id,
        input_dict=data_types_utils.build_artifact_dict(
            execution_invocation.input_dict),
        output_dict=data_types_utils.build_artifact_dict(
            execution_invocation.output_dict),
        exec_properties=data_types_utils.build_value_dict(
            execution_invocation.execution_properties),
        execution_output_uri=execution_invocation.output_metadata_uri,
        stateful_working_dir=execution_invocation.stateful_working_dir,
        tmp_dir=execution_invocation.tmp_dir,
        pipeline_node=execution_invocation.pipeline_node,
        pipeline_info=execution_invocation.pipeline_info,
        pipeline_run_id=execution_invocation.pipeline_run_id)
