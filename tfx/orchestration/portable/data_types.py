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
from typing import Any, Dict, Iterable, List, Mapping

import attr
from tfx import types
from tfx.proto.orchestration import executor_invocation_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import artifact_utils

from ml_metadata.proto import metadata_store_pb2
from ml_metadata.proto import metadata_store_service_pb2


def _build_artifact_dict(
    proto_dict: Mapping[str, metadata_store_service_pb2.ArtifactStructList]
) -> Dict[str, List[types.Artifact]]:
  """Builds ExecutionInfo input/output artifact dicts."""
  result = {}
  for k, v in proto_dict.items():
    result[k] = []
    for artifact_struct in v.elements:
      if not artifact_struct.HasField('artifact'):
        raise RuntimeError('Only support artifact oneof field')
      artifact_and_type = artifact_struct.artifact
      result[k].append(
          artifact_utils.deserialize_artifact(artifact_and_type.type,
                                              artifact_and_type.artifact))
  return result


def _build_proto_artifact_dict(
    artifact_dict: Mapping[str, Iterable[types.Artifact]]
) -> Dict[str, metadata_store_service_pb2.ArtifactStructList]:
  """Builds PythonExecutorExecutionInfo input/output artifact dicts."""
  result = {}
  if not artifact_dict:
    return result
  for k, v in artifact_dict.items():
    artifact_list = metadata_store_service_pb2.ArtifactStructList()
    for artifact in v:
      artifact_struct = metadata_store_service_pb2.ArtifactStruct(
          artifact=metadata_store_service_pb2.ArtifactAndType(
              artifact=artifact.mlmd_artifact, type=artifact.artifact_type))
      artifact_list.elements.append(artifact_struct)
    result[k] = artifact_list
  return result


def _build_exec_property_dict(
    proto_dict: Mapping[str, metadata_store_pb2.Value]
) -> Dict[str, types.Property]:
  """Builds ExecutionInfo.exec_properties."""
  result = {}
  for k, v in proto_dict.items():
    result[k] = getattr(v, v.WhichOneof('value'))
  return result


def _build_proto_exec_property_dict(
    exec_properties: Mapping[str, types.Property]
) -> Dict[str, metadata_store_pb2.Value]:
  """Builds PythonExecutorExecutionInfo.execution_properties."""
  result = {}
  if not exec_properties:
    return result
  for k, v in exec_properties.items():
    value = metadata_store_pb2.Value()
    if isinstance(v, str):
      value.string_value = v
    elif isinstance(v, int):
      value.int_value = v
    elif isinstance(v, float):
      value.double_value = v
    else:
      raise RuntimeError('Unsupported type {} for key {}'.format(type(v), k))
    result[k] = value
  return result


# TODO(b/150979622): We should introduce an id that is not changed across
# retires of the same component run and pass it to executor operators for
# human-readability purpose.
# TODO(b/165359991): Restore 'auto_attribs=True' once we drop Python3.5 support.
# TODO(b/172065067): Clean up ExecutionInfo to match placeholder SDK structure.
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
  # checkpoints.
  stateful_working_dir = attr.ib(type=str, default=None)
  # A tempory dir for executions and it is expected to be cleared up at the end
  # of executions in both success and failure cases.
  tmp_dir = attr.ib(type=str, default=None)
  # The config of this Node.
  pipeline_node = attr.ib(type=pipeline_pb2.PipelineNode, default=None)
  # The config of the pipeline that this node is running in.
  pipeline_info = attr.ib(type=pipeline_pb2.PipelineInfo, default=None)
  # The id of the pipeline run that this execution is in.
  pipeline_run_id = attr.ib(type=str, default=None)
  # LINT.ThenChange(../../proto/orchestration/executor_invocation.proto)

  def to_proto(
      self) -> executor_invocation_pb2.ExecutorInvocation:
    return executor_invocation_pb2.ExecutorInvocation(
        execution_id=self.execution_id,
        input_dict=_build_proto_artifact_dict(self.input_dict),
        output_dict=_build_proto_artifact_dict(self.output_dict),
        execution_properties=_build_proto_exec_property_dict(
            self.exec_properties),
        output_metadata_uri=self.execution_output_uri,
        stateful_working_dir=self.stateful_working_dir,
        tmp_dir=self.tmp_dir,
        pipeline_node=self.pipeline_node,
        pipeline_info=self.pipeline_info,
        pipeline_run_id=self.pipeline_run_id)

  @classmethod
  def from_proto(
      cls, executor_invocation: executor_invocation_pb2.ExecutorInvocation
  ) -> 'ExecutionInfo':
    return cls(
        execution_id=executor_invocation.execution_id,
        input_dict=_build_artifact_dict(executor_invocation.input_dict),
        output_dict=_build_artifact_dict(executor_invocation.output_dict),
        exec_properties=_build_exec_property_dict(
            executor_invocation.execution_properties),
        execution_output_uri=executor_invocation.output_metadata_uri,
        stateful_working_dir=executor_invocation.stateful_working_dir,
        tmp_dir=executor_invocation.tmp_dir,
        pipeline_node=executor_invocation.pipeline_node,
        pipeline_info=executor_invocation.pipeline_info,
        pipeline_run_id=executor_invocation.pipeline_run_id)
