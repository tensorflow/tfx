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
r"""Shared IR serialization logic used by TFleX python executor binary."""

import base64

from tfx.orchestration.portable import base_executor_operator
from tfx.proto.orchestration import executor_invocation_pb2
from tfx.types import artifact_utils
from ml_metadata.proto import metadata_store_pb2
from ml_metadata.proto import metadata_store_service_pb2

PORTABLE_LAUNCHER_PACKAGE = ('test/learning/tfx/tflex/pluggable_orchestrator/'
                             'python_executor_binary')
PORTABLE_LAUNCHER_BINARY = 'python_executor_binary.par'


def _build_artifact_dict(proto_dict):
  """Build ExecutionInfo input/output artifact dicts."""
  artifact_dict = {}
  for k, v in proto_dict.items():
    artifact_dict[k] = []
    for artifact_struct in v.elements:
      if not artifact_struct.HasField('artifact'):
        raise RuntimeError('Only support artifact oneof field')
      artifact_and_type = artifact_struct.artifact
      artifact_dict[k].append(
          artifact_utils.deserialize_artifact(artifact_and_type.type,
                                              artifact_and_type.artifact))
  return artifact_dict


def _build_proto_artifact_dict(artifact_dict):
  """Build PythonExecutorExecutionInfo input/output artifact dicts."""
  proto_dict = {}
  for k, v in artifact_dict.items():
    artifact_list = metadata_store_service_pb2.ArtifactStructList()
    for artifact in v:
      artifact_struct = metadata_store_service_pb2.ArtifactStruct(
          artifact=metadata_store_service_pb2.ArtifactAndType(
              artifact=artifact.mlmd_artifact, type=artifact.artifact_type))
      artifact_list.elements.append(artifact_struct)
    proto_dict[k] = artifact_list
  return proto_dict


def _build_exec_property_dict(proto_dict):
  """Build ExecutionInfo.exec_properties."""
  exec_property_dict = {}
  for k, v in proto_dict.items():
    exec_property_dict[k] = getattr(v, v.WhichOneof('value'))
  return exec_property_dict


def _build_proto_exec_property_dict(exec_properties):
  """Build PythonExecutorExecutionInfo.execution_properties."""
  proto_dict = {}
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
    proto_dict[k] = value
  return proto_dict


def deserialize_execution_info(
    execution_info_b64: str) -> base_executor_operator.ExecutionInfo:
  """De-serialize the ExecutionInfo class from a binary string."""
  execution_info_proto = executor_invocation_pb2.ExecutorInvocation.FromString(
      base64.b64decode(execution_info_b64))
  execution_info = base_executor_operator.ExecutionInfo(
      executor_output_uri=execution_info_proto.output_metadata_uri,
      stateful_working_dir=execution_info_proto.stateful_working_dir)

  execution_info.exec_properties = _build_exec_property_dict(
      execution_info_proto.execution_properties)

  execution_info.input_dict = _build_artifact_dict(
      execution_info_proto.input_dict)
  execution_info.output_dict = _build_artifact_dict(
      execution_info_proto.output_dict)
  return execution_info


def serialize_execution_info(
    execution_info: base_executor_operator.ExecutionInfo) -> str:
  """Serialize the ExecutionInfo class from a binary string."""
  execution_info_proto = executor_invocation_pb2.ExecutorInvocation(
      output_metadata_uri=execution_info.executor_output_uri,
      stateful_working_dir=execution_info.stateful_working_dir,
      execution_properties=_build_proto_exec_property_dict(
          execution_info.exec_properties),
      input_dict=_build_proto_artifact_dict(execution_info.input_dict),
      output_dict=_build_proto_artifact_dict(execution_info.output_dict))

  return base64.b64encode(
      execution_info_proto.SerializeToString()).decode('ascii')
