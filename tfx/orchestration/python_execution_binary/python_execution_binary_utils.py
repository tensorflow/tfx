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

from typing import Dict, Iterable, List, Mapping

from tfx import types
from tfx.orchestration import metadata
from tfx.orchestration.portable import data_types
from tfx.proto.orchestration import executable_spec_pb2
from tfx.proto.orchestration import executor_invocation_pb2
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


def deserialize_execution_info(
    execution_info_b64: str) -> data_types.ExecutionInfo:
  """De-serializes the ExecutionInfo class from a binary string."""
  execution_info_proto = (
      executor_invocation_pb2.ExecutorInvocation.FromString(
          base64.b64decode(execution_info_b64)))
  result = data_types.ExecutionInfo(
      execution_output_uri=execution_info_proto.output_metadata_uri,
      stateful_working_dir=execution_info_proto.stateful_working_dir,
      pipeline_info=execution_info_proto.pipeline_info,
      pipeline_node=execution_info_proto.pipeline_node)

  result.exec_properties = _build_exec_property_dict(
      execution_info_proto.execution_properties)

  result.input_dict = _build_artifact_dict(execution_info_proto.input_dict)
  result.output_dict = _build_artifact_dict(execution_info_proto.output_dict)
  return result


def deserialize_mlmd_connection_config(
    mlmd_connection_config_b64: str) -> metadata.ConnectionConfigType:
  """De-serializes an MLMD connection config from base64 flag."""
  mlmd_connection_config = (
      executor_invocation_pb2.MLMDConnectionConfig.FromString(
          base64.b64decode(mlmd_connection_config_b64)))
  return getattr(mlmd_connection_config,
                 mlmd_connection_config.WhichOneof('connection_config'))


def deserialize_executable_spec(
    executable_spec_b64: str) -> executable_spec_pb2.PythonClassExecutableSpec:
  """De-serializes an executable spec from base64 flag."""
  return executable_spec_pb2.PythonClassExecutableSpec.FromString(
      base64.b64decode(executable_spec_b64))


def serialize_mlmd_connection_config(
    connection_config: metadata.ConnectionConfigType) -> str:
  """Serializes an MLMD connection config into a base64 flag of its wrapper."""
  mlmd_wrapper = executor_invocation_pb2.MLMDConnectionConfig()
  for name, descriptor in (executor_invocation_pb2.MLMDConnectionConfig
                           .DESCRIPTOR.fields_by_name.items()):
    if descriptor.message_type.full_name == connection_config.DESCRIPTOR.full_name:
      getattr(mlmd_wrapper, name).CopyFrom(connection_config)
      break
  return base64.b64encode(mlmd_wrapper.SerializeToString()).decode('ascii')


def serialize_executable_spec(
    executable_spec: executable_spec_pb2.PythonClassExecutableSpec) -> str:
  """Serializes an executable spec into a base64 flag."""
  return base64.b64encode(executable_spec.SerializeToString()).decode('ascii')


def serialize_execution_info(execution_info: data_types.ExecutionInfo) -> str:
  """Serializes the ExecutionInfo class from a base64 flag."""
  execution_info_proto = executor_invocation_pb2.ExecutorInvocation(
      output_metadata_uri=execution_info.execution_output_uri,
      stateful_working_dir=execution_info.stateful_working_dir,
      execution_properties=_build_proto_exec_property_dict(
          execution_info.exec_properties),
      input_dict=_build_proto_artifact_dict(execution_info.input_dict),
      output_dict=_build_proto_artifact_dict(execution_info.output_dict),
      pipeline_info=execution_info.pipeline_info,
      pipeline_node=execution_info.pipeline_node)

  return base64.b64encode(
      execution_info_proto.SerializeToString()).decode('ascii')
