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
import copy
from typing import Any, List, Mapping, Sequence, Union

from tfx.orchestration import data_types_utils
from tfx.orchestration import metadata
from tfx.orchestration.portable import data_types
from tfx.proto.orchestration import executable_spec_pb2
from tfx.proto.orchestration import execution_invocation_pb2
from tfx.proto.orchestration import execution_result_pb2
from tfx.proto.orchestration import metadata_pb2
from tfx.types import artifact

from ml_metadata.proto import metadata_store_pb2


def deserialize_execution_info(
    execution_info_b64: str) -> data_types.ExecutionInfo:
  """De-serializes the ExecutionInfo class from a url safe base64 encoded binary string."""
  execution_info_proto = execution_invocation_pb2.ExecutionInvocation.FromString(
      base64.urlsafe_b64decode(execution_info_b64))
  return data_types.ExecutionInfo.from_proto(execution_info_proto)


def deserialize_mlmd_connection_config(
    mlmd_connection_config_b64: str) -> metadata.ConnectionConfigType:
  """De-serializes an MLMD connection config from base64 flag."""
  mlmd_connection_config = (
      metadata_pb2.MLMDConnectionConfig.FromString(
          base64.b64decode(mlmd_connection_config_b64)))
  return getattr(mlmd_connection_config,
                 mlmd_connection_config.WhichOneof('connection_config'))


def deserialize_executable_spec(
    executable_spec_b64: str,
    with_beam: bool = False,
) -> Union[executable_spec_pb2.PythonClassExecutableSpec,
           executable_spec_pb2.BeamExecutableSpec]:
  """De-serializes an executable spec from base64 flag."""
  if with_beam:
    return executable_spec_pb2.BeamExecutableSpec.FromString(
        base64.b64decode(executable_spec_b64))
  return executable_spec_pb2.PythonClassExecutableSpec.FromString(
      base64.b64decode(executable_spec_b64))


def serialize_mlmd_connection_config(
    connection_config: metadata.ConnectionConfigType) -> str:
  """Serializes an MLMD connection config into a base64 flag of its wrapper."""
  mlmd_wrapper = metadata_pb2.MLMDConnectionConfig()
  for name, descriptor in (
      metadata_pb2.MLMDConnectionConfig.DESCRIPTOR.fields_by_name.items()):
    if descriptor.message_type.full_name == connection_config.DESCRIPTOR.full_name:
      getattr(mlmd_wrapper, name).CopyFrom(connection_config)
      break
  return base64.b64encode(mlmd_wrapper.SerializeToString()).decode('ascii')


def serialize_executable_spec(
    executable_spec: Union[executable_spec_pb2.PythonClassExecutableSpec,
                           executable_spec_pb2.BeamExecutableSpec]
) -> str:
  """Serializes an executable spec into a base64 flag."""
  return base64.b64encode(executable_spec.SerializeToString()).decode('ascii')


def serialize_execution_info(execution_info: data_types.ExecutionInfo) -> str:
  """Serializes the ExecutionInfo class from a base64 flag."""
  execution_info_proto = execution_info.to_proto()
  return base64.b64encode(
      execution_info_proto.SerializeToString()).decode('ascii')


def ensure_artifact_list(
    value: Union[artifact.Artifact, Sequence[artifact.Artifact]]
) -> List[artifact.Artifact]:
  """Ensures value is a list of artifacts."""
  return [value] if isinstance(value, artifact.Artifact) else list(value)


def convert_to_mlmd_artifact_list(
    value: Union[artifact.Artifact, Sequence[artifact.Artifact]]
) -> List[metadata_store_pb2.Artifact]:
  """Convert tfx.Artifacts to MLMD Artifacts."""
  return (
      [value.mlmd_artifact]
      if isinstance(value, artifact.Artifact)
      else [v.mlmd_artifact for v in value]
  )


def get_updated_execution_invocation(
    original_execution_invocation: data_types.ExecutionInfo,
    hook_fn_kwargs: Mapping[str, Any],
) -> data_types.ExecutionInfo:
  """Gets updated execution invocation by running a pre-execution hook function."""
  execution_invocation = copy.deepcopy(original_execution_invocation)
  for key, value in hook_fn_kwargs.items():
    if key in execution_invocation.output_dict:
      execution_invocation.output_dict[key] = ensure_artifact_list(value)
  return execution_invocation


def get_updated_executor_output(
    original_execution_invocation: data_types.ExecutionInfo,
    original_execution_output: execution_result_pb2.ExecutorOutput,
    hook_fn_kwargs: Mapping[str, Any],
) -> execution_result_pb2.ExecutorOutput:
  """Gets updated executor output by running a post-execution hook function."""
  result = execution_result_pb2.ExecutorOutput()
  result.CopyFrom(original_execution_output)
  output_artifacts = data_types_utils.unpack_executor_output_artifacts(
      original_execution_output.output_artifacts
  )
  for key, value in hook_fn_kwargs.items():
    if (
        key in original_execution_invocation.output_dict
        or key in output_artifacts
    ):
      output_artifacts[key] = convert_to_mlmd_artifact_list(value)

  result.output_artifacts.clear()
  for key, value in output_artifacts.items():
    result.output_artifacts[key].artifacts.extend(value)

  return result
