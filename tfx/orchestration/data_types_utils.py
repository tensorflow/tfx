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
"""Data types util shared for orchestration."""
from typing import Dict, Iterable, List, Mapping

from tfx import types
from tfx.types import artifact_utils

from ml_metadata.proto import metadata_store_pb2
from ml_metadata.proto import metadata_store_service_pb2


def build_artifact_dict(
    proto_dict: Mapping[str, metadata_store_service_pb2.ArtifactStructList]
) -> Dict[str, List[types.Artifact]]:
  """Converts input/output artifact dict."""
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


def build_artifact_struct_dict(
    artifact_dict: Mapping[str, Iterable[types.Artifact]]
) -> Dict[str, metadata_store_service_pb2.ArtifactStructList]:
  """Converts input/output artifact dict."""
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


def build_exec_property_dict(
    proto_dict: Mapping[str, metadata_store_pb2.Value]
) -> Dict[str, types.Property]:
  """Converts exec_properties dict."""
  result = {}
  for k, v in proto_dict.items():
    result[k] = getattr(v, v.WhichOneof('value'))
  return result


def build_exec_property_value_dict(
    exec_properties: Mapping[str, types.Property]
) -> Dict[str, metadata_store_pb2.Value]:
  """Converts exec_properties dict."""
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
