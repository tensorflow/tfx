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
from typing import Dict, Iterable, List, Mapping, Optional

from tfx import types
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import artifact_utils
from tfx.utils import json_utils
from tfx.utils import proto_utils

from google.protobuf import message
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


def build_value_dict(
    metadata_value_dict: Mapping[str, metadata_store_pb2.Value]
) -> Dict[str, types.Property]:
  """Converts MLMD value dict into plain value dict."""
  result = {}
  for k, v in metadata_value_dict.items():
    result[k] = getattr(v, v.WhichOneof('value'))
  return result


def build_metadata_value_dict(
    value_dict: Mapping[str, types.ExecPropertyTypes]
) -> Dict[str, metadata_store_pb2.Value]:
  """Converts plain value dict into MLMD value dict."""
  result = {}
  if not value_dict:
    return result
  for k, v in value_dict.items():
    if v is None:
      continue
    value = metadata_store_pb2.Value()
    result[k] = set_metadata_value(value, v)
  return result


def build_pipeline_value_dict(
    value_dict: Dict[str, types.ExecPropertyTypes]
) -> Dict[str, pipeline_pb2.Value]:
  """Converts plain value dict into pipeline_pb2.Value dict."""
  result = {}
  if not value_dict:
    return result
  for k, v in value_dict.items():
    if v is None:
      continue
    value = pipeline_pb2.Value()
    result[k] = set_parameter_value(value, v)
  return result


def get_parsed_value(
    value: metadata_store_pb2.Value,
    schema: Optional[pipeline_pb2.Value.Schema]) -> types.ExecPropertyTypes:
  """Converts MLMD value into parsed (non-)primitive value."""

  def parse_value(
      value: str, value_type: pipeline_pb2.Value.Schema.ValueType
  ) -> types.ExecPropertyTypes:
    if value_type.HasField('list_type'):
      list_value = json_utils.loads(value)
      return [parse_value(val, value_type.list_type) for val in list_value]
    elif value_type.HasField('proto_type'):
      return proto_utils.deserialize_proto_message(
          value, value_type.proto_type.message_type,
          value_type.proto_type.file_descriptors)
    else:
      return value

  if schema and value.HasField('string_value'):
    if schema.value_type.HasField('boolean_type'):
      return json_utils.loads(value.string_value)
    else:
      return parse_value(value.string_value, schema.value_type)
  else:
    return getattr(value, value.WhichOneof('value'))


def build_parsed_value_dict(
    value_dict: Mapping[str, pipeline_pb2.Value]
) -> Dict[str, types.ExecPropertyTypes]:
  """Converts MLMD value into parsed (non-)primitive value dict."""
  result = {}
  if not value_dict:
    return result
  for k, v in value_dict.items():
    if not v.HasField('field_value'):
      raise RuntimeError('Field value missing for %s' % k)
    result[k] = get_parsed_value(v.field_value,
                                 v.schema if v.HasField('schema') else None)
  return result


def get_metadata_value_type(
    value: types.ExecPropertyTypes) -> metadata_store_pb2.PropertyType:
  """Gets the metadata property type of a property value from a value.

  Args:
    value: The property value represented by pipeline_pb2.Value or a primitive
      property value type.

  Returns:
    A metadata_store_pb2.PropertyType.

  Raises:
    RuntimeError: If property value is still in RuntimeParameter form
    ValueError: The value type is not supported.
  """
  if isinstance(value, int):
    return metadata_store_pb2.INT
  elif isinstance(value, float):
    return metadata_store_pb2.DOUBLE
  elif isinstance(value, pipeline_pb2.Value):
    which = value.WhichOneof('value')
    if which != 'field_value':
      raise RuntimeError('Expecting field_value but got %s.' % value)

    value_type = value.field_value.WhichOneof('value')
    if value_type == 'int_value':
      return metadata_store_pb2.INT
    elif value_type == 'double_value':
      return metadata_store_pb2.DOUBLE
    elif value_type == 'string_value':
      return metadata_store_pb2.STRING
    else:
      raise ValueError('Unexpected value type %s' % value_type)
  elif isinstance(value, (str, bool, message.Message, list)):
    return metadata_store_pb2.STRING
  else:
    raise ValueError('Unexpected value type %s' % type(value))


def get_value(tfx_value: pipeline_pb2.Value) -> types.Property:
  """Gets the primitive type value of a pipeline_pb2.Value instance.

  Args:
    tfx_value: A pipeline_pb2.Value message.

  Returns:
    The primitive type value of the tfx value.

  Raises:
    RuntimeError: when the value is still in RuntimeParameter form.
  """
  which = tfx_value.WhichOneof('value')
  if which != 'field_value':
    raise RuntimeError('Expecting field_value but got %s.' % tfx_value)

  return getattr(tfx_value.field_value,
                 tfx_value.field_value.WhichOneof('value'))


def get_metadata_value(
    value: metadata_store_pb2.Value) -> Optional[types.Property]:
  """Gets the primitive type value of a metadata_store_pb2.Value instance.

  Args:
    value: A metadata_store_pb2.Value message.

  Returns:
    The primitive type value of metadata_store_pb2.Value instance if set, `None`
    otherwise.
  """
  which = value.WhichOneof('value')
  return None if which is None else getattr(value, which)


def set_metadata_value(
    metadata_value: metadata_store_pb2.Value,
    value: types.ExecPropertyTypes) -> metadata_store_pb2.Value:
  """Sets metadata property based on tfx value.

  Args:
    metadata_value: A metadata_store_pb2.Value message to be set.
    value: The value of the property in pipeline_pb2.Value form.

  Returns:
    A Value proto filled with the provided value.

  Raises:
    ValueError: If value type is not supported or is still RuntimeParameter.
  """
  parameter_value = pipeline_pb2.Value()
  set_parameter_value(parameter_value, value, set_schema=False)
  metadata_value.CopyFrom(parameter_value.field_value)
  return metadata_value


def set_parameter_value(
    parameter_value: pipeline_pb2.Value,
    value: types.ExecPropertyTypes,
    set_schema: Optional[bool] = True) -> pipeline_pb2.Value:
  """Sets field value and schema based on tfx value.

  Args:
    parameter_value: A pipeline_pb2.Value message to be set.
    value: The value of the property.
    set_schema: Boolean value indicating whether to set schema in
      pipeline_pb2.Value.

  Returns:
    A pipeline_pb2.Value proto with field_value and optionally schema filled
    based on input property.

  Raises:
    ValueError: If value type is not supported.
  """

  def get_value_and_set_type(
      value: types.ExecPropertyTypes,
      value_type: pipeline_pb2.Value.Schema.ValueType) -> types.Property:
    """Returns serialized value and sets value_type."""
    if isinstance(value, bool):
      if set_schema:
        value_type.boolean_type.SetInParent()
      return value
    elif isinstance(value, message.Message):
      # TODO(b/171794016): Investigate if file descripter set is needed for
      # tfx-owned proto already build in the launcher binary.
      if set_schema:
        proto_type = value_type.proto_type
        proto_type.message_type = type(value).DESCRIPTOR.full_name
        proto_utils.build_file_descriptor_set(value,
                                              proto_type.file_descriptors)
      return proto_utils.proto_to_json(value)
    elif isinstance(value, list) and len(value):
      if set_schema:
        value_type.list_type.SetInParent()
      value = [
          get_value_and_set_type(val, value_type.list_type) for val in value
      ]
      return json_utils.dumps(value)
    elif isinstance(value, (int, float, str)):
      return value
    else:
      raise ValueError('Unexpected type %s' % type(value))

  if isinstance(value, int) and not isinstance(value, bool):
    parameter_value.field_value.int_value = value
  elif isinstance(value, float):
    parameter_value.field_value.double_value = value
  elif isinstance(value, str):
    parameter_value.field_value.string_value = value
  elif isinstance(value, pipeline_pb2.Value):
    which = value.WhichOneof('value')
    if which != 'field_value':
      raise ValueError('Expecting field_value but got %s.' % value)
    parameter_value.field_value.CopyFrom(value.field_value)
  elif isinstance(value, bool):
    parameter_value.schema.value_type.boolean_type.SetInParent()
    parameter_value.field_value.string_value = json_utils.dumps(value)
  elif isinstance(value, (list, message.Message)):
    parameter_value.field_value.string_value = get_value_and_set_type(
        value, parameter_value.schema.value_type)
  else:
    raise ValueError('Unexpected type %s' % type(value))

  return parameter_value
