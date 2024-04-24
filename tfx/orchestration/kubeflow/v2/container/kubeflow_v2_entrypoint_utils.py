# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions used in kubeflow_v2_run_executor.py."""

import hashlib
from typing import Any, Dict, List, Mapping, MutableMapping, Optional

from absl import logging
from kfp.pipeline_spec import pipeline_spec_pb2 as pipeline_pb2
from tfx.components.evaluator import constants
from tfx.orchestration.kubeflow.v2 import compiler_utils
from tfx.types import artifact
from tfx.types import artifact_utils
from tfx.types import standard_component_specs
from tfx.utils import import_utils
import yaml

from google.protobuf import struct_pb2
from google.protobuf import json_format
from ml_metadata.proto import metadata_store_pb2

# Old execution property name. This is mapped to utils.INPUT_BASE_KEY.
_OLD_INPUT_BASE_PROPERTY_NAME = 'input_base_uri'

# Max value for int64
_INT64_MAX = 1 << 63


def parse_raw_artifact_dict(
    inputs_dict: Any,
    name_from_id: Optional[MutableMapping[int, str]] = None
) -> Dict[str, List[artifact.Artifact]]:
  """Parses a map from key to a list of a single Artifact from pb objects.

  Parses a mapping field in a protobuf message, whose value is an
  ExecutorInput.ArtifactList message, to a Python dict, whose value is a list of
  TFX Artifact Python objects.

  Args:
    inputs_dict: the mapping field in the proto message.
    name_from_id: the dict used to store the id to string-typed name mapping.

  Returns:
    dictionary of the parsed Python Artifact objects.
  """
  if name_from_id is None:
    name_from_id = {}
  result = {}
  for k, v in inputs_dict.items():
    result[k] = [
        _parse_raw_artifact(single_artifact, name_from_id)
        for single_artifact in v.artifacts
    ]
  return result


def _get_hashed_id(full_name: str, name_from_id: MutableMapping[int,
                                                                str]) -> int:
  """Converts the string-typed name to int-typed ID."""
  # Built-in hash function will not exceed the range of int64, which is the
  # type of id in metadata artifact proto.
  result = int(hashlib.sha256(full_name.encode('utf-8')).hexdigest(),
               16) % _INT64_MAX
  name_from_id[result] = full_name
  return result


def _get_full_name(artifact_id: int, name_from_id: Mapping[int, str]) -> str:
  """Converts the int-typed id to full string name."""
  return name_from_id[artifact_id]


# TODO(b/169583143): Remove this workaround when TFX migrates to use str-typed
# id/name to identify artifacts.
# Currently the contract is:
# - In TFX stack, artifact IDs are integers.
# - In pipeline stack, artifact IDs are strings, with format determined by the
#   type and implementation of the metadata store in use.
# Therefore conversion is needed when parsing RuntimeArtifact populated by
# pipeline and also when writing out ExecutorOutput.
# This function is expected to be executed right before the TFX container
# writes out ExecutorOutput pb message. It converts the int-typed ID fields to
# string-typed ones conforming the contract with the metadata store being used.
def refactor_model_blessing(model_blessing: artifact.Artifact,
                            name_from_id: Mapping[int, str]) -> None:
  """Changes id-typed custom properties to string-typed runtime artifact name."""
  if model_blessing.has_custom_property(
      constants.ARTIFACT_PROPERTY_BASELINE_MODEL_ID_KEY):
    model_blessing.set_string_custom_property(
        constants.ARTIFACT_PROPERTY_BASELINE_MODEL_ID_KEY,
        _get_full_name(
            artifact_id=model_blessing.get_int_custom_property(
                constants.ARTIFACT_PROPERTY_BASELINE_MODEL_ID_KEY),
            name_from_id=name_from_id))
  if model_blessing.has_custom_property(
      constants.ARTIFACT_PROPERTY_CURRENT_MODEL_ID_KEY):
    model_blessing.set_string_custom_property(
        constants.ARTIFACT_PROPERTY_CURRENT_MODEL_ID_KEY,
        _get_full_name(
            artifact_id=model_blessing.get_int_custom_property(
                constants.ARTIFACT_PROPERTY_CURRENT_MODEL_ID_KEY),
            name_from_id=name_from_id))


def parse_execution_properties(exec_properties: Any) -> Dict[str, Any]:
  """Parses a map from key to Value proto as execution properties.

  Parses a mapping field in a protobuf message, whose value is a Kubeflow Value
  proto messages, to a Python dict, whose value is a Python primitive object.

  Args:
    exec_properties: the mapping field in the proto message, representing the
      execution properties of the component.

  Returns:
    dictionary of the parsed execution properties.
  """
  result = {}
  for k, v in exec_properties.items():
    # TODO(b/159835994): Remove this once pipeline populates INPUT_BASE_KEY
    if k == _OLD_INPUT_BASE_PROPERTY_NAME:
      k = standard_component_specs.INPUT_BASE_KEY
    # Translate each field from Value pb to plain value.
    result[k] = getattr(v, v.WhichOneof('value'))
    if result[k] is None:
      raise TypeError('Unrecognized type encountered at field %s of execution'
                      ' properties %s' % (k, exec_properties))

  return result


def translate_executor_output(
    output_dict: Mapping[str, List[artifact.Artifact]],
    name_from_id: Mapping[int,
                          str]) -> Dict[str, pipeline_pb2.ArtifactList]:
  """Translates output_dict to a Kubeflow ArtifactList mapping."""
  result = {}
  for k, v in output_dict.items():
    result[k] = pipeline_pb2.ArtifactList(artifacts=[
        to_runtime_artifact(
            artifact_utils.get_single_instance(v), name_from_id)
    ])

  return result


def _get_json_value_mapping(
    mlmd_value_mapping: Dict[str, metadata_store_pb2.Value]) -> Dict[str, Any]:
  """Converts a mapping field with MLMD Value to JSON Value."""

  def get_json_value(
      mlmd_value: metadata_store_pb2.Value) -> artifact.JsonValueType:
    if not mlmd_value.HasField('value'):
      return None
    elif mlmd_value.WhichOneof('value') == 'int_value':
      return float(mlmd_value.int_value)
    elif mlmd_value.WhichOneof('value') == 'double_value':
      return mlmd_value.double_value
    elif mlmd_value.WhichOneof('value') == 'string_value':
      return mlmd_value.string_value
    elif mlmd_value.WhichOneof('value') == 'struct_value':
      return artifact._decode_struct_value(mlmd_value.struct_value)  # pylint: disable=protected-access
    else:
      raise TypeError('Get unknown type of value: {}'.format(mlmd_value))

  return {k: get_json_value(v) for k, v in mlmd_value_mapping.items()}


def _get_json_metadata_mapping(
    artifact_instance: artifact.Artifact) -> Dict[str, Any]:
  """Converts Artifact properties to a JSON dictionary."""
  properties_json = _get_json_value_mapping(
      artifact_instance.mlmd_artifact.properties)
  custom_properties_json = _get_json_value_mapping(
      artifact_instance.mlmd_artifact.custom_properties)
  metadata_dict = {}
  for key, value in properties_json.items():
    metadata_dict[key] = value
  for key, value in custom_properties_json.items():
    if key in artifact_instance.artifact_type.properties:
      key = artifact.CUSTOM_PROPERTIES_PREFIX + key
    metadata_dict[key] = value
  return metadata_dict


def to_runtime_artifact(
    artifact_instance: artifact.Artifact,
    name_from_id: Mapping[int, str]) -> pipeline_pb2.RuntimeArtifact:
  """Converts TFX artifact instance to RuntimeArtifact proto message."""
  metadata = struct_pb2.Struct()
  json_format.ParseDict(_get_json_metadata_mapping(artifact_instance), metadata)
  result = pipeline_pb2.RuntimeArtifact(
      uri=artifact_instance.uri, metadata=metadata)
  # TODO(b/135056715): Change to a unified getter/setter of Artifact type
  # once it's ready.
  # Try convert tfx artifact id to string-typed name. This should be the case
  # when running on an environment where metadata access layer is not running
  # in user space.
  id_or_none = getattr(artifact_instance, 'id', None)
  if (id_or_none is not None and id_or_none in name_from_id):
    result.name = name_from_id[id_or_none]
  else:
    logging.warning('Cannot convert ID back to runtime name for artifact %s',
                    artifact_instance)
  return result


def _retrieve_class_path(type_schema: pipeline_pb2.ArtifactTypeSchema) -> str:
  """Gets the class path from an artifact type schema."""
  if type_schema.WhichOneof('kind') == 'schema_title':
    title = type_schema.schema_title

  if type_schema.WhichOneof('kind') == 'instance_schema':
    data = yaml.safe_load(type_schema.instance_schema)
    title = data.get('title', 'tfx.Artifact')

  if title in compiler_utils.TITLE_TO_CLASS_PATH:
    # For first party types, the actual import path is maintained in
    # TITLE_TO_CLASS_PATH map.
    return compiler_utils.TITLE_TO_CLASS_PATH[title]
  else:
    # For custom types, the import path is encoded as the schema title.
    return title


def _parse_raw_artifact(
    artifact_pb: pipeline_pb2.RuntimeArtifact,
    name_from_id: MutableMapping[int, str]) -> artifact.Artifact:
  """Parses RuntimeArtifact proto message without artifact_type."""
  # This parser can only reserve what's inside the RuntimeArtifact pb message.

  # Recovers the type information from artifact type schema.
  # TODO(b/170261670): Replace this workaround by a more resilient
  # implementation. Currently custom artifact type can hardly be supported.
  assert artifact_pb.type, 'RuntimeArtifact is expected to have a type.'

  # 1. Import the artifact class from preloaded TFX library.
  type_path = _retrieve_class_path(artifact_pb.type)
  artifact_cls = import_utils.import_class_by_path(type_path)

  # 2. Copy properties and custom properties to the MLMD artifact pb.
  mlmd_artifact = metadata_store_pb2.Artifact()
  # TODO(b/135056715): Change to a unified getter/setter of Artifact type
  # once it's ready.
  if artifact_pb.name:
    # TODO(b/169583143): Remove this workaround when TFX migrates to use
    # str-typed id/name to identify artifacts.
    # Convert and populate the MLMD artifact ID.
    mlmd_artifact.id = _get_hashed_id(artifact_pb.name, name_from_id)

  mlmd_artifact.uri = artifact_pb.uri
  for k, v in artifact_pb.properties.items():
    mlmd_artifact.properties[k].CopyFrom(compiler_utils.get_mlmd_value(v))

  for k, v in artifact_pb.custom_properties.items():
    mlmd_artifact.custom_properties[k].CopyFrom(
        compiler_utils.get_mlmd_value(v))

  # Translate metadata items into properties and custom properties.
  mlmd_artifact_type = artifact_cls().artifact_type
  metadata_dict = json_format.MessageToDict(artifact_pb.metadata)
  for k, v in metadata_dict.items():
    if k in mlmd_artifact_type.properties:
      property_type = mlmd_artifact_type.properties[k]
      if property_type == metadata_store_pb2.INT and isinstance(v, float):
        mlmd_artifact.properties[k].int_value = int(v)
        continue
      elif property_type == metadata_store_pb2.DOUBLE and isinstance(v, float):
        mlmd_artifact.properties[k].double_value = v
        continue
      elif property_type == metadata_store_pb2.STRING and isinstance(v, str):
        mlmd_artifact.properties[k].string_value = v
        continue
      elif property_type == metadata_store_pb2.STRUCT:
        mlmd_artifact.properties[k].struct_value.CopyFrom(
            artifact._encode_struct_value(v))  # pylint: disable=protected-access
        continue
      # We fell through, which means the property doesn't actually fit the
      # schema. Therefore, we treat it as a custom property.

    # First, we drop the custom property prefix if we had to drop it because
    # of a property name conflict.
    if k.startswith(artifact.CUSTOM_PROPERTIES_PREFIX):
      stripped_k = k[len(artifact.CUSTOM_PROPERTIES_PREFIX):]
      if stripped_k in mlmd_artifact_type.properties:
        k = stripped_k
    mlmd_artifact.custom_properties[k].struct_value.CopyFrom(
        artifact._encode_struct_value(v))  # pylint: disable=protected-access

  # 3. Instantiate the artifact Python object.
  result = artifact_cls()
  result.set_mlmd_artifact(mlmd_artifact)

  return result
