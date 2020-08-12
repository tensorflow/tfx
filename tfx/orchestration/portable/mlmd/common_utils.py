# Lint as: python3
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
"""Common MLMD utility libraries."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Text, TypeVar, Union

from absl import logging

from tfx import types
from tfx.orchestration import metadata
from tfx.proto.orchestration import pipeline_pb2
import ml_metadata as mlmd
from ml_metadata.proto import metadata_store_pb2

MetadataType = TypeVar('MetadataType', metadata_store_pb2.ArtifactType,
                       metadata_store_pb2.ContextType,
                       metadata_store_pb2.ExecutionType)


def get_metadata_value_type(
    value: Union[pipeline_pb2.Value, types.Property]
) -> metadata_store_pb2.PropertyType:
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
  elif isinstance(value, Text):
    return metadata_store_pb2.STRING
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


def set_metadata_value(
    metadata_value: metadata_store_pb2.Value,
    value: Union[pipeline_pb2.Value, types.Property]) -> None:
  """Set metadata property based on tfx value.

  Args:
    metadata_value: A metadata_store_pb2.Value message to be set.
    value: The value of the property in pipeline_pb2.Value form.

  Returns:
    None

  Raises:
    RuntimeError: If value type is not supported or is still RuntimeParameter.
  """
  if isinstance(value, int):
    metadata_value.int_value = value
  elif isinstance(value, float):
    metadata_value.double_value = value
  elif isinstance(value, Text):
    metadata_value.string_value = value
  elif isinstance(value, pipeline_pb2.Value):
    which = value.WhichOneof('value')
    if which != 'field_value':
      raise RuntimeError('Expecting field_value but got %s.' % value)

    metadata_value.CopyFrom(value.field_value)
  else:
    raise RuntimeError('Unexpected type %s' % type(value))


def register_type_if_not_exist(
    metadata_handler: metadata.Metadata,
    metadata_type: MetadataType,
) -> MetadataType:
  """Registers a metadata type if not exists.

  Uses existing type if schema is superset of what is needed. Otherwise tries
  to register new metadata type.

  Args:
    metadata_handler: A handler to access MLMD store.
    metadata_type: The metadata type to register if does not exist.

  Returns:
    A MetadataType with id

  Raises:
    RuntimeError: If new metadata type conflicts with existing schema in MLMD.
    ValueError: If metadata type is not expected.
  """
  if metadata_type.id:
    return metadata_type

  if isinstance(metadata_type, metadata_store_pb2.ArtifactType):
    get_type_handler = metadata_handler.store.get_artifact_type
    put_type_handler = metadata_handler.store.put_artifact_type
  elif isinstance(metadata_type, metadata_store_pb2.ContextType):
    get_type_handler = metadata_handler.store.get_context_type
    put_type_handler = metadata_handler.store.put_context_type
  elif isinstance(metadata_type, metadata_store_pb2.ExecutionType):
    get_type_handler = metadata_handler.store.get_execution_type
    put_type_handler = metadata_handler.store.put_execution_type
  else:
    raise ValueError('Unexpected value type: %s.' % type(metadata_type))

  try:
    type_id = put_type_handler(metadata_type, can_add_fields=True)
    logging.debug('Registering a new metadata type with id %s.', type_id)
    metadata_type.id = type_id
    return metadata_type
  except mlmd.errors.AlreadyExistsError:
    type_name = metadata_type.name
    existing_type = get_type_handler(type_name)
    assert existing_type is not None, (
        'Not expected to get None when getting type %s.' % type_name)

    # If the existing type is a super set of the proposed type, directly use it.
    # Otherwise, there is a type conflict since the AlreadyExistsError already
    # indicates the existing type is not a subset of the proposed type.
    if all(
        existing_type.properties.get(k) == metadata_type.properties.get(k)
        for k in metadata_type.properties.keys()):
      return existing_type
    else:
      warning_str = (
          'Missing or modified key in properties comparing with '
          'existing metadata type with the same type name. Existing type: '
          '%s, New type: %s') % (existing_type, metadata_type)
      logging.warning(warning_str)
      raise RuntimeError(warning_str)
