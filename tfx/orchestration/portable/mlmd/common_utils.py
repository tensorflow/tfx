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

import copy
from typing import Any, TypeVar

from absl import logging
from tfx.orchestration import metadata

import ml_metadata as mlmd
from ml_metadata.proto import metadata_store_pb2


MetadataType = TypeVar(
    'MetadataType',
    metadata_store_pb2.ArtifactType,
    metadata_store_pb2.ContextType,
    metadata_store_pb2.ExecutionType,
)


def _assert_metadata_type_match(
    queried_metadata_type: MetadataType, new_metadata_type: MetadataType
) -> bool:
  a = copy.deepcopy(queried_metadata_type)
  b = copy.deepcopy(new_metadata_type)
  a.ClearField('id')
  b.ClearField('id')
  return a == b


def register_type_if_not_exist(
    metadata_handle: metadata.Metadata,
    metadata_type: MetadataType,
) -> MetadataType:
  """Registers a metadata type if not exists.

  Uses existing type if schema is superset of what is needed. Otherwise tries
  to register new metadata type.

  Args:
    metadata_handle: A handler to access MLMD store.
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
    get_type_handler = metadata_handle.store.get_artifact_type
    put_type_handler = metadata_handle.store.put_artifact_type
  elif isinstance(metadata_type, metadata_store_pb2.ContextType):
    get_type_handler = metadata_handle.store.get_context_type
    put_type_handler = metadata_handle.store.put_context_type
  elif isinstance(metadata_type, metadata_store_pb2.ExecutionType):
    get_type_handler = metadata_handle.store.get_execution_type
    put_type_handler = metadata_handle.store.put_execution_type
  else:
    raise ValueError('Unexpected value type: %s.' % type(metadata_type))

  try:
    queried_metadata_type = get_type_handler(metadata_type.name)
    if queried_metadata_type is not None:
      assert _assert_metadata_type_match(queried_metadata_type, metadata_type)
      return queried_metadata_type
  except (mlmd.errors.NotFoundError, AssertionError):
    res = _put_type_handler(get_type_handler, put_type_handler, metadata_type)
    return res


def _put_type_handler(
    get_type_handler: Any, put_type_handler: Any, metadata_type: MetadataType
) -> MetadataType:
  """register the type with error handling.

  Args:
    get_type_handler: A handler to access MLMD store.
    put_type_handler: A handler to access MLMD store.
    metadata_type: The metadata type to register if does not exist.

  Returns:
    A MetadataType with id

  Raises:
    RuntimeError: If new metadata type conflicts with existing schema in MLMD.
    ValueError: If metadata type is not expected.
  """
  try:
    put_type_handler(metadata_type, can_add_fields=True, can_omit_fields=True)
    return get_type_handler(metadata_type.name)
  except mlmd.errors.AlreadyExistsError as exc:
    existing_type = get_type_handler(metadata_type.name)
    assert existing_type is not None, (
        'Not expected to get None when getting type %s.' % metadata_type.name)
    warning_str = (
        'Conflicting properties comparing with existing metadata type '
        'with the same type name. Existing type: '
        '%s, New type: %s, error: %s') % (existing_type, metadata_type, exc)
    logging.warning(warning_str)
    raise RuntimeError(warning_str) from exc
