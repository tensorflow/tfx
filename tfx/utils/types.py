# Copyright 2019 Google LLC. All Rights Reserved.
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
"""TFX type definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json

from typing import Any
from typing import Dict
from typing import List
from typing import Text

from ml_metadata.metadata_store import types
from ml_metadata.proto import metadata_store_pb2

# Indicating there is an execution producing it.
ARTIFACT_STATE_PENDING = 'pending'
# Indicating artifact ready to be consumed.
ARTIFACT_STATE_PUBLISHED = 'published'
# Indicating no data in artifact uri although it's not marked as deleted.
ARTIFACT_STATE_MISSING = 'missing'
# Indicating artifact should be garbage collected.
ARTIFACT_STATE_MARKED_FOR_DELETION = 'MARKED_FOR_DELETION'
# Indicating artifact being garbage collected.
ARTIFACT_STATE_DELETED = 'deleted'

# Default split of examples data.
DEFAULT_EXAMPLE_SPLITS = ['train', 'eval']


class TfxType(types.Artifact):
  """Base Tfx Type used for orchestration.

  This is used for type checking and inter component communicating. Currently
  it wraps a tupel of
  (ml_metadata.proto.Artifact, ml_metadata.proto.ArtifactType)
  with additional properties for accessing internal state.
  """

  def __init__(self, type_name, split = ''):
    """Construct an instance of TfxType.

    Each instance of TfxTypes wraps an Artifact and its type internally. When
    first created, the artifact will have an empty URI (which will be filled by
    orchestration system before first usage).

    Args:
      type_name: Name of underlying ArtifactType.
      split: Which split this instance of articact maps to.
    """
    super(TfxType, self).__init__(
        _create_tfx_artifact(type_name, split),
        _create_tfx_artifact_type(type_name))
    self.__dict__['source'] = None

  def __str__(self):
    return '{}:{}.{}'.format(self.type.name, self.uri, str(self.id))

  def __repr__(self):
    return self.__str__()

  def __deepcopy__(self, memo):
    """Create a deep copy of the artifact.

    Note that if the artifact has an ID then this is copied as well.

    Args:
      memo: information for memoization (not used)

    Returns:
      a deep copy with a new ID and new state. All information is erased.
    """
    del memo
    result = TfxType(self.type.name, self.split)
    result.set_artifact_and_type(
        copy.deepcopy(self.artifact), copy.deepcopy(self.type))
    return result

  def json_dict(self):
    """Returns a dict suitable for json serialization."""
    # TODO(martinz): reconcile these two method names.
    # Decide if this should be public or private.
    return self._create_pre_json()

  @classmethod
  def from_types_artifact(cls, art):
    """Create a TfxType from an types.Artifact object."""
    result = TfxType(art.type.name, art.uri)
    result.set_artifact_and_type(art.artifact, art.type)
    return result

  @classmethod
  def parse_from_json_dict(cls, d):
    """Creates a instance of TfxType from a json deserialized dict."""
    return TfxType.from_types_artifact(types.Artifact.from_json(d))

  @property
  def type_id(self):
    """Id of underlying artifact type."""
    return self.artifact.type_id

  @property
  def artifact_type(self):
    """The underlying artifact type."""
    return self.type

  def __setattr__(self, attr, value):
    """Set an attribute."""
    # We need this because __setattr__ in the base clase has a higher
    # priority than setting a property.
    if attr == 'id':
      self.artifact.id = value
    elif attr == 'type_id':
      self.artifact.type_id = value
      self.type.id = value
    elif attr == 'source':
      self.__dict__['source'] = value
    else:
      super(TfxType, self).__setattr__(attr, value)

  @property
  def type_name(self):
    """Name of underlying artifact type."""
    return self.type.name

  @property
  def source(self):
    """Name of underlying artifact type."""
    return self.__dict__['source']

  def set_artifact(self, artifact):
    """Set entire artifact in this object."""
    # TODO(martinz): check for type consistency.
    self.__dict__['_artifact'] = artifact

  def set_artifact_type(self, artifact_type):
    """Set entire ArtifactType in this object."""
    # TODO(martinz): check for type consistency.
    self.__dict__['_type'] = artifact_type
    self.artifact.type_id = artifact_type.id

  def set_artifact_and_type(self, artifact,
                            artifact_type):
    self.__dict__['_artifact'] = artifact
    self.__dict__['_type'] = artifact_type
    if not self._is_consistent():
      raise ValueError('Type is not internally consistent')

  def __setstate__(self, state):
    self.set_artifact_and_type(state[0], state[1])

  def __getstate__(self):
    return self.artifact, self.type

  def set_string_custom_property(self, key, value):
    """Set a custom property of string type."""
    self.set_custom_property(key, value)

  def set_int_custom_property(self, key, value):
    """Set a custom property of int type."""
    self.set_custom_property(key, int(value))


def parse_tfx_type_dict(json_str):
  """Parse a dict from key to dictonary of TfxType from its json format."""
  # TODO(martinz): See types.ArtifactStruct.
  tfx_artifacts = {}
  for k, l in json.loads(json_str).items():
    tfx_artifacts[k] = [TfxType.parse_from_json_dict(v) for v in l]
  return tfx_artifacts


def jsonify_tfx_type_dict(artifact_dict):
  """Serialize a dict from key to list of TfxType into json format."""
  return types.create_json(artifact_dict)


def get_single_instance(artifact_list):
  """Get a single instance of TfxType from a list of length one.

  Args:
    artifact_list: A list of TfxType objects whose length must be one.

  Returns:
    The single TfxType object in artifact_list.

  Raises:
    ValueError: If length of artifact_list is not one.
  """
  if len(artifact_list) != 1:
    raise ValueError('expected list length of one but got {}'.format(
        len(artifact_list)))
  return artifact_list[0]


def get_single_uri(artifact_list):
  """Get the uri of TfxType from a list of length one.

  Args:
    artifact_list: A list of TfxType objects whose length must be one.

  Returns:
    The uri of the single TfxType object in artifact_list.

  Raises:
    ValueError: If length of artifact_list is not one.
  """
  return get_single_instance(artifact_list).uri


def _get_split_instance(artifact_list, split):
  """Get an instance of TfxType with matching split from given list.

  Args:
    artifact_list: A list of TfxType objects whose length must be one.
    split: Name of split.

  Returns:
    The single TfxType object in artifact_list with matching split.

  Raises:
    ValueError: If number with matching split in artifact_list is not one.
  """
  matched = [x for x in artifact_list if x.split == split]
  if len(matched) != 1:
    raise ValueError('{} elements matches split {}'.format(len(matched), split))
  return matched[0]


def get_split_uri(artifact_list, split):
  """Get the uri of TfxType with matching split from given list.

  Args:
    artifact_list: A list of TfxType objects whose length must be one.
    split: Name of split.

  Returns:
    The uri of TfxType object in artifact_list with matching split.

  Raises:
    ValueError: If number with matching split in artifact_list is not one.
  """
  return _get_split_instance(artifact_list, split).uri


def _create_tfx_artifact_type(type_name
                             ):
  """Create an ArtifactType for TfxType.__init__()."""
  artifact_type = metadata_store_pb2.ArtifactType()
  artifact_type.name = type_name
  # TODO(martinz): remove type_name as a property.
  artifact_type.properties['type_name'] = metadata_store_pb2.STRING
  # This indicates the state of an artifact. A state can be any of the
  # followings: PENDING, PUBLISHED, MISSING, DELETING, DELETED
  # TODO(ruoyu): Maybe switch to artifact top-level state if it's supported.
  artifact_type.properties['state'] = metadata_store_pb2.STRING
  # Span number of an artifact. For the same artifact type produced by the
  # same executor, this number should always increase.
  artifact_type.properties['span'] = metadata_store_pb2.INT
  # Comma separated splits recognized. Empty string means artifact has no
  # split.
  artifact_type.properties['split'] = metadata_store_pb2.STRING
  return artifact_type


def _create_tfx_artifact(type_name,
                         split):
  """Create an Artifact for TfxType.__init__()."""
  artifact = metadata_store_pb2.Artifact()
  # TODO(martinz): consider whether type_name needs to be hard-coded into the
  # artifact.
  artifact.properties['type_name'].string_value = type_name
  artifact.properties['split'].string_value = split
  return artifact
