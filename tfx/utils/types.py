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

import json
import builtins

from typing import Any
from typing import Dict
from typing import List
from typing import Text

from ml_metadata.proto import metadata_store_pb2
from google.protobuf import json_format

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


class TfxType(object):
  """Base Tfx Type used for orchestration.

  This is used for type checking and inter component communicating. Currently
  it wraps a tuple of
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
    artifact_type = metadata_store_pb2.ArtifactType()
    artifact_type.name = type_name
    artifact_type.properties['type_name'] = metadata_store_pb2.STRING
    # This is a temporary solution due to b/123435989.
    artifact_type.properties['name'] = metadata_store_pb2.STRING
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

    self.artifact_type = artifact_type

    artifact = metadata_store_pb2.Artifact()
    artifact.properties['type_name'].string_value = type_name
    artifact.properties['split'].string_value = split

    self.artifact = artifact

  def __str__(self):
    return '{}:{}.{}'.format(self.artifact_type.name, self.uri, str(self.id))

  def __repr__(self):
    return self.__str__()

  def json_dict(self):
    """Returns a dict suitable for json serialization."""
    return {
        'artifact':
            json.loads(json_format.MessageToJson(self.artifact)),
        'artifact_type':
            json.loads(json_format.MessageToJson(self.artifact_type)),
    }

  @classmethod
  def parse_from_json_dict(cls, d):
    """Creates a instance of TfxType from a json deserialized dict."""
    artifact = metadata_store_pb2.Artifact()
    json_format.Parse(json.dumps(d['artifact']), artifact)
    artifact_type = metadata_store_pb2.ArtifactType()
    json_format.Parse(json.dumps(d['artifact_type']), artifact_type)
    result = TfxType(artifact_type.name, artifact.uri)
    result.set_artifact_type(artifact_type)
    result.set_artifact(artifact)
    return result

  @property
  def uri(self):
    """URI of underlying artifact."""
    return self.artifact.uri

  @uri.setter
  def uri(self, uri):
    """Set URI of underlying artifact."""
    self.artifact.uri = uri

  @property
  def id(self):
    """Id of underlying artifact."""
    return self.artifact.id

  @id.setter
  def id(self, artifact_id):
    """Set id of underlying artifact."""
    self.artifact.id = artifact_id

  @property
  def span(self):
    """Span of underlying artifact."""
    return self.artifact.properties['span'].int_value

  @span.setter
  def span(self, span):
    """Set span of underlying artifact."""
    self.artifact.properties['span'].int_value = span

  @property
  def type_id(self):
    """Id of underlying artifact type."""
    return self.artifact.type_id

  @type_id.setter
  def type_id(self, type_id):
    """Set id of underlying artifact type."""
    self.artifact.type_id = type_id

  @property
  def type_name(self):
    """Name of underlying artifact type."""
    return self.artifact_type.name

  @property
  def state(self):
    """State of the underlying artifact."""
    return self.artifact.properties['state'].string_value

  @state.setter
  def state(self, state):
    """Set state of the underlying artifact."""
    self.artifact.properties['state'].string_value = state

  @property
  def split(self):
    """Split of the underlying artifact is in."""
    return self.artifact.properties['split'].string_value

  @split.setter
  def split(self, split):
    """Set state of the underlying artifact."""
    self.artifact.properties['split'].string_value = split

  def set_artifact(self, artifact):
    """Set entire artifact in this object."""
    self.artifact = artifact

  def set_artifact_type(self, artifact_type):
    """Set entire ArtifactType in this object."""
    self.artifact_type = artifact_type
    self.artifact.type_id = artifact_type.id

  def set_string_custom_property(self, key, value):
    """Set a custom property of string type."""
    self.artifact.custom_properties[key].string_value = value

  def set_int_custom_property(self, key, value):
    """Set a custom property of int type."""
    self.artifact.custom_properties[key].int_value = builtins.int(value)


def parse_tfx_type_dict(json_str):
  """Parse a dict from key to list of TfxType from its json format."""
  tfx_artifacts = {}
  for k, l in json.loads(json_str).items():
    tfx_artifacts[k] = [TfxType.parse_from_json_dict(v) for v in l]
  return tfx_artifacts


def jsonify_tfx_type_dict(artifact_dict):
  """Serialize a dict from key to list of TfxType into json format."""
  d = {}
  for k, l in artifact_dict.items():
    d[k] = [v.json_dict() for v in l]
  return json.dumps(d)


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
