# Lint as: python2, python3
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
"""TFX artifact type definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import builtins
import json
from typing import Any, Dict, Optional, Text

from google.protobuf import json_format
from ml_metadata.proto import metadata_store_pb2
from tfx.utils import json_utils


class ArtifactState(object):
  """Enumeration of possible Artifact states."""

  # Indicates that there is a pending execution producing the artifact.
  PENDING = 'pending'
  # Indicates that the artifact ready to be consumed.
  PUBLISHED = 'published'
  # Indicates that the no data at the artifact uri, though the artifact is not
  # marked as deleted.
  MISSING = 'missing'
  # Indicates that the artifact should be garbage collected.
  MARKED_FOR_DELETION = 'MARKED_FOR_DELETION'
  # Indicates that the artifact has been garbage collected.
  DELETED = 'deleted'


# Default split of examples data.
DEFAULT_EXAMPLE_SPLITS = ['train', 'eval']


class Artifact(json_utils.Jsonable):
  """TFX artifact used for orchestration.

  This is used for type-checking and inter-component communication. Currently,
  it wraps a tuple of (ml_metadata.proto.Artifact,
  ml_metadata.proto.ArtifactType) with additional property accessors for
  internal state.

  A user may create a subclass of Artifact and override the TYPE_NAME property
  with the type for this artifact subclass. Users of the subclass may then omit
  the "type_name" field when construction the object.

  Note: the behavior of this class is experimental, without backwards
  compatibility guarantees, and may change in upcoming releases.
  """

  TYPE_NAME = None

  def __init__(self, type_name: Optional[Text] = None):
    """Construct an instance of Artifact.

    Used by TFX internal implementation: create an empty Artifact with
    type_name and optional split info specified. The remaining info will be
    filled in during compiling and running time. The Artifact should be
    transparent to end users and should not be initiated directly by pipeline
    users.

    Args:
      type_name: Name of underlying ArtifactType (optional if the ARTIFACT_TYPE
        field is provided for the Artifact subclass).
    """
    # TODO(b/138664975): either deprecate or remove string-based artifact type
    # definition before 0.16.0 release.
    if self.__class__ != Artifact:
      if type_name:
        raise ValueError(
            ('The "type_name" field must not be passed for Artifact subclass '
             '%s.') % self.__class__)
      type_name = self.__class__.TYPE_NAME
      if not (type_name and isinstance(type_name, (str, Text))):
        raise ValueError(
            ('The Artifact subclass %s must override the TYPE_NAME attribute '
             'with a string type name identifier (got %r instead).') %
            (self.__class__, type_name))

    if not type_name:
      raise ValueError(
          'The "type_name" field must be passed to specify a type for this '
          'Artifact.')

    # Type name string.
    self._type_name = type_name
    # MLMD artifact type proto object.
    self._artifact_type = self._construct_artifact_type(type_name)
    # Underlying MLMD artifact proto object.
    self._artifact = metadata_store_pb2.Artifact()

  def _construct_artifact_type(self, type_name):
    artifact_type = metadata_store_pb2.ArtifactType()
    artifact_type.name = type_name
    # Comma separated of splits for an artifact. Empty string means artifact
    # has no split. This will be removed soon and replaced with artifact
    # type-specific properties.
    artifact_type.properties['split_names'] = metadata_store_pb2.STRING
    return artifact_type

  def set_mlmd_artifact(self, artifact: metadata_store_pb2.Artifact):
    """Replace the MLMD artifact object on this artifact."""
    self._artifact = artifact

  def set_mlmd_artifact_type(self,
                             artifact_type: metadata_store_pb2.ArtifactType):
    """Set entire ArtifactType in this object."""
    self._artifact_type = artifact_type
    self._artifact.type_id = artifact_type.id

  def __repr__(self):
    return 'Artifact(type_name: {}, uri: {}, id: {})'.format(
        self._artifact_type.name, self.uri, str(self.id))

  def to_json_dict(self) -> Dict[Text, Any]:
    return {
        'artifact':
            json.loads(
                json_format.MessageToJson(
                    message=self._artifact, preserving_proto_field_name=True)),
        'artifact_type':
            json.loads(
                json_format.MessageToJson(
                    message=self._artifact_type,
                    preserving_proto_field_name=True)),
    }

  @classmethod
  def from_json_dict(cls, dict_data: Dict[Text, Any]) -> Any:
    artifact = metadata_store_pb2.Artifact()
    json_format.Parse(json.dumps(dict_data['artifact']), artifact)
    artifact_type = metadata_store_pb2.ArtifactType()
    json_format.Parse(json.dumps(dict_data['artifact_type']), artifact_type)
    result = Artifact(artifact_type.name)
    result.set_mlmd_artifact_type(artifact_type)
    result.set_mlmd_artifact(artifact)
    return result

  # Read-only properties.
  @property
  def type_name(self):
    return self._type_name

  @property
  def artifact_type(self):
    return self._artifact_type

  @property
  def mlmd_artifact(self):
    return self._artifact

  # Settable properties for all artifact types.
  @property
  def uri(self) -> Text:
    """Artifact URI."""
    return self._artifact.uri

  @uri.setter
  def uri(self, uri: Text):
    """Setter for artifact URI."""
    self._artifact.uri = uri

  @property
  def id(self) -> int:
    """Id of underlying artifact."""
    return self._artifact.id

  @id.setter
  def id(self, artifact_id: int):
    """Set id of underlying artifact."""
    self._artifact.id = artifact_id

  @property
  def type_id(self) -> int:
    """Id of underlying artifact type."""
    return self._artifact.type_id

  @type_id.setter
  def type_id(self, type_id: int):
    """Set id of underlying artifact type."""
    self._artifact.type_id = type_id

  # System-managed properties for all artifact types. Will be deprecated soon
  # in favor of a unified getter / setter interface and MLMD context.
  #
  # TODO(b/135056715): Rely on MLMD context for pipeline grouping for
  # artifacts once it's ready.
  #
  # The following system properties are used:
  #   - name: The name of the artifact, used to differentiate same type of
  #       artifact produced by the same component (in a subsequent change, this
  #       information will move to the associated ML Metadata Event object).
  #   - state: The state of an artifact; can be one of PENDING, PUBLISHED,
  #       MISSING, DELETING, DELETED (in a subsequent change, this information
  #       will move to a top-level ML Metadata Artifact attribute).
  #   - pipeline_name: The name of the pipeline that produces the artifact (in
  #       a subsequent change, this information will move to an associated ML
  #       Metadata Context attribute).
  #   - producer_component: The name of the component that produces the
  #       artifact (in a subsequent change, this information will move to the
  #       associated ML Metadata Event object).
  def _get_system_property(self, key: Text) -> Text:
    if key in self._artifact.custom_properties:
      return self._artifact.custom_properties[key].string_value
    if (key in self._artifact_type.properties and
        key in self._artifact.properties):
      # Legacy artifact types which have explicitly defined system properties.
      return self._artifact.properties[key].string_value
    return ''

  def _set_system_property(self, key: Text, value: Text):
    if (key in self._artifact_type.properties and
        key in self._artifact.properties):
      # Clear non-custom property in legacy artifact types.
      del self._artifact.properties[key]
    self._artifact.custom_properties[key].string_value = value

  @property
  def name(self) -> Text:
    """Name of the underlying artifact."""
    return self._get_system_property('name')

  @name.setter
  def name(self, name: Text):
    """Set name of the underlying artifact."""
    self._set_system_property('name', name)

  @property
  def state(self) -> Text:
    """State of the underlying artifact."""
    return self._get_system_property('state')

  @state.setter
  def state(self, state: Text):
    """Set state of the underlying artifact."""
    self._set_system_property('state', state)

  @property
  def pipeline_name(self) -> Text:
    """Name of the pipeline that produce the artifact."""
    return self._get_system_property('pipeline_name')

  @pipeline_name.setter
  def pipeline_name(self, pipeline_name: Text):
    """Set name of the pipeline that produce the artifact."""
    self._set_system_property('pipeline_name', pipeline_name)

  @property
  def producer_component(self) -> Text:
    """Producer component of the artifact."""
    return self._get_system_property('producer_component')

  @producer_component.setter
  def producer_component(self, producer_component: Text):
    """Set producer component of the artifact."""
    self._set_system_property('producer_component', producer_component)

  # Type-specific artifacts properties. Will be deprecated soon in favor of a
  # unified getter / setter interface.
  @property
  def span(self) -> int:
    """Span of underlying artifact."""
    return self._artifact.properties['span'].int_value

  @span.setter
  def span(self, span: int):
    """Set span of underlying artifact."""
    self._artifact.properties['span'].int_value = span

  @property
  def split_names(self) -> Text:
    """Split of the underlying artifact is in."""
    return self._artifact.properties['split_names'].string_value

  @split_names.setter
  def split_names(self, split: Text):
    """Set state of the underlying artifact."""
    self._artifact.properties['split_names'].string_value = split

  # Custom property accessors.
  def set_string_custom_property(self, key: Text, value: Text):
    """Set a custom property of string type."""
    self._artifact.custom_properties[key].string_value = value

  def set_int_custom_property(self, key: Text, value: int):
    """Set a custom property of int type."""
    self._artifact.custom_properties[key].int_value = builtins.int(value)

  def get_string_custom_property(self, key: Text, value: Text) -> Text:
    """Get a custom property of string type."""
    return self._artifact.custom_properties[key].string_value

  def get_int_custom_property(self, key: Text) -> int:
    """Get a custom property of int type."""
    return self._artifact.custom_properties[key].int_value
