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

import json
import builtins

from typing import Any, Dict, Optional, Text

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


class Artifact(object):
  """TFX artifact used for orchestration.

  This is used for type-checking and inter-component communication. Currently,
  it wraps a tuple of (ml_metadata.proto.Artifact,
  ml_metadata.proto.ArtifactType) with additional property accessors for
  internal state.
  """

  def __init__(self, type_name: Text, split: Optional[Text] = ''):
    """Construct an instance of Artifact.

    Used by TFX internal implementation: create an empty Artifact with
    type_name and optional split info specified. The remaining info will be
    filled in during compiling and running time. The Artifact should be
    transparent to end users and should not be initiated directly by pipeline
    users.

    Args:
      type_name: Name of underlying ArtifactType.
      split: Which split this instance of articact maps to.
    """
    artifact_type = metadata_store_pb2.ArtifactType()
    artifact_type.name = type_name
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
    # TODO(b/135056715): Rely on MLMD context for pipeline grouping for
    # artifacts once it's ready.
    # The name of the pipeline that produces the artifact.
    artifact_type.properties['pipeline_name'] = metadata_store_pb2.STRING
    # The name of the component that produces the artifact.
    artifact_type.properties['producer_component'] = metadata_store_pb2.STRING
    # The name of the artifact, used to differentiate same type of artifact
    # produced by the same component.
    artifact_type.properties['name'] = metadata_store_pb2.STRING

    self.artifact_type = artifact_type

    artifact = metadata_store_pb2.Artifact()
    artifact.properties['type_name'].string_value = type_name
    artifact.properties['split'].string_value = split

    self.artifact = artifact

  def __repr__(self):
    return 'Artifact(type_name: {}, uri: {}, split: {}, id: {})'.format(
        self.artifact_type.name, self.uri, str(self.split), str(self.id))

  def json_dict(self) -> Dict[Text, Any]:
    """Returns a dict suitable for json serialization."""
    return {
        'artifact':
            json.loads(json_format.MessageToJson(self.artifact)),
        'artifact_type':
            json.loads(json_format.MessageToJson(self.artifact_type)),
    }

  @classmethod
  def parse_from_json_dict(cls, d: Dict[Text, Any]):
    """Creates a instance of Artifact from a json deserialized dict."""
    artifact = metadata_store_pb2.Artifact()
    json_format.Parse(json.dumps(d['artifact']), artifact)
    artifact_type = metadata_store_pb2.ArtifactType()
    json_format.Parse(json.dumps(d['artifact_type']), artifact_type)
    result = Artifact(artifact_type.name)
    result.set_artifact_type(artifact_type)
    result.set_artifact(artifact)
    return result

  @property
  def uri(self) -> Text:
    """URI of underlying artifact."""
    return self.artifact.uri

  @uri.setter
  def uri(self, uri: Text):
    """Set URI of underlying artifact."""
    self.artifact.uri = uri

  @property
  def id(self) -> int:
    """Id of underlying artifact."""
    return self.artifact.id

  @id.setter
  def id(self, artifact_id: int):
    """Set id of underlying artifact."""
    self.artifact.id = artifact_id

  @property
  def span(self) -> int:
    """Span of underlying artifact."""
    return self.artifact.properties['span'].int_value

  @span.setter
  def span(self, span: int):
    """Set span of underlying artifact."""
    self.artifact.properties['span'].int_value = span

  @property
  def type_id(self) -> int:
    """Id of underlying artifact type."""
    return self.artifact.type_id

  @type_id.setter
  def type_id(self, type_id: int):
    """Set id of underlying artifact type."""
    self.artifact.type_id = type_id

  @property
  def type_name(self) -> Text:
    """Name of underlying artifact type."""
    return self.artifact_type.name

  @property
  def state(self) -> Text:
    """State of the underlying artifact."""
    return self.artifact.properties['state'].string_value

  @state.setter
  def state(self, state: Text):
    """Set state of the underlying artifact."""
    self.artifact.properties['state'].string_value = state

  @property
  def split(self) -> Text:
    """Split of the underlying artifact is in."""
    return self.artifact.properties['split'].string_value

  @split.setter
  def split(self, split: Text):
    """Set state of the underlying artifact."""
    self.artifact.properties['split'].string_value = split

  @property
  def pipeline_name(self) -> Text:
    """Name of the pipeline that produce the artifact."""
    return self.artifact.properties['pipeline_name'].string_value

  @pipeline_name.setter
  def pipeline_name(self, pipeline_name: Text):
    """Set name of the pipeline that produce the artifact."""
    self.artifact.properties['pipeline_name'].string_value = pipeline_name

  @property
  def producer_component(self) -> Text:
    """Producer component of the artifact."""
    return self.artifact.properties['producer_component'].string_value

  @producer_component.setter
  def producer_component(self, producer_component: Text):
    """Set producer component of the artifact."""
    self.artifact.properties[
        'producer_component'].string_value = producer_component

  @property
  def name(self) -> Text:
    """Name of the artifact."""
    return self.artifact.properties['name'].string_value

  @name.setter
  def name(self, name: Text):
    """Set the name of the artifact."""
    self.artifact.properties['name'].string_value = name

  def set_artifact(self, artifact: metadata_store_pb2.Artifact):
    """Set entire artifact in this object."""
    self.artifact = artifact

  def set_artifact_type(self, artifact_type: metadata_store_pb2.ArtifactType):
    """Set entire ArtifactType in this object."""
    self.artifact_type = artifact_type
    self.artifact.type_id = artifact_type.id

  def set_string_custom_property(self, key: Text, value: Text):
    """Set a custom property of string type."""
    self.artifact.custom_properties[key].string_value = value

  def set_int_custom_property(self, key: Text, value: int):
    """Set a custom property of int type."""
    self.artifact.custom_properties[key].int_value = builtins.int(value)
