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
"""TFX Channel definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect

from typing import Any, Dict, Iterable, Optional, Text, Type

from google.protobuf import json_format
from ml_metadata.proto import metadata_store_pb2
from tfx.types.artifact import Artifact
from tfx.utils import json_utils


class Channel(json_utils.Jsonable):
  """Tfx Channel.

  TFX Channel is an abstract concept that connects data producers and data
  consumers. It contains restriction of the artifact type that should be fed
  into or read from it.

  Attributes:
    mlmd_artifact_type: The MLMD artifact type protobuf describing the artifact
      type taken by this channel.
  """

  # TODO(b/125348988): Add support for real Channel in addition to static ones.
  def __init__(
      self,
      type: Optional[Type[Artifact]] = None,  # pylint: disable=redefined-builtin
      mlmd_artifact_type: Optional[metadata_store_pb2.ArtifactType] = None,
      artifacts: Optional[Iterable[Artifact]] = None,
      producer_component_id: Optional[Text] = None,
      output_key: Optional[Text] = None):
    """Initialization of Channel.

    Args:
      type: Subclass of Artifact that represents the type of this Channel.
        Recommended way to construct a channel when an Artifact subclass for
        the channel's type is available.
      mlmd_artifact_type: Proto message defining the underlying ArtifactType.
        Used in place of `type` for constructing the Channel when an Artifact
        subclass for the MLMD artifact type is not available. Optional and
        intended for internal use.
      artifacts: (Optional) A collection of artifacts as the values that can be
        read from the Channel. This is used to construct a static Channel.
      producer_component_id: (Optional) Producer component id of the Channel.
      output_key: (Optional) The output key when producer component produces
        the artifacts in this Channel.
    """
    if type:
      if not (inspect.isclass(type) and issubclass(type, Artifact)):  # pytype: disable=wrong-arg-types
        raise ValueError(
            ('Argument "type" of Channel constructor must be a subclass of '
             'tfx.Artifact (got %r).') % (type,))
      mlmd_artifact_type = type._construct_artifact_type()  # pylint: disable=protected-access
    elif mlmd_artifact_type:
      if not isinstance(mlmd_artifact_type, metadata_store_pb2.ArtifactType):
        raise ValueError(
            ('Argument "mlmd_artifact_type" of Channel constructor must be a '
             'protobuf message of type metadata_store_pb2.ArtifactType '
             '(got %r)') % (mlmd_artifact_type,))
    else:
      raise ValueError(
          'Channel constructor must be passed one of the "type" or '
          '"mlmd_artifact_type" arguments.')

    self.mlmd_artifact_type = mlmd_artifact_type
    self._artifacts = artifacts or []
    self._validate_type()
    # The following fields will be populated during compilation time.
    self.producer_component_id = producer_component_id
    self.output_key = output_key

  @property
  def type_name(self):
    return self.mlmd_artifact_type.name

  def __repr__(self):
    artifacts_str = '\n    '.join(repr(a) for a in self._artifacts)
    return 'Channel(\n    type_name: {}\n    artifacts: [{}]\n)'.format(
        self.type_name, artifacts_str)

  def to_json_dict(self) -> Dict[Text, Any]:
    return {
        'artifact_type':
            json.loads(
                json_format.MessageToJson(
                    message=self.mlmd_artifact_type,
                    preserving_proto_field_name=True)),
        'artifacts':
            list(a.to_json_dict() for a in self._artifacts),
        'producer_info': (
            self.producer_info.to_json_dict() if self.producer_info else None),
    }

  @classmethod
  def from_json_dict(cls, dict_data: Dict[Text, Any]) -> Any:
    artifact_type = metadata_store_pb2.ArtifactType()
    json_format.Parse(json.dumps(dict_data['artifact_type']), artifact_type)
    artifacts = list(Artifact.from_json_dict(a) for a in dict_data['artifacts'])
    producer_info = ChannelProducerInfo.from_json_dict(
        dict_data['producer_info']) if dict_data.get('producer_info') else None
    return Channel(
        mlmd_artifact_type=artifact_type,
        artifacts=artifacts,
        producer_info=producer_info)

  def _validate_type(self) -> None:
    for artifact in self._artifacts:
      if artifact.type_name != self.type_name:
        raise ValueError(
            "Artifacts provided do not match Channel's artifact type {}".format(
                self.type_name))

  def get(self) -> Iterable[Artifact]:
    """Returns all artifacts that can be get from this Channel.

    Returns:
      An artifact collection.
    """
    # TODO(b/125037186): We should support dynamic query against a Channel
    # instead of a static Artifact collection.
    return self._artifacts
