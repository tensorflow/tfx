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
import json
from typing import Any, Dict, Iterable, Optional, Text, Type
import absl

from tfx.types import artifact_utils
from tfx.types.artifact import Artifact
from tfx.utils import json_utils
from google.protobuf import json_format
from ml_metadata.proto import metadata_store_pb2


class Channel(json_utils.Jsonable):
  """Tfx Channel.

  TFX Channel is an abstract concept that connects data producers and data
  consumers. It contains restriction of the artifact type that should be fed
  into or read from it.

  Attributes:
    type: The artifact type class that the Channel takes.
  """

  # TODO(b/125348988): Add support for real Channel in addition to static ones.
  def __init__(
      self,
      type: Optional[Type[Artifact]] = None,  # pylint: disable=redefined-builtin
      artifacts: Optional[Iterable[Artifact]] = None,
      matching_channel_name: Optional[Text] = None,
      producer_component_id: Optional[Text] = None,
      output_key: Optional[Text] = None):
    """Initialization of Channel.

    Args:
      type: Subclass of Artifact that represents the type of this Channel.
      artifacts: (Optional) A collection of artifacts as the values that can be
        read from the Channel. This is used to construct a static Channel.
      matching_channel_name: This targets to the key of an input Channel dict
        in a Component. The artifacts count of this channel will be decided at
        runtime in Driver, based on the artifacts count of the target channel.
        Only one of `artifacts` and `matching_channel_name` should be set.
      producer_component_id: (Optional) Producer component id of the Channel.
      output_key: (Optional) The output key when producer component produces
        the artifacts in this Channel.
    """
    if not (inspect.isclass(type) and issubclass(type, Artifact)):  # pytype: disable=wrong-arg-types
      raise ValueError(
          'Argument "type" of Channel constructor must be a subclass of '
          'tfx.Artifact (got %r).' % (type,))

    self.type = type
    self._artifacts = artifacts or []
    self.matching_channel_name = matching_channel_name
    if self.matching_channel_name and self._artifacts:
      # TODO(b/161548528): change to error after MP support multiple artifacts.
      absl.logging.warning(
          'Only one of `artifacts` and `matching_channel_name` should be set.')
    self._validate_type()
    # The following fields will be populated during compilation time.
    self.producer_component_id = producer_component_id
    self.output_key = output_key

  @property
  def type_name(self):
    return self.type.TYPE_NAME

  def __repr__(self):
    artifacts_str = '\n    '.join(repr(a) for a in self._artifacts)
    return 'Channel(\n    type_name: {}\n    artifacts: [{}]\n)'.format(
        self.type_name, artifacts_str)

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

  def to_json_dict(self) -> Dict[Text, Any]:
    return {
        'type':
            json.loads(
                json_format.MessageToJson(
                    message=self.type._get_artifact_type(),  # pylint: disable=protected-access
                    preserving_proto_field_name=True)),
        'artifacts':
            list(a.to_json_dict() for a in self._artifacts),
        'producer_component_id':
            (self.producer_component_id if self.producer_component_id else None
            ),
        'output_key': (self.output_key if self.output_key else None),
    }

  @classmethod
  def from_json_dict(cls, dict_data: Dict[Text, Any]) -> Any:
    artifact_type = metadata_store_pb2.ArtifactType()
    json_format.Parse(json.dumps(dict_data['type']), artifact_type)
    type_cls = artifact_utils.get_artifact_type_class(artifact_type)
    artifacts = list(Artifact.from_json_dict(a) for a in dict_data['artifacts'])
    producer_component_id = dict_data.get('producer_component_id', None)
    output_key = dict_data.get('output_key', None)
    return Channel(
        type=type_cls,
        artifacts=artifacts,
        producer_component_id=producer_component_id,
        output_key=output_key)
