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

import inspect
import json
import textwrap
from typing import Any, cast, Dict, Iterable, List, Optional, Type, Union
from absl import logging

from tfx.dsl.placeholder import placeholder
from tfx.types import artifact_utils
from tfx.types.artifact import Artifact
from tfx.utils import deprecation_utils
from tfx.utils import doc_controls
from tfx.utils import json_utils
from google.protobuf import json_format
from google.protobuf import message
from ml_metadata.proto import metadata_store_pb2

# Property type for artifacts, executions and contexts.
Property = Union[int, float, str]
ExecPropertyTypes = Union[int, float, str, bool, message.Message, List[Any]]


class BaseChannel:
  """An abstract type for Channels that connects pipeline nodes.

  This class should be used by components that wish to handle more than one
  type of this BaseChannel's child classes.

  Attributes:
    type: The artifact type class that the Channel takes.
  """

  def __init__(self, type: Type[Artifact]):  # pylint: disable=redefined-builtin
    if not (inspect.isclass(type) and issubclass(type, Artifact)):  # pytype: disable=wrong-arg-types
      raise ValueError(
          'Argument "type" of BaseChannel constructor must be a subclass of '
          'tfx.Artifact (got %r).' % (type,))
    self.type = type

  @property
  def type_name(self):
    """Name of the artifact type class that Channel takes."""
    return self.type.TYPE_NAME


class Channel(json_utils.Jsonable, BaseChannel):
  """Tfx Channel.

  TFX Channel is an abstract concept that connects data producers and data
  consumers. It contains restriction of the artifact type that should be fed
  into or read from it.
  """

  # TODO(b/125348988): Add support for real Channel in addition to static ones.
  def __init__(
      self,
      type: Type[Artifact],  # pylint: disable=redefined-builtin
      additional_properties: Optional[Dict[str, Property]] = None,
      additional_custom_properties: Optional[Dict[str, Property]] = None,
      # TODO(b/161490287): deprecate static artifact.
      artifacts: Optional[Iterable[Artifact]] = None,
      producer_component_id: Optional[str] = None,
      output_key: Optional[str] = None):
    """Initialization of Channel.

    Args:
      type: Subclass of Artifact that represents the type of this Channel.
      additional_properties: (Optional) A mapping of properties which will be
        added to artifacts when this channel is used as an output of components.
        This is experimental and is subject to change in the future.
      additional_custom_properties: (Optional) A mapping of custom_properties
        which will be added to artifacts when this channel is used as an output
        of components. This is experimental and is subject to change in the
        future.
      artifacts: Deprecated and ignored, kept only for backward compatibility.
      producer_component_id: (Optional) Producer component id of the Channel.
        This argument is internal/experimental and is subject to change in the
        future.
      output_key: (Optional) The output key when producer component produces the
        artifacts in this Channel. This argument is internal/experimental and is
        subject to change in the future.
    """
    super().__init__(type=type)

    self.additional_properties = additional_properties or {}
    self.additional_custom_properties = additional_custom_properties or {}

    # The following fields will be populated during compilation time.
    self.producer_component_id = producer_component_id
    self.output_key = output_key

    if artifacts:
      logging.warning(
          'Artifacts param is ignored by Channel constructor, please remove!')
    self._artifacts = []
    self._matching_channel_name = None

  def __repr__(self):
    artifacts_str = '\n    '.join(repr(a) for a in self._artifacts)
    return textwrap.dedent("""\
        Channel(
            type_name: {}
            artifacts: [{}]
            additional_properties: {}
            additional_custom_properties: {}
        )""").format(self.type_name, artifacts_str, self.additional_properties,
                     self.additional_custom_properties)

  def _validate_type(self) -> None:
    for artifact in self._artifacts:
      if artifact.type_name != self.type_name:
        raise ValueError(
            "Artifacts provided do not match Channel's artifact type {}".format(
                self.type_name))

  # TODO(b/161490287): deprecate static artifact.
  @doc_controls.do_not_doc_inheritable
  def set_artifacts(self, artifacts: Iterable[Artifact]) -> 'Channel':
    """Sets artifacts for a static Channel. Will be deprecated."""
    if self._matching_channel_name:
      raise ValueError(
          'Only one of `artifacts` and `matching_channel_name` should be set.')
    self._artifacts = artifacts
    self._validate_type()
    return self

  @doc_controls.do_not_doc_inheritable
  def get(self) -> Iterable[Artifact]:
    """Returns all artifacts that can be get from this Channel.

    Returns:
      An artifact collection.
    """
    # TODO(b/125037186): We should support dynamic query against a Channel
    # instead of a static Artifact collection.
    return self._artifacts

  # TODO(b/185957572): deprecate matching_channel_name.
  @property
  @deprecation_utils.deprecated(
      None, '`matching_channel_name` will be deprecated soon.')
  @doc_controls.do_not_doc_inheritable
  def matching_channel_name(self) -> str:
    return self._matching_channel_name

  # TODO(b/185957572): deprecate matching_channel_name.
  @matching_channel_name.setter
  def matching_channel_name(self, matching_channel_name: str):
    # This targets to the key of an input Channel dict in a Component.
    # The artifacts count of this channel will be decided at runtime in Driver,
    # based on the artifacts count of the target channel.
    if self._artifacts:
      raise ValueError(
          'Only one of `artifacts` and `matching_channel_name` should be set.')
    self._matching_channel_name = matching_channel_name

  @doc_controls.do_not_doc_inheritable
  def to_json_dict(self) -> Dict[str, Any]:
    return {
        'type':
            json.loads(
                json_format.MessageToJson(
                    message=self.type._get_artifact_type(),  # pylint: disable=protected-access
                    preserving_proto_field_name=True)),
        'artifacts':
            list(a.to_json_dict() for a in self._artifacts),
        'additional_properties':
            self.additional_properties,
        'additional_custom_properties':
            self.additional_custom_properties,
        'producer_component_id':
            (self.producer_component_id if self.producer_component_id else None
            ),
        'output_key': (self.output_key if self.output_key else None),
    }

  @classmethod
  @doc_controls.do_not_doc_inheritable
  def from_json_dict(cls, dict_data: Dict[str, Any]) -> Any:
    artifact_type = metadata_store_pb2.ArtifactType()
    json_format.Parse(json.dumps(dict_data['type']), artifact_type)
    type_cls = artifact_utils.get_artifact_type_class(artifact_type)
    artifacts = list(Artifact.from_json_dict(a) for a in dict_data['artifacts'])
    additional_properties = dict_data['additional_properties']
    additional_custom_properties = dict_data['additional_custom_properties']
    producer_component_id = dict_data.get('producer_component_id', None)
    output_key = dict_data.get('output_key', None)
    return Channel(
        type=type_cls,
        additional_properties=additional_properties,
        additional_custom_properties=additional_custom_properties,
        producer_component_id=producer_component_id,
        output_key=output_key).set_artifacts(artifacts)

  def future(self) -> placeholder.ChannelWrappedPlaceholder:
    return placeholder.ChannelWrappedPlaceholder(self)


@doc_controls.do_not_generate_docs
class UnionChannel(BaseChannel):
  """Union of multiple Channels with the same type.

  Prefer to use union() to create UnionChannel.

  Currently future() method is only support for Channel class, so conditional
  does not yet work with channel union.
  """

  def __init__(self, type: Type[Artifact], input_channels: List[BaseChannel]):  # pylint: disable=redefined-builtin
    super().__init__(type=type)

    if not input_channels:
      raise ValueError('At least one input channel expected.')

    self.channels = []
    for c in input_channels:
      if isinstance(c, UnionChannel):
        self.channels.extend(cast(UnionChannel, c).channels)
      elif isinstance(c, Channel):
        self.channels.append(c)
      else:
        raise ValueError('Unexpected channel type: %s.' % c.type_name)

    self._validate_type()

  def _validate_type(self):
    for channel in self.channels:
      if not isinstance(channel, Channel) or channel.type != self.type:
        raise TypeError(
            'Unioned channels must have the same type. Expected %s (got %s).' %
            (self.type, channel.type))


def union(input_channels: Iterable[BaseChannel]) -> UnionChannel:
  """Convenient method to combine multiple input channels into union channel."""
  input_channels = list(input_channels)
  assert input_channels, 'Not expecting empty input channels list.'
  return UnionChannel(input_channels[0].type, input_channels)


@doc_controls.do_not_generate_docs
class LoopVarChannel(BaseChannel):
  """LoopVarChannel is a channel that is marked as a ForEach loop variable.

  There is no special functionality for this channel itself; it just marks the
  channel as a loop variable, and holds the context ID for the corresponding
  ForEachContext.
  """

  def __init__(self, wrapped: BaseChannel, context_id: str):
    """LoopVarChannel constructor.

    Arguments:
      wrapped: A wrapped BaseChannel.
      context_id: An ID for the corresponding ForEachContext.
    """
    super().__init__(wrapped.type)
    self._wrapped = wrapped
    self._context_id = context_id

  @property
  def wrapped(self) -> BaseChannel:
    return self._wrapped

  @property
  def context_id(self) -> str:
    return self._context_id
