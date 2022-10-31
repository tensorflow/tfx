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

from tfx.dsl.context_managers import dsl_context
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
Property = Union[int, float, str, message.Message]
ExecPropertyTypes = Union[int, float, str, bool, message.Message, List[Any],
                          Dict[Any, Any]]
_EXEC_PROPERTY_CLASSES = (int, float, str, bool, message.Message, list, dict)


def _is_artifact_type(value: Any):
  return inspect.isclass(value) and issubclass(value, Artifact)


def _is_property_dict(value: Any):
  return (
      isinstance(value, dict) and
      all(isinstance(k, str) for k in value.keys()) and
      all(isinstance(v, _EXEC_PROPERTY_CLASSES) for v in value.values()))


class BaseChannel:
  """An abstraction for component (BaseNode) artifact inputs.

  `BaseChannel` is often interchangeably used with the term 'channel' (not
  capital `Channel` which points to the legacy class name).

  Component takes artifact inputs distinguished by each "input key". For
  example:

      trainer = Trainer(
          examples=example_gen.outputs['examples'])
          ^^^^^^^^
          input key
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                   channel

  Here "examples" is the input key of the `Examples` artifact type.
  `example_gen.outputs['examples']` is a channel. Typically a single channel
  refers to a *list of `Artifact` of a homogeneous type*. Since channel is a
  declarative abstraction it is not strictly bound to the actual artifact, but
  is more of an *input selector*.

  The most commonly used channel type is an `OutputChannel` (in the form of
  `component.outputs["key"]`, which selects the artifact produced by the
  component in the same pipeline run (in synchronous execution mode; more
  information on OutputChannel docstring), and is typically a single artifact.

  Attributes:
    type: The artifact type class that the Channel takes.
  """

  def __init__(self, type: Type[Artifact]):  # pylint: disable=redefined-builtin
    if not _is_artifact_type(type):
      raise ValueError(
          'Argument "type" of BaseChannel constructor must be a subclass of '
          f'tfx.Artifact (got {type}).')
    self._artifact_type = type

  @property
  def type(self):  # pylint: disable=redefined-builtin
    return self._artifact_type

  @type.setter
  def type(self, value: Type[Artifact]):  # pylint: disable=redefined-builtin
    self._set_type(value)

  @doc_controls.do_not_generate_docs
  def _set_type(self, value: Type[Artifact]):
    raise NotImplementedError('Cannot change artifact type.')

  @property
  def type_name(self):
    """Name of the artifact type class that Channel takes."""
    return self.type.TYPE_NAME

  def future(self) -> placeholder.ChannelWrappedPlaceholder:
    return placeholder.ChannelWrappedPlaceholder(self)


class Channel(json_utils.Jsonable, BaseChannel):
  """Legacy channel interface.

  `Channel` used to represent the `BaseChannel` concept in the early TFX code,
  but due to having too much features in the same class, we refactored it to
  multiple classes:

  - BaseChannel for the general input abstraction
  - OutputChannel for `component.outputs['key']`.
  - MLMDQueryChannel for simple filter-based input resolution.

  Please do not use this class directly, but instead use the alternatives. This
  class won't be removed in TFX 1.x due to backward compatibility guarantee
  though.
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

    if additional_properties is not None:
      self._validate_additional_properties(additional_properties)
    self.additional_properties = additional_properties or {}

    if additional_custom_properties is not None:
      self._validate_additional_custom_properties(additional_custom_properties)
    self.additional_custom_properties = additional_custom_properties or {}

    if producer_component_id is not None:
      self._validate_producer_component_id(producer_component_id)
    # Use a protected attribute & getter/setter property as OutputChannel is
    # overriding it.
    self._producer_component_id = producer_component_id

    if output_key is not None:
      self._validate_output_key(output_key)
    self.output_key = output_key

    if artifacts:
      logging.warning(
          'Artifacts param is ignored by Channel constructor, please remove!')
    self._artifacts = []
    self._matching_channel_name = None

  def _set_type(self, value: Type[Artifact]) -> None:
    """Mutate artifact type."""
    if not _is_artifact_type(value):
      raise TypeError(
          f'artifact_type should be a subclass of tfx.Artifact (got {value}).')
    self._artifact_type = value

  @property
  @doc_controls.do_not_generate_docs
  def producer_component_id(self) -> Optional[str]:
    return self._producer_component_id

  @producer_component_id.setter
  @doc_controls.do_not_generate_docs
  def producer_component_id(self, value: str) -> None:
    self._validate_producer_component_id(value)
    self._producer_component_id = value

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

  def _validate_additional_properties(self, value: Any) -> None:
    if not _is_property_dict(value):
      raise ValueError(
          f'Invalid additional_properties {value}. '
          f'Must be a {Dict[str, Property]} type.')

  def _validate_additional_custom_properties(self, value: Any) -> None:
    if not _is_property_dict(value):
      raise ValueError(
          f'Invalid additional_custom_properties {value}. '
          f'Must be a {Dict[str, Property]} type.')

  def _validate_producer_component_id(self, value: Any) -> None:
    if not isinstance(value, str):
      raise ValueError(
          f'Invalid producer_component_id {value}. Must be a str type.')

  def _validate_output_key(self, value: Any) -> None:
    if not isinstance(value, str):
      raise ValueError(f'Invalid output_key {value}. Must be a str type.')

  def _validate_static_artifacts(self, artifacts: Iterable[Artifact]) -> None:
    for artifact in artifacts:
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
    self._validate_static_artifacts(artifacts)
    self._artifacts = artifacts
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

  @doc_controls.do_not_generate_docs
  def as_output_channel(
      self, producer_component: Any, output_key: str) -> 'OutputChannel':
    """Internal method to derive OutputChannel from the Channel instance.

    Return value (OutputChannel instance) is based on the shallow copy of self,
    so that any attribute change in one is reflected on the others.

    Args:
      producer_component: A BaseNode instance that is producing this channel.
      output_key: Corresponding node.outputs key for this channel.

    Returns:
      An OutputChannel instance that shares attributes with self.
    """
    # Disable pylint false alarm for safe access of protected attributes.
    # pylint: disable=protected-access
    result = OutputChannel(self.type, producer_component, output_key)
    result.additional_properties = self.additional_properties
    result.additional_custom_properties = self.additional_custom_properties
    result.set_artifacts(self._artifacts)
    return result


class OutputChannel(Channel):
  """Channel that is used for `node.outputs['key']`.

  For OutputChannel being used as another component's inputs, it has the
  following implications (only in synchronous execution mode):

  - It refers to the output artifact produced from *the same pipeline run*.
  - Imposes data dependency between producer component and the consumer
    component.

  It's worth mentioning that in most cases (except for a small portion of
  special ExampleGens that produces multiple Examples), output channel of
  standard components in synchronous mode resolved to 1 artifact.

  For asynchronous execution mode, each component is no longer data dependent,
  and there is no pipeline run context to group artifacts. Therefore an
  OutputChannel refers to all artifacts produced from the component in the
  pipeline. This no longer resolves to 1 artifact, so in an asynchronous
  execution mode you need to further filter output channel with resolver
  functions (e.g. `tfx.dsl.inputs.latest_created`).

  OutputChannel has additional functionalities to manipulate the component
  output specification:

  - `additional_properties` or `additional_custom_properties` to statically
    set output artifacts' properties or custom_properties.
  - Garbage collection policy per component outputs.
  """

  def __init__(
      self,
      artifact_type: Type[Artifact],
      producer_component: Any,
      output_key: str,
      additional_properties: Optional[Dict[str, Property]] = None,
      additional_custom_properties: Optional[Dict[str, Property]] = None,
  ):
    super().__init__(
        type=artifact_type,
        output_key=output_key,
        additional_properties=additional_properties,
        additional_custom_properties=additional_custom_properties,
    )
    self._producer_component = producer_component
    self._garbage_collection_policy = None
    self._predefined_artifact_uris = None

  def __repr__(self) -> str:
    return (
        f'{self.__class__.__name__}('
        f'artifact_type={self.type_name}, '
        f'producer_component_id={self.producer_component_id}, '
        f'output_key={self.output_key}, '
        f'additional_properties={self.additional_properties}, '
        f'additional_custom_properties={self.additional_custom_properties})')

  @doc_controls.do_not_generate_docs
  def set_producer_component(self, value: Any):
    self._producer_component = value

  @property
  @doc_controls.do_not_generate_docs
  def producer_component_id(self) -> str:
    return self._producer_component.id

  @doc_controls.do_not_generate_docs
  def as_output_channel(
      self, producer_component: Any, output_key: str) -> 'OutputChannel':
    if self._producer_component != producer_component:
      raise ValueError(
          f'producer_component mismatch: {self._producer_component} != '
          f'{producer_component}.')
    if self.output_key != output_key:
      raise ValueError(
          f'output_key mismatch: {self.output_key} != {output_key}.')
    return self

  @doc_controls.do_not_generate_docs
  def set_external(self, predefined_artifact_uris: List[str]) -> None:
    self._predefined_artifact_uris = predefined_artifact_uris


@doc_controls.do_not_generate_docs
class UnionChannel(BaseChannel):
  """Union of multiple channels with the same type.

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

  def __init__(self, wrapped: BaseChannel,
               for_each_context: dsl_context.DslContext):
    """LoopVarChannel constructor.

    Arguments:
      wrapped: A wrapped BaseChannel.
      for_each_context: An ID for the corresponding ForEachContext.
    """
    super().__init__(wrapped.type)
    self._wrapped = wrapped
    self._for_each_context = for_each_context

  @property
  def wrapped(self) -> BaseChannel:
    return self._wrapped

  @property
  def for_each_context(self) -> dsl_context.DslContext:
    return self._for_each_context


@doc_controls.do_not_generate_docs
class PipelineOutputChannel(OutputChannel):
  """PipelineOutputChannel wraps a channnel to be used in the outer pipeline.

  It wraps a channel produced from within an inner composable pipeline, to be
  used in the outer composable pipeline.
  """

  def __init__(self,
               wrapped: BaseChannel,
               pipeline: Optional[Any] = None,
               output_key: str = ''):
    self._wrapped = wrapped
    self._pipeline = pipeline
    if isinstance(wrapped, Channel):
      additional_properties = wrapped.additional_properties
      additional_custom_properties = wrapped.additional_custom_properties
    else:
      additional_properties = {}
      additional_custom_properties = {}
    super().__init__(
        artifact_type=wrapped.type,
        producer_component=pipeline,
        output_key=output_key or '',
        additional_properties=additional_properties,
        additional_custom_properties=additional_custom_properties)

  @property
  def wrapped(self) -> BaseChannel:
    return self._wrapped

  @property
  def pipeline(self) -> Any:
    return self._pipeline

  @pipeline.setter
  def pipeline(self, pipeline: Any):
    self._pipeline = pipeline
    self._producer_component = pipeline

  def __eq__(self, other):
    if isinstance(other, PipelineOutputChannel):
      return (self.wrapped == other.wrapped and
              self.pipeline == other.pipeline and
              self.output_key == other.output_key)
    return False

  def __hash__(self):
    return id(self.wrapped) + id(self.pipeline) + id(self.output_key)


@doc_controls.do_not_generate_docs
class PipelineInputChannel(BaseChannel):
  """PipelineInputChannel wraps a channnel to be used in an inner pipeline.

  It wraps a channel produced from a outer/parent composable pipeline, to be
  used in the inner composable pipeline.
  """

  def __init__(self, wrapped: BaseChannel, output_key: str):
    super().__init__(type=wrapped.type)
    self._wrapped = wrapped
    self._output_key = output_key
    self._pipeline = None

  @property
  def wrapped(self) -> BaseChannel:
    return self._wrapped

  @property
  def output_key(self) -> str:
    return self._output_key

  @property
  def pipeline(self) -> Any:
    return self._pipeline

  @pipeline.setter
  def pipeline(self, pipeline: Any):
    self._pipeline = pipeline
