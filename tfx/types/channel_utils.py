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
"""TFX Channel utilities.

DO NOT USE THIS MODULE DIRECTLY. This module is a private module, and all public
symbols are already available from one of followings:

- `tfx.v1.types.BaseChannel`
- `tfx.v1.testing.Channel`
- `tfx.v1.dsl.union`
- `tfx.v1.dsl.experimental.artifact_query`
- `tfx.v1.dsl.experimental.external_pipeline_artifact_query`

Consider other symbols as private.
"""

import typing
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Type, cast

from tfx.dsl.placeholder import placeholder as ph
from tfx.proto.orchestration import placeholder_pb2
from tfx.types import artifact
from tfx.types import channel

from ml_metadata.proto import metadata_store_pb2


class ChannelForTesting(channel.BaseChannel):
  """Dummy channel for testing."""

  def __init__(
      self,
      artifact_type: Type[artifact.Artifact],
      artifact_ids: Sequence[int] = (),
  ):
    super().__init__(artifact_type)
    self.artifact_ids = artifact_ids

  def __hash__(self):
    return hash(self.type)

  def __eq__(self, other):
    return isinstance(other, ChannelForTesting) and self.type == other.type

  def get_data_dependent_node_ids(self) -> Set[str]:
    return set()

  def future(self) -> channel.ChannelWrappedPlaceholder:
    return channel.ChannelWrappedPlaceholder(self)


def as_channel(artifacts: Iterable[artifact.Artifact]) -> channel.Channel:
  """Converts artifact collection of the same artifact type into a Channel.

  Args:
    artifacts: An iterable of Artifact.

  Returns:
    A static Channel containing the source artifact collection.

  Raises:
    ValueError when source is not a non-empty iterable of Artifact.
  """
  try:
    first_element = next(iter(artifacts))
    if isinstance(first_element, artifact.Artifact):
      return channel.Channel(type=first_element.type).set_artifacts(artifacts)
    else:
      raise ValueError('Invalid artifact iterable: {}'.format(artifacts))
  except StopIteration as e:
    raise ValueError(
        'Cannot convert empty artifact iterable into Channel') from e


def unwrap_channel_dict(
    channel_dict: Dict[str,
                       channel.Channel]) -> Dict[str, List[artifact.Artifact]]:
  """Unwrap dict of channels to dict of lists of Artifact.

  Args:
    channel_dict: a dict of Text -> Channel

  Returns:
    a dict of Text -> List[Artifact]
  """
  return dict((k, list(v.get())) for k, v in channel_dict.items())


def get_individual_channels(
    input_channel: channel.BaseChannel) -> List[channel.Channel]:
  """Converts BaseChannel into a list of Channels."""
  if isinstance(input_channel, channel.Channel):
    return [input_channel]
  elif isinstance(input_channel, channel.UnionChannel):
    return [
        chan for chan in cast(channel.UnionChannel, input_channel).channels
        if isinstance(chan, channel.Channel)]
  else:
    raise NotImplementedError(
        f'Unsupported Channel type: {type(input_channel)}')


def union(channels: Iterable[channel.BaseChannel]) -> channel.UnionChannel:
  """Returns the union of channels.

  All channels should have the same artifact type, otherwise an error would be
  raised. Returned channel deduplicates the inputs so each artifact is
  guaranteed to be present at most once. `union()` does NOT guarantee any
  ordering of artifacts for the consumer component.

  Args:
    channels: An iterable of BaseChannels.

  Returns:
    A BaseChannel that represents the union of channels.
  """
  return channel.UnionChannel(channels)


def artifact_query(
    artifact_type: Type[artifact.Artifact],
    *,
    producer_component_id: Optional[str] = None,
    output_key: Optional[str] = None,
) -> channel.Channel:
  """Creates a MLMD query based channel."""
  if output_key is not None and producer_component_id is None:
    raise ValueError('producer_component_id must be set to use output_key.')
  return channel.Channel(
      artifact_type,
      producer_component_id=producer_component_id,
      output_key=output_key)


def external_pipeline_artifact_query(
    artifact_type: Type[artifact.Artifact],
    *,
    owner: str,
    pipeline_name: str,
    producer_component_id: str,
    output_key: str,
    pipeline_run_id: str = '',
    pipeline_run_tags: Sequence[str] = (),
) -> channel.ExternalPipelineChannel:
  """Helper function to construct a query to get artifacts from an external pipeline.

  Args:
    artifact_type: Subclass of Artifact for this channel.
    owner: Owner of the pipeline.
    pipeline_name: Name of the pipeline the artifacts belong to.
    producer_component_id: Id of the component produces the artifacts.
    output_key: The output key when producer component produces the artifacts in
      this Channel.
    pipeline_run_id: (Optional) Pipeline run id the artifacts belong to.
    pipeline_run_tags: (Optional) A list of tags the artifacts belong to. It is
      an AND relationship between tags. For example, if tags=['tag1', 'tag2'],
      then only artifacts belonging to the run with both 'tag1' and 'tag2' will
      be returned. Only one of pipeline_run_id and pipeline_run_tags can be set.

  Returns:
    channel.ExternalPipelineChannel instance.

  Raises:
    ValueError, if owner or pipeline_name is missing, or both pipeline_run_id
      and pipeline_run_tags are set.
  """
  if not owner or not pipeline_name:
    raise ValueError('owner or pipeline_name is missing.')

  if pipeline_run_id and pipeline_run_tags:
    raise ValueError(
        'pipeline_run_id and pipeline_run_tags cannot be both set.'
    )

  run_context_predicates = []
  for tag in pipeline_run_tags:
    # TODO(b/264728226): Find a better way to construct the tag name that used
    # in MLMD. Tag names that used in MLMD are constructed in tflex_mlmd_api.py,
    # but it is not visible in this file.
    mlmd_store_tag = '__tag_' + tag + '__'
    run_context_predicates.append((
        mlmd_store_tag,
        metadata_store_pb2.Value(bool_value=True),
    ))

  return channel.ExternalPipelineChannel(
      artifact_type=artifact_type,
      owner=owner,
      pipeline_name=pipeline_name,
      producer_component_id=producer_component_id,
      output_key=output_key,
      pipeline_run_id=pipeline_run_id,
      run_context_predicates=run_context_predicates,
  )


def get_dependent_channels(
    placeholder: ph.Placeholder,
) -> Iterator[channel.Channel]:
  """Yields all Channels used in/ the given placeholder."""
  for p in placeholder.traverse():
    if isinstance(p, ph.ChannelWrappedPlaceholder):
      yield typing.cast(channel.Channel, p.channel)


def unwrap_simple_channel_placeholder(
    placeholder: ph.Placeholder,
) -> channel.Channel:
  """Unwraps a `x.future()[0].value` placeholder and returns its `x`.

  Args:
    placeholder: A placeholder expression.

  Returns:
    The (only) channel involved in the expression.

  Raises:
    ValueError: If the input placeholder is anything more complex than
      `some_channel.future()[0].value`, and in particular if it involves
      multiple channels, arithmetic operations or input/output artifacts.
  """
  # Validate that it's the right shape.
  outer_ph = placeholder.encode()
  index_op = outer_ph.operator.artifact_value_op.expression.operator.index_op
  cwp = index_op.expression.placeholder
  if (
      # This catches the case where we've been navigating down non-existent
      # proto paths above and been getting default messages all along. If this
      # sub-message is present, then the whole chain was correct.
      not index_op.expression.HasField('placeholder')
      # ChannelWrappedPlaceholder uses INPUT_ARTIFACT for some reason, and has
      # no key when encoded with encode().
      or cwp.type != placeholder_pb2.Placeholder.Type.INPUT_ARTIFACT
      or cwp.key
      # For the `[0]` part of the desired shape.
      or index_op.index != 0
  ):
    raise ValueError(
        'Expected placeholder of shape somechannel.future()[0].value, but got'
        f' {placeholder}.'
    )

  # Now that we know there's only one channel inside, we can just extract it:
  return next(
      p.channel
      for p in placeholder.traverse()
      if isinstance(p, ph.ChannelWrappedPlaceholder)
  )


def encode_placeholder_with_channels(
    placeholder: ph.Placeholder,
    channel_to_key_fn: Callable[[channel.BaseChannel], str],
) -> placeholder_pb2.PlaceholderExpression:
  """Encodes the placeholder with the given channel keys.

  When a ChannelWrappedPlaceholder is initially constructed, it does not have a
  key. This means that Predicates, which comprise of ChannelWrappedPlaceholders,
  do not contain sufficient information to be evaluated at run time. Thus, after
  the Pipeline is fully defined, the compiler will use this method to fill in
  the keys when encoding the placeholder/predicate.

  Within the context of each node, each Channel corresponds to a key in the
  input_dict (see `ResolverStrategy.resolve_artifacts`).
  The Placeholder/Predicate has an internal tree data structure to keep track of
  all the placeholders and operations. As we traverse this tree to create this
  proto, `channel_to_key_fn` is called each time a ChannelWrappedPlaceholder is
  encountered, and its output is used as the placeholder key in the Placeholder
  proto produced during encoding.

  Note that the ChannelWrappedPlaceholder itself is unchanged after this
  function returns.

  Args:
    placeholder: The placeholder to be encoded.
    channel_to_key_fn: The function used to determine the placeholder key for
      each ChannelWrappedPlaceholder. If None, no attempt to fill in the
      placeholder keys will be made.

  Returns:
    A PlaceholderExpression proto that represent the given placeholder. Note
    that the given Placeholder remains unchanged.
  """
  for p in placeholder.traverse():
    if isinstance(p, ph.ChannelWrappedPlaceholder):
      p.set_key(channel_to_key_fn(p.channel))
  try:
    return placeholder.encode()
  finally:
    for p in placeholder.traverse():
      if isinstance(p, ph.ChannelWrappedPlaceholder):
        p.set_key(None)
