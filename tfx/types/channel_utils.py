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
"""TFX Channel utilities."""

from typing import cast, Dict, Iterable, List

from tfx.dsl.input_resolution import resolver_function
from tfx.types import artifact
from tfx.types import channel
from tfx.types import resolved_channel


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
  except StopIteration:
    raise ValueError('Cannot convert empty artifact iterable into Channel')


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
    return list(cast(channel.UnionChannel, input_channel).channels)
  elif isinstance(input_channel, channel.LoopVarChannel):
    return get_individual_channels(
        cast(channel.LoopVarChannel, input_channel).wrapped)
  elif isinstance(input_channel, resolved_channel.ResolvedChannel):
    input_channel = cast(resolved_channel.ResolvedChannel, input_channel)
    return resolver_function.get_dependent_channels(input_channel.output_node)
  else:
    raise RuntimeError(f'Unexpected Channel type: {type(input_channel)}')


def get_dependent_node_ids(channel_: channel.BaseChannel) -> Iterable[str]:
  """Returns an iterable of data dependent node ids for the channel."""
  # pytype: disable=attribute-error
  if isinstance(channel_, channel.OutputChannel):
    yield channel_.producer_component_id
  elif isinstance(channel_, channel.PipelineInputChannel):
    yield channel_.pipeline.id
  elif isinstance(channel_, channel.Channel):
    # Raw Channel is not considered as a data dependent usage. If dependency is
    # needed, a task dependency can be set from DSL.
    # TODO(b/219645784): Reevaluate the dependency semantics of Channel.
    return
  elif isinstance(channel_, channel.UnionChannel):
    for each_channel in channel_.channels:
      yield from get_dependent_node_ids(each_channel)
  elif isinstance(channel_, channel.LoopVarChannel):
    yield from get_dependent_node_ids(channel_.wrapped)
  elif isinstance(channel_, resolved_channel.ResolvedChannel):
    for each_channel in resolver_function.get_dependent_channels(
        channel_.output_node):
      yield from get_dependent_node_ids(each_channel)
  else:
    raise TypeError(f'Invalid channel type {type(channel_)}')
