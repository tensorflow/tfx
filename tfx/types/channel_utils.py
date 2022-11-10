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

from typing import cast, Dict, Iterable, List, Type

from tfx.types import artifact
from tfx.types import channel


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


def external_project_artifact_query(
    artifact_type: Type[artifact.Artifact],
    *,
    project_owner: str,
    project_name: str,
    producer_component_id: str,
    output_key: str,
    pipeline_name: str = '',
    pipeline_run_id: str = '',
    mlmd_service_target: str = ''
) -> channel.ExternalProjectChannel:
  """Helper function to construct a query to get artifacts from an MLMD db.

  Args:
    artifact_type: Subclass of Artifact for this channel.
    project_owner: Onwer of the MLMD db.
    project_name: Name of the MLMD db.
    producer_component_id: Id of the component produces the artifacts.
    output_key: The output key when producer component produces the artifacts in
      this Channel.
    pipeline_name: (Optional) Name of the pipeline the artifacts belong to. If
      not provided, default to project name.
    pipeline_run_id: (Optional) Pipeline run id the artifacts belong to.
    mlmd_service_target: (Optional) Service target of the MLMD db.

  Returns:
    channel.ExternalProjectChannel instance.
  """
  return channel.ExternalProjectChannel(
      artifact_type=artifact_type,
      project_owner=project_owner,
      project_name=project_name,
      pipeline_name=pipeline_name,
      producer_component_id=producer_component_id,
      output_key=output_key,
      pipeline_run_id=pipeline_run_id,
      mlmd_service_target=mlmd_service_target,
  )
