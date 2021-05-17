# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Experimental Resolver for getting the artifacts based on Span."""

from typing import Dict, List, Optional, Text

from tfx import types
from tfx.components.example_gen import utils
from tfx.dsl.components.common import resolver
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.proto import range_config_pb2
from tfx.types import artifact_utils
from tfx.utils import doc_controls

import ml_metadata as mlmd


def _get_span_custom_property(artifact: types.Artifact) -> int:
  # For backward compatibility, span may be stored as a string.
  str_span = artifact.get_string_custom_property(utils.SPAN_PROPERTY_NAME)
  if str_span:
    return int(str_span)
  return artifact.get_int_custom_property(utils.SPAN_PROPERTY_NAME)


class SpanRangeStrategy(resolver.ResolverStrategy):
  """SpanRangeStrategy resolves artifacts based on "span" property.

  Note that this ResolverStrategy is experimental and is subject to change in
  terms of both interface and implementation.

  Don't construct SpanRangeStrategy directly, example usage:
  ```
    examples_resolver = Resolver(
        instance_name='span_resolver',
        strategy_class=SpanRangeStrategy,
        config={'range_config': range_config},
        examples=Channel(type=Examples, producer_component_id=example_gen.id))
    examples_resolver.outputs['examples']
  ```
  """

  def __init__(self, range_config: range_config_pb2.RangeConfig):
    self._range_config = range_config

  def _resolve(self, input_dict: Dict[Text, List[types.Artifact]]):
    result = {}

    for k, artifact_list in input_dict.items():
      in_range_artifacts = []

      if self._range_config.HasField('static_range'):
        start_span_number = self._range_config.static_range.start_span_number
        end_span_number = self._range_config.static_range.end_span_number
        # Get the artifacts within range.
        for artifact in artifact_list:
          if not artifact.has_custom_property(utils.SPAN_PROPERTY_NAME):
            raise RuntimeError(f'Span does not exist for {artifact}')
          span = _get_span_custom_property(artifact)
          if span >= start_span_number and span <= end_span_number:
            in_range_artifacts.append(artifact)

      elif self._range_config.HasField('rolling_range'):
        start_span_number = self._range_config.rolling_range.start_span_number
        num_spans = self._range_config.rolling_range.num_spans
        if num_spans <= 0:
          raise ValueError('num_spans should be positive number.')
        most_recent_span = -1
        # Get most recent span number.
        for artifact in artifact_list:
          if not artifact.has_custom_property(utils.SPAN_PROPERTY_NAME):
            raise RuntimeError(f'Span does not exist for {artifact}')
          span = _get_span_custom_property(artifact)
          if span > most_recent_span:
            most_recent_span = span

        start_span_number = max(start_span_number,
                                most_recent_span - num_spans + 1)
        end_span_number = most_recent_span
        # Get the artifacts within range.
        for artifact in artifact_list:
          span = _get_span_custom_property(artifact)
          if span >= start_span_number and span <= end_span_number:
            in_range_artifacts.append(artifact)

      else:
        raise ValueError('RangeConfig type is not supported.')

      result[k] = sorted(
          in_range_artifacts,
          key=_get_span_custom_property,
          reverse=True)

    return result

  @doc_controls.do_not_generate_docs
  def resolve(
      self,
      pipeline_info: data_types.PipelineInfo,
      metadata_handler: metadata.Metadata,
      source_channels: Dict[Text, types.Channel],
  ) -> resolver.ResolveResult:
    pipeline_context = metadata_handler.get_pipeline_context(pipeline_info)
    if pipeline_context is None:
      raise RuntimeError(f'Pipeline context absent for {pipeline_context}')

    candidate_dict = {}
    for k, c in source_channels.items():
      candidate_artifacts = metadata_handler.get_qualified_artifacts(
          contexts=[pipeline_context],
          type_name=c.type_name,
          producer_component_id=c.producer_component_id,
          output_key=c.output_key)
      candidate_dict[k] = [
          artifact_utils.deserialize_artifact(a.type, a.artifact)
          for a in candidate_artifacts
      ]

    resolved_dict = self._resolve(candidate_dict)
    resolve_state_dict = {
        k: bool(artifact_list) for k, artifact_list in resolved_dict.items()
    }

    return resolver.ResolveResult(
        per_key_resolve_result=resolved_dict,
        per_key_resolve_state=resolve_state_dict)

  @doc_controls.do_not_generate_docs
  def resolve_artifacts(
      self, store: mlmd.MetadataStore,
      input_dict: Dict[Text, List[types.Artifact]]
  ) -> Optional[Dict[Text, List[types.Artifact]]]:
    """Resolves artifacts from channels by querying MLMD.

    Args:
      store: An MLMD MetadataStore object.
      input_dict: The input_dict to resolve from.

    Returns:
      If `min_count` for every input is met, returns a
      Dict[Text, List[Artifact]]. Otherwise, return None.

    Raises:
      RuntimeError: if input_dict contains artifact without span property.
    """
    resolved_dict = self._resolve(input_dict)
    all_min_count_met = all(
        bool(artifact_list) for artifact_list in resolved_dict.values())
    return resolved_dict if all_min_count_met else None
