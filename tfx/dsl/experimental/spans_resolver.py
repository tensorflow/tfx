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
"""Experimental Resolver for getting the latest artifact."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Dict, Optional, Text

from tfx import types
from tfx.components.example_gen import utils
from tfx.dsl.resolvers import base_resolver
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.proto import range_config_pb2
from tfx.types import artifact_utils
from tfx.types.standard_artifacts import Examples


class SpansResolver(base_resolver.BaseResolver):
  """Resolver that returns a range of spans in a given Examples channel.

  Note that this Resolver is experimental and is subject to change in terms of
  both interface and implementation.
  """

  def __init__(self, 
               range_config: Optional[range_config_pb2.RangeConfig] = None,
               merge_same_artifact_type: bool = False):
    self._range_config = range_config
    self._merge_same_artifact_type = merge_same_artifact_type

  def resolve(
      self,
      pipeline_info: data_types.PipelineInfo,
      metadata_handler: metadata.Metadata,
      source_channels: Dict[Text, types.Channel],
  ) -> base_resolver.ResolveResult:
    artifacts_dict = {}
    resolve_state_dict = {}

    # Verifies that duplicate spans are not added to output over multiple
    # channels; specifically for when `merge_same_artifact_type` is set to True.
    total_processed_spans = set()

    pipeline_context = metadata_handler.get_pipeline_context(pipeline_info)
    if pipeline_context is None:
      raise RuntimeError('Pipeline context absent for %s' % pipeline_context)
    for k, c in source_channels.items():
      if c.type_name != Examples.TYPE_NAME:
        raise ValueError('Channel does not contain Example artifacts: %s' % k)

      # Make sure that artifacts with the same span are not both resolved.
      processed_spans = set()
      if self._merge_same_artifact_type:
        # If flag is true, only one output channel, with a name of 'Examples'.
        k = Examples.TYPE_NAME
        processed_spans = total_processed_spans

      candidate_artifacts = metadata_handler.get_qualified_artifacts(
          contexts=[pipeline_context],
          type_name=c.type_name,
          producer_component_id=c.producer_component_id,
          output_key=c.output_key)

      # TODO(jjma): This is a quick fix to incorporate version into this
      # ordering. Sorting by last update times makes sure that newer versions
      # are ahead of older versions (since newer versions logically are 
      # processed more recently than old versions). Then sorting by span makes
      # sure that artifacts are ordered first by latest span, then by latest 
      # version.
      previous_artifacts = sorted(
          candidate_artifacts,
          key=lambda a: a.artifact.last_update_time_since_epoch, reverse=True)
      previous_artifacts = sorted(
          previous_artifacts, 
          key=lambda a: int(
              a.artifact.custom_properties[utils.SPAN_PROPERTY_NAME].string_value),
          reverse=True)

      # Create new list of artifacts or get list already at key, if merging
      # multiple channels into one.
      artifacts_dict[k] = artifacts_dict.get(k, [])
      resolve_state_dict[k] = False

      if self._range_config:
        if self._range_config.HasField('static_range'):
          # TODO(jjma): Optimize this by sending a more specific MLMD query.
          lower_bound = self._range_config.static_range.start_span_number
          upper_bound = self._range_config.static_range.end_span_number

          for a in previous_artifacts:
            span = int(
                a.artifact.custom_properties[utils.SPAN_PROPERTY_NAME].string_value)
            if span not in processed_spans and (lower_bound <= span and 
                                                span <= upper_bound):
              artifacts_dict[k].append(
                  artifact_utils.deserialize_artifact(a.type, a.artifact))
              processed_spans.add(span)
        
          resolve_state_dict[k] = (
              len(artifacts_dict[k]) == (upper_bound - lower_bound + 1))

        elif self._range_config.HasField('rolling_range'):
          start_span = self._range_config.rolling_range.start_span_number
          num_spans = self._range_config.rolling_range.num_spans
          num_skip = self._range_config.rolling_range.skip_num_recent_spans

          for i, a in enumerate(previous_artifacts[num_skip:]):
            span = int(
                a.artifact.custom_properties[utils.SPAN_PROPERTY_NAME].string_value)
            if span not in processed_spans and (span >= start_span and
                                                i < num_spans):
              artifacts_dict[k].append(
                  artifact_utils.deserialize_artifact(a.type, a.artifact))
              processed_spans.add(span)
            else:
              break
          
          resolve_state_dict[k] = len(artifacts_dict[k]) == num_spans

      elif len(previous_artifacts) > 0:
        # Default behavior is to fetch single latest span artifact.
        latest_artifact = previous_artifacts[0]
        artifacts_dict[k].append(
            artifact_utils.deserialize_artifact(
                latest_artifact.type, latest_artifact.artifact))
        resolve_state_dict[k] = True

    if self._merge_same_artifact_type:
      # Update resolver_state_dict properly if merging into one channel.
      if self._range_config:
        if self._range_config.HasField('static_range'):
          resolve_state_dict[Examples.TYPE_NAME] = (
              len(artifacts_dict[Examples.TYPE_NAME]) == (
                  self._range_config.static_range.end_span_number - 
                  self._range_config.static_range.start_span_number + 1))
        elif self._range_config.HasField('rolling_range'):
          resolve_state_dict[Examples.TYPE_NAME] = (
              len(artifacts_dict[Examples.TYPE_NAME]) == (
                  self._range_config.rolling_range.num_spans))

    return base_resolver.ResolveResult(
        per_key_resolve_result=artifacts_dict,
        per_key_resolve_state=resolve_state_dict)
