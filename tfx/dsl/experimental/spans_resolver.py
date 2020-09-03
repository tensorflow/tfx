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
               range_config: Optional[range_config_pb2.RangeConfig] = None):
    self._range_config = range_config

  def resolve(
      self,
      pipeline_info: data_types.PipelineInfo,
      metadata_handler: metadata.Metadata,
      source_channels: Dict[Text, types.Channel],
  ) -> base_resolver.ResolveResult:
    artifacts_dict = {}
    resolve_state_dict = {}
    pipeline_context = metadata_handler.get_pipeline_context(pipeline_info)
    if pipeline_context is None:
      raise RuntimeError('Pipeline context absent for %s' % pipeline_context)
    for k, c in source_channels.items():
      if c.type_name != Examples.TYPE_NAME:
        raise ValueError('Resolving non-Example artifacts is not supported.')

      candidate_artifacts = metadata_handler.get_qualified_artifacts(
          contexts=[pipeline_context],
          type_name=c.type_name,
          producer_component_id=c.producer_component_id,
          output_key=c.output_key)

      previous_artifacts = sorted(
          candidate_artifacts, 
          key=lambda a: int(
              a.artifact.custom_properties[utils.SPAN_PROPERTY_NAME].string_value),
          reverse=True)

      if self._range_config:
        if self._range_config.HasField('static_range'):
          # TODO(jjma): Optimize this by sending a more specific MLMD query.
          artifacts_dict[k] = []
          lower_bound = self._range_config.static_range.start_span_number
          upper_bound = self._range_config.static_range.end_span_number
          for a in previous_artifacts:
            span = int(
                a.artifact.custom_properties[utils.SPAN_PROPERTY_NAME].string_value)
            if lower_bound <= span and span <= upper_bound:
              artifacts_dict[k].append(
                  artifact_utils.deserialize_artifact(a.type, a.artifact))
        
          resolve_state_dict[k] = (
              len(artifacts_dict[k]) == (upper_bound - lower_bound + 1))

        # TODO(jjma): Add rolling range support after adding RollingRange.

      else:
        # Default behavior is to fetch single latest span artifact.
        latest_artifact = previous_artifacts[0]
        artifacts_dict[k] = [
            artifact_utils.deserialize_artifact(
                latest_artifact.type, latest_artifact.artifact)
        ]
        resolve_state_dict[k] = True

    return base_resolver.ResolveResult(
        per_key_resolve_result=artifacts_dict,
        per_key_resolve_state=resolve_state_dict)
