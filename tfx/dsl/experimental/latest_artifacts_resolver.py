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

from typing import Dict, Optional, Text, Type

from ml_metadata.proto import metadata_store_pb2
from tfx import types
from tfx.dsl.resolvers import base_resolver
from tfx.orchestration import data_types
from tfx.orchestration import metadata


def _generate_tfx_artifact(mlmd_artifact: metadata_store_pb2.Artifact,
                           artifact_type: Type[types.Artifact]):
  result = artifact_type()
  result.set_mlmd_artifact(mlmd_artifact)
  return result


class LatestArtifactsResolver(base_resolver.BaseResolver):
  """Resolver that return the latest n artifacts in a given channel.

  Note that this Resolver is experimental and is subject to change in terms of
  both interface and implementation.
  """

  def __init__(self, desired_num_of_artifacts: Optional[int] = 1):
    self._desired_num_of_artifact = desired_num_of_artifacts

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
    artifacts_in_context = metadata_handler.get_published_artifacts_by_type_within_context(
        [c.type_name for c in source_channels.values()], pipeline_context.id)
    for k, c in source_channels.items():
      previous_artifacts = sorted(
          artifacts_in_context[c.type_name], key=lambda m: m.id, reverse=True)
      if len(previous_artifacts) >= self._desired_num_of_artifact:
        artifacts_dict[k] = [
            _generate_tfx_artifact(a, c.type)
            for a in previous_artifacts[:self._desired_num_of_artifact]
        ]
        resolve_state_dict[k] = True
      else:
        artifacts_dict[k] = [
            _generate_tfx_artifact(a, c.type) for a in previous_artifacts
        ]
        resolve_state_dict[k] = False

    return base_resolver.ResolveResult(
        per_key_resolve_result=artifacts_dict,
        per_key_resolve_state=resolve_state_dict)
