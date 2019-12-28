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

from ml_metadata.proto import metadata_store_pb2
from tfx import types
from tfx.dsl.resolvers import base_resolver
from tfx.orchestration import metadata


def _generate_tfx_artifact(mlmd_artifact: metadata_store_pb2.Artifact,
                           type_name: Text):
  result = types.Artifact(type_name=type_name)
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
      metadata_handler: metadata.Metadata,
      source_channels: Dict[Text, types.Channel],
  ) -> base_resolver.ResolveResult:
    artifacts_dict = {}
    resolve_state_dict = {}
    for k, c in source_channels.items():
      previous_artifacts = sorted(
          metadata_handler.get_artifacts_by_type(c.type_name),
          key=lambda m: m.id,
          reverse=True)
      if len(previous_artifacts) >= self._desired_num_of_artifact:
        artifacts_dict[k] = [
            _generate_tfx_artifact(a, c.type_name)
            for a in previous_artifacts[:self._desired_num_of_artifact]
        ]
        resolve_state_dict[k] = True
      else:
        artifacts_dict[k] = [
            _generate_tfx_artifact(a, c.type_name) for a in previous_artifacts
        ]
        resolve_state_dict[k] = False

    return base_resolver.ResolveResult(
        per_key_resolve_result=artifacts_dict,
        per_key_resolve_state=resolve_state_dict)
