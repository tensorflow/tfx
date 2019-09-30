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

from typing import Dict, Text

from tfx import types
from tfx.dsl.resolvers import base_resolver
from tfx.orchestration import metadata


class LatestArtifactResolver(base_resolver.BaseResolver):
  """Resolver that return the latest artifact in a given channel.

  Note that this Resolver is experimental and is subject to change in terms of
  both interface and implementation.
  """

  def resolve(
      self,
      metadata_handler: metadata.Metadata,
      source_channels: Dict[Text, types.Channel],
  ) -> base_resolver.ResolveResult:
    artifacts_dict = {}
    for k, c in source_channels.items():
      previous_artifacts = metadata_handler.get_artifacts_by_type(c.type_name)
      if previous_artifacts:
        latest_mlmd_artifact = max(previous_artifacts, key=lambda m: m.id)
        result_artifact = types.Artifact(type_name=c.type_name)
        result_artifact.set_artifact(latest_mlmd_artifact)
        artifacts_dict[k] = ([result_artifact], True)
      else:
        artifacts_dict[k] = ([], False)

    return base_resolver.ResolveResult(per_key_resolve_result=artifacts_dict)
