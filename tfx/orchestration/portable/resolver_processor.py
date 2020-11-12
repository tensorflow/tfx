# Copyright 2020 Google LLC. All Rights Reserved.
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
"""In process inplementation of Resolvers."""

from typing import Dict, List, Optional

from tfx import types
from tfx.dsl.experimental import latest_artifacts_resolver
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.dsl.resolvers import base_resolver
from tfx.proto.orchestration import pipeline_pb2

_ResolverConfig = pipeline_pb2.ResolverConfig
_LatestArtifactsResolver = latest_artifacts_resolver.LatestArtifactsResolver
_LatestBlessedModelResolver = (latest_blessed_model_resolver
                               .LatestBlessedModelResolver)


class ResolverProcessor:
  """ResolverProcessor resolves artifacts in process."""

  def __init__(self, resolver: base_resolver.BaseResolver):
    self._resolver = resolver

  def resolve_inputs(
      self, context: base_resolver.ResolverContext,
      input_dict: Dict[str, List[types.Artifact]]
  ) -> Optional[Dict[str, List[types.Artifact]]]:
    """Resolves artifacts in input_dict by optionally querying MLMD.

    Args:
      context: A ResolverContext for resolver runtime.
      input_dict: Inputs to be resolved.

    Returns:
      The resolved input_dict.
    """
    return self._resolver.resolve_artifacts(context, input_dict)


class ResolverProcessorFactory:
  """Factory class for building ResolverProcessors."""

  @classmethod
  def from_resolver_config(
      cls, resolver_config: _ResolverConfig) -> List[ResolverProcessor]:
    """Build a list of ResolverProcessors from ResolverConfig."""
    resolver_policy = resolver_config.resolver_policy
    if resolver_policy == _ResolverConfig.RESOLVER_POLICY_UNSPECIFIED:
      resolvers = []
    elif resolver_policy == _ResolverConfig.LATEST_ARTIFACT:
      resolvers = [_LatestArtifactsResolver()]
    elif resolver_policy == _ResolverConfig.LATEST_BLESSED_MODEL:
      resolvers = [_LatestBlessedModelResolver()]
    else:
      raise ValueError('Unknown resolver policy {}'.format(resolver_policy))
    return [ResolverProcessor(resolver) for resolver in resolvers]
