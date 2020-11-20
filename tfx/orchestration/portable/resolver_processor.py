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
from tfx.dsl.resolvers import base_resolver
from tfx.orchestration.portable.resolver import factory as resolver_factory
from tfx.proto.orchestration import pipeline_pb2


class ResolverProcessor:
  """ResolverProcessor resolves artifacts in process."""

  def __init__(self, resolver: base_resolver.BaseResolver):
    self._resolver = resolver

  def __call__(
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


def make_resolver_processors(
    resolver_config: pipeline_pb2.ResolverConfig) -> List[ResolverProcessor]:
  """Factory function for ResolverProcessors from ResolverConfig."""
  result = []
  for resolver_pb in resolver_config.resolvers:
    resolver = resolver_factory.get_resolver_instance(resolver_pb)
    result.append(ResolverProcessor(resolver))
  return result
