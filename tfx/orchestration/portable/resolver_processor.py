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

from typing import Mapping, Sequence, Dict, List, Optional

from tfx import types
from tfx.orchestration import metadata
from tfx.orchestration.portable.resolver import factory as resolver_factory
from tfx.proto.orchestration import pipeline_pb2


class ResolverStepProcessor:
  """ResolverStepProcessor for processing single ResolverStep.

  Note that input and the ouptut type of __call__ is identical, thus resolver
  steps can be chained where the output of the former step would be fed into
  the next step. If the output is None, chained processing will be halted and
  the output of all steps would be considered None immediately.
  """

  def __init__(self, resolver_step: pipeline_pb2.ResolverConfig.ResolverStep):
    self._resolver = resolver_factory.make_resolver_strategy_instance(
        resolver_step)
    self._input_keys = set(resolver_step.input_keys)

  def __call__(
      self, metadata_handler: metadata.Metadata,
      input_dict: Mapping[str, Sequence[types.Artifact]]
  ) -> Optional[Dict[str, List[types.Artifact]]]:
    """Resolves artifacts in input_dict by optionally querying MLMD.

    Args:
      metadata_handler: A metadata handler to access MLMD store.
      input_dict: Inputs to be resolved.

    Returns:
      The resolved input_dict.
    """
    filtered_keys = self._input_keys or set(input_dict.keys())
    filtered_inputs = {
        key: list(value)
        for key, value in input_dict.items()
        if key in filtered_keys
    }
    bypassed_inputs = {
        key: list(value)
        for key, value in input_dict.items()
        if key not in filtered_keys
    }
    result = self._resolver.resolve_artifacts(
        metadata_handler.store,
        filtered_inputs)
    if result is not None:
      result.update(bypassed_inputs)
    return result


def make_resolver_processors(
    resolver_config: pipeline_pb2.ResolverConfig
) -> List[ResolverStepProcessor]:
  """Factory function for ResolverProcessors from ResolverConfig."""
  return [ResolverStepProcessor(step)
          for step in resolver_config.resolver_steps]
