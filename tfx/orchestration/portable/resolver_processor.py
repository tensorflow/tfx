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
from tfx.orchestration import metadata
from tfx.proto.orchestration import pipeline_pb2


class ResolverProcessor(object):
  """PythonResolverOperator resolves artifacts in process."""

  # TODO(muyangy): This map is subject to change if the structure of
  # ResolverConfig changes.
  _RESOLVER_POLICY_TO_RESOLVER_CLASS = {
      pipeline_pb2.ResolverConfig.RESOLVER_POLICY_UNSPECIFIED:
          None,
      pipeline_pb2.ResolverConfig.LATEST_ARTIFACT:
          latest_artifacts_resolver.LatestArtifactsResolver,
      pipeline_pb2.ResolverConfig.LATEST_BLESSED_MODEL:
          latest_blessed_model_resolver.LatestBlessedModelResolver,
  }

  def __init__(self, node_inputs: pipeline_pb2.NodeInputs):
    resolver_policy = node_inputs.resolver_config.resolver_policy
    if resolver_policy not in self._RESOLVER_POLICY_TO_RESOLVER_CLASS:
      raise ValueError(
          "Resolver_policy {} is not supported.".format(resolver_policy))
    resolver_class = self._RESOLVER_POLICY_TO_RESOLVER_CLASS.get(
        resolver_policy)
    self._resolver = None
    if resolver_class:
      self._resolver = resolver_class()

  def ResolveInputs(
      self, metadata_handler: metadata.Metadata,
      input_dict: Dict[str, List[types.Artifact]]
  ) -> Optional[Dict[str, List[types.Artifact]]]:
    """Resolves artifacts in input_dict by optionally querying MLMD.

    Args:
      metadata_handler: A metadata handler to access MLMD store.
      input_dict: Inputs to be resolved.

    Returns:
      The resolved input_dict.
    """
    return (self._resolver.resolve_artifacts(metadata_handler, input_dict)
            if self._resolver else input_dict)
