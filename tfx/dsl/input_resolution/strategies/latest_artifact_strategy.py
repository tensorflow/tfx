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
"""Experimental Resolver for getting the latest artifact."""

from typing import Dict, List, Optional

from tfx import types
from tfx.dsl.components.common import resolver
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.types import artifact_utils
from tfx.utils import doc_controls

import ml_metadata as mlmd


class LatestArtifactStrategy(resolver.ResolverStrategy):
  """Strategy that resolves the latest n(=1) artifacts per each channel.

  Note that this ResolverStrategy is experimental and is subject to change in
  terms of both interface and implementation.

  Don't construct LatestArtifactStrategy directly, example usage:
  ```
    model_resolver = Resolver(
        instance_name='latest_model_resolver',
        strategy_class=LatestArtifactStrategy,
        model=Channel(type=Model))
    model_resolver.outputs['model']
  ```
  """

  def __init__(self, desired_num_of_artifacts: Optional[int] = 1):
    self._desired_num_of_artifact = desired_num_of_artifacts

  def _resolve(self, input_dict: Dict[str, List[types.Artifact]]):
    result = {}
    for k, artifact_list in input_dict.items():
      sorted_artifact_list = sorted(
          artifact_list, key=lambda a: a.id, reverse=True)
      result[k] = sorted_artifact_list[:min(
          len(sorted_artifact_list), self._desired_num_of_artifact)]
    return result

  @doc_controls.do_not_generate_docs
  def resolve(
      self,
      pipeline_info: data_types.PipelineInfo,
      metadata_handler: metadata.Metadata,
      source_channels: Dict[str, types.Channel],
  ) -> resolver.ResolveResult:
    pipeline_context = metadata_handler.get_pipeline_context(pipeline_info)
    if pipeline_context is None:
      raise RuntimeError('Pipeline context absent for %s' % pipeline_context)

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
        k: len(artifact_list) >= self._desired_num_of_artifact
        for k, artifact_list in resolved_dict.items()
    }

    return resolver.ResolveResult(
        per_key_resolve_result=resolved_dict,
        per_key_resolve_state=resolve_state_dict)

  @doc_controls.do_not_generate_docs
  def resolve_artifacts(
      self, store: mlmd.MetadataStore,
      input_dict: Dict[str, List[types.Artifact]]
  ) -> Optional[Dict[str, List[types.Artifact]]]:
    """Resolves artifacts from channels by querying MLMD.

    Args:
      store: An MLMD MetadataStore object.
      input_dict: The input_dict to resolve from.

    Returns:
      If `min_count` for every input is met, returns a
      Dict[str, List[Artifact]]. Otherwise, return None.
    """
    resolved_dict = self._resolve(input_dict)
    all_min_count_met = all(
        len(artifact_list) >= self._desired_num_of_artifact
        for artifact_list in resolved_dict.values())
    return resolved_dict if all_min_count_met else None
