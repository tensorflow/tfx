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
from tfx.components.evaluator import constants as evaluator
from tfx.dsl.components.common import resolver
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.utils import doc_controls

import ml_metadata as mlmd


class LatestBlessedModelStrategy(resolver.ResolverStrategy):
  """LatestBlessedModelStrategy resolves the latest blessed Model artifact.

  Note that this ResolverStrategy is experimental and is subject to change in
  terms of both interface and implementation.

  Don't construct LatestBlessedModelStrategy directly, example usage:
  ```
    model_resolver = Resolver(
        instance_name='latest_blessed_model_resolver',
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing))
    model_resolver.outputs['model']
  ```
  """

  def _resolve(self, input_dict: Dict[str, List[types.Artifact]],
               model_channel_key: str, model_blessing_channel_key: str):
    all_models = input_dict[model_channel_key]
    all_models.sort(key=lambda a: a.id, reverse=True)
    all_model_blessings = input_dict[model_blessing_channel_key]

    # Makes a dict of {model_id : ModelBlessing artifact} for blessed models.
    all_blessed_model_ids = dict(
        (  # pylint: disable=g-complex-comprehension
            a.get_int_custom_property(
                evaluator.ARTIFACT_PROPERTY_CURRENT_MODEL_ID_KEY), a)
        for a in all_model_blessings
        if a.get_int_custom_property(
            evaluator.ARTIFACT_PROPERTY_BLESSED_KEY) == 1)

    result = {model_channel_key: [], model_blessing_channel_key: []}
    # Iterates all models, if blessed, set as result. As the model list was
    # sorted, it is guaranteed to get the latest blessed model.
    for model in all_models:
      if model.id in all_blessed_model_ids:
        result[model_channel_key] = [model]
        model_blessing = all_blessed_model_ids[model.id]
        result[model_blessing_channel_key] = [model_blessing]
        break

    return result

  @doc_controls.do_not_generate_docs
  def resolve(
      self,
      pipeline_info: data_types.PipelineInfo,
      metadata_handler: metadata.Metadata,
      source_channels: Dict[str, types.Channel],
  ) -> resolver.ResolveResult:
    # First, checks whether we have exactly Model and ModelBlessing Channels.
    model_channel_key = None
    model_blessing_channel_key = None
    assert len(source_channels) == 2, 'Expecting 2 input Channels'
    for k, c in source_channels.items():
      if issubclass(c.type, standard_artifacts.Model):
        model_channel_key = k
      elif issubclass(c.type, standard_artifacts.ModelBlessing):
        model_blessing_channel_key = k
      else:
        raise RuntimeError('Only expecting Model or ModelBlessing, got %s' %
                           c.type)
    assert model_channel_key is not None, 'Expecting Model as input'
    assert model_blessing_channel_key is not None, ('Expecting ModelBlessing as'
                                                    ' input')

    model_channel = source_channels[model_channel_key]
    model_blessing_channel = source_channels[model_blessing_channel_key]
    # Gets the pipeline context as the artifact search space.
    pipeline_context = metadata_handler.get_pipeline_context(pipeline_info)
    if pipeline_context is None:
      raise RuntimeError('Pipeline context absent for %s' % pipeline_context)

    candidate_dict = {}
    # Gets all models in the search space and sort in reverse order by id.
    all_models = metadata_handler.get_qualified_artifacts(
        contexts=[pipeline_context],
        type_name=model_channel.type_name,
        producer_component_id=model_channel.producer_component_id,
        output_key=model_channel.output_key)
    candidate_dict[model_channel_key] = [
        artifact_utils.deserialize_artifact(a.type, a.artifact)
        for a in all_models
    ]
    # Gets all ModelBlessing artifacts in the search space.
    all_model_blessings = metadata_handler.get_qualified_artifacts(
        contexts=[pipeline_context],
        type_name=model_blessing_channel.type_name,
        producer_component_id=model_blessing_channel.producer_component_id,
        output_key=model_blessing_channel.output_key)
    candidate_dict[model_blessing_channel_key] = [
        artifact_utils.deserialize_artifact(a.type, a.artifact)
        for a in all_model_blessings
    ]

    resolved_dict = self._resolve(candidate_dict, model_channel_key,
                                  model_blessing_channel_key)
    resolve_state_dict = {
        k: bool(artifact_list) for k, artifact_list in resolved_dict.items()
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

    Raises:
      RuntimeError: if input_dict contains unsupported artifact types.
    """
    model_channel_key = None
    model_blessing_channel_key = None
    assert len(input_dict) == 2, 'Expecting 2 input Channels'
    for k, artifact_list in input_dict.items():
      if not artifact_list:
        # If model or model blessing channel has no artifacts, the min_count
        # can not be met, short cut to return None here.
        return None
      artifact = artifact_list[0]
      if issubclass(type(artifact), standard_artifacts.Model):
        model_channel_key = k
      elif issubclass(type(artifact), standard_artifacts.ModelBlessing):
        model_blessing_channel_key = k
      else:
        raise RuntimeError('Only expecting Model or ModelBlessing, got %s' %
                           artifact.TYPE_NAME)
    assert model_channel_key is not None, 'Expecting Model as input'
    assert model_blessing_channel_key is not None, ('Expecting ModelBlessing as'
                                                    ' input')

    resolved_dict = self._resolve(input_dict, model_channel_key,
                                  model_blessing_channel_key)
    all_min_count_met = all(
        bool(artifact_list) for artifact_list in resolved_dict.values())
    return resolved_dict if all_min_count_met else None
