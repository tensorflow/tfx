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

from typing import Dict, Text

from tfx import types
from tfx.components.evaluator import constants as evaluator
from tfx.dsl.resolvers import base_resolver
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.types import artifact_utils
from tfx.types import standard_artifacts


class LatestBlessedModelResolver(base_resolver.BaseResolver):
  """Special Resolver that return the latest blessed model.

  Note that this Resolver is experimental and is subject to change in terms of
  both interface and implementation.
  """

  def resolve(
      self,
      pipeline_info: data_types.PipelineInfo,
      metadata_handler: metadata.Metadata,
      source_channels: Dict[Text, types.Channel],
  ) -> base_resolver.ResolveResult:
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

    # Gets all models in the search space and sort in reverse order by id.
    all_models = metadata_handler.get_qualified_artifacts(
        contexts=[pipeline_context],
        type_name=model_channel.type_name,
        producer_component_id=model_channel.producer_component_id,
        output_key=model_channel.output_key)
    all_models.sort(key=lambda a: a.artifact.id, reverse=True)
    # Gets all ModelBlessing artifacts in the search space.
    all_model_blessings = metadata_handler.get_qualified_artifacts(
        contexts=[pipeline_context],
        type_name=model_blessing_channel.type_name,
        producer_component_id=model_blessing_channel.producer_component_id,
        output_key=model_blessing_channel.output_key)
    # Makes a dict of {model_id : ModelBlessing artifact} for blessed models.
    all_blessed_model_ids = dict(
        (  # pylint: disable=g-complex-comprehension
            a.artifact.custom_properties[
                evaluator.ARTIFACT_PROPERTY_CURRENT_MODEL_ID_KEY].int_value, a)
        for a in all_model_blessings
        if a.artifact.custom_properties[
            evaluator.ARTIFACT_PROPERTY_BLESSED_KEY].int_value == 1)

    artifacts_dict = {model_channel_key: [], model_blessing_channel_key: []}
    resolve_state_dict = {
        model_channel_key: False,
        model_blessing_channel_key: False
    }
    # Iterates all models, if blessed, set as result. As the model list was
    # sorted, it is guaranteed to get the latest blessed model.
    for model in all_models:
      if model.artifact.id in all_blessed_model_ids:
        artifacts_dict[model_channel_key] = [
            artifact_utils.deserialize_artifact(model.type, model.artifact)
        ]
        model_blessing = all_blessed_model_ids[model.artifact.id]
        artifacts_dict[model_blessing_channel_key] = [
            artifact_utils.deserialize_artifact(model_blessing.type,
                                                model_blessing.artifact)
        ]
        resolve_state_dict[model_channel_key] = True
        resolve_state_dict[model_blessing_channel_key] = True
        break

    return base_resolver.ResolveResult(
        per_key_resolve_result=artifacts_dict,
        per_key_resolve_state=resolve_state_dict)
