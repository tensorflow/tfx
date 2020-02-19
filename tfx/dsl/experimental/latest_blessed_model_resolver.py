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

from typing import Dict, Text, Type

from ml_metadata.proto import metadata_store_pb2
from tfx import types
from tfx.components.evaluator import constants as evaluator
from tfx.dsl.resolvers import base_resolver
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.types import standard_artifacts


def _generate_tfx_artifact(mlmd_artifact: metadata_store_pb2.Artifact,
                           artifact_type: Type[types.Artifact]):
  result = artifact_type()
  result.set_mlmd_artifact(mlmd_artifact)
  return result


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

    # Gets the pipeline context as the artifact search space.
    pipeline_context = metadata_handler.get_pipeline_context(pipeline_info)
    if pipeline_context is None:
      raise RuntimeError('Pipeline context absent for %s' % pipeline_context)
    # Gets all artifacts of interests within context with one call.
    artifacts_in_context = metadata_handler.get_published_artifacts_by_type_within_context(
        [
            source_channels[model_channel_key].type_name,
            source_channels[model_blessing_channel_key].type_name
        ], pipeline_context.id)
    # Gets all models in the search space and sort in reverse order by id.
    all_models = sorted(
        artifacts_in_context[source_channels[model_channel_key].type_name],
        key=lambda m: m.id,
        reverse=True)
    # Gets all ModelBlessing artifacts in the search space.
    all_model_blessings = artifacts_in_context[
        source_channels[model_blessing_channel_key].type_name]
    # Makes a dict of {model_id : ModelBlessing artifact} for blessed models.
    all_blessed_model_ids = dict(
        (  # pylint: disable=g-complex-comprehension
            a.custom_properties[
                evaluator.ARTIFACT_PROPERTY_CURRENT_MODEL_ID_KEY].int_value, a)
        for a in all_model_blessings
        if a.custom_properties[
            evaluator.ARTIFACT_PROPERTY_BLESSED_KEY].int_value == 1)

    artifacts_dict = {model_channel_key: [], model_blessing_channel_key: []}
    resolve_state_dict = {
        model_channel_key: False,
        model_blessing_channel_key: False
    }
    # Iterates all models, if blessed, set as result. As the model list was
    # sorted, it is guaranteed to get the latest blessed model.
    for model in all_models:
      if model.id in all_blessed_model_ids:
        artifacts_dict[model_channel_key] = [
            _generate_tfx_artifact(model, standard_artifacts.Model)
        ]
        artifacts_dict[model_blessing_channel_key] = [
            _generate_tfx_artifact(all_blessed_model_ids[model.id],
                                   standard_artifacts.ModelBlessing)
        ]
        resolve_state_dict[model_channel_key] = True
        resolve_state_dict[model_blessing_channel_key] = True
        break

    return base_resolver.ResolveResult(
        per_key_resolve_result=artifacts_dict,
        per_key_resolve_state=resolve_state_dict)
