# Lint as: python3
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
"""Experimental Resolver for getting the specific model artifact."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Dict
from typing import Text

from tfx import types
from tfx.components.evaluator import constants as evaluator
from tfx.dsl.resolvers import base_resolver
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.types import artifact_utils
from tfx.types import standard_artifacts


class SpecificModelResolver(base_resolver.BaseResolver):
  """Resolver that returns the specific model in a given pipeline_info.

  Note that this Resolver is experimental and is subject to change in terms of
  both interface and implementation.
  """

  def __init__(self, pipeline_info: data_types.PipelineInfo):
    self._pipeline_info = pipeline_info

  def resolve(
      self,
      pipeline_info: data_types.PipelineInfo,
      metadata_handler: metadata.Metadata,
      source_channels: Dict[Text, types.Channel],
  ) -> base_resolver.ResolveResult:
    del pipeline_info
    for k, c in source_channels.items():
      if issubclass(c.type, standard_artifacts.Model):
        model_channel_key = k
      elif issubclass(c.type, standard_artifacts.ModelBlessing):
        model_blessing_channel_key = k

    assert model_channel_key is not None, 'Expecting Model as input'
    assert model_blessing_channel_key is not None, ('Expecting ModelBlessing as'
                                                    ' input')

    model_channel = source_channels[model_channel_key]
    model_blessing_channel = source_channels[model_blessing_channel_key]

    pipeline_context = metadata_handler.get_pipeline_run_context(
        self._pipeline_info)

    if pipeline_context is None:
      raise ValueError('Pipeline context absent for %s' % self._pipeline_info)
    published_artifacts = metadata_handler.get_published_artifacts_by_type_within_context(
        [model_blessing_channel.type_name, model_channel.type_name],
        pipeline_context.id)
    if not published_artifacts[model_channel.type_name]:
      raise ValueError('Model absent for %s' % self._pipeline_info)
    if len(published_artifacts[model_channel.type_name]) > 1:
      raise RuntimeError('More than 1 model to push for %s' %
                         self._pipeline_info)

    model_blessing_map = {}
    for model_blessing in published_artifacts[model_blessing_channel.type_name]:
      if model_blessing.custom_properties[
          evaluator.ARTIFACT_PROPERTY_BLESSED_KEY].int_value == 1:
        model_blessing_map[model_blessing.custom_properties[
            evaluator.ARTIFACT_PROPERTY_CURRENT_MODEL_ID_KEY]
                           .int_value] = model_blessing

    artifacts_dict = {model_channel_key: [], model_blessing_channel_key: []}
    resolve_state_dict = {
        model_channel_key: False,
        model_blessing_channel_key: False
    }

    model = published_artifacts[model_channel.type_name][0]
    artifacts_dict[model_channel_key].append(
        artifact_utils.deserialize_artifact(
            metadata_handler.store.get_artifact_type(model_channel.type_name),
            model))
    resolve_state_dict[model_channel_key] = True
    if model.id in model_blessing_map:
      model_blessing = model_blessing_map[model.id]
      artifacts_dict[model_blessing_channel_key].append(
          artifact_utils.deserialize_artifact(
              metadata_handler.store.get_artifact_type(
                  model_blessing_channel.type_name), model_blessing))
      resolve_state_dict[model_blessing_channel_key] = True
    return base_resolver.ResolveResult(
        per_key_resolve_result=artifacts_dict,
        per_key_resolve_state=resolve_state_dict)
