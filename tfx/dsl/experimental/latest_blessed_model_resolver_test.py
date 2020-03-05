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
"""Resolver for getting latest n artifacts."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Standard Imports
import tensorflow as tf
from ml_metadata.proto import metadata_store_pb2
from tfx import types
from tfx.components.model_validator import constants as model_validator
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.types import standard_artifacts


class LatestBlessedModelResolverTest(tf.test.TestCase):

  def setUp(self):
    super(LatestBlessedModelResolverTest, self).setUp()
    self._connection_config = metadata_store_pb2.ConnectionConfig()
    self._connection_config.sqlite.SetInParent()
    self._pipeline_info = data_types.PipelineInfo(
        pipeline_name='my_pipeline', pipeline_root='/tmp', run_id='my_run_id')
    self._component_info = data_types.ComponentInfo(
        component_type='a.b.c',
        component_id='my_component',
        pipeline_info=self._pipeline_info)

  def _set_model_blessing_bit(self, artifact: types.Artifact, model_id: int,
                              is_blessed: int):
    artifact.mlmd_artifact.custom_properties[
        model_validator.ARTIFACT_PROPERTY_BLESSED_KEY].int_value = is_blessed
    artifact.mlmd_artifact.custom_properties[
        model_validator
        .ARTIFACT_PROPERTY_CURRENT_MODEL_ID_KEY].int_value = model_id

  def testGetLatestBlessedModelArtifact(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      contexts = m.register_pipeline_contexts_if_not_exists(self._pipeline_info)
      # Model with id 1, will be blessed.
      model_one = standard_artifacts.Model()
      model_one.uri = 'model_one'
      m.publish_artifacts([model_one])
      # Model with id 2, will be blessed.
      model_two = standard_artifacts.Model()
      model_two.uri = 'model_two'
      m.publish_artifacts([model_two])
      # Model with id 3, will not be blessed.
      model_three = standard_artifacts.Model()
      model_three.uri = 'model_three'
      m.publish_artifacts([model_three])

      model_blessing_one = standard_artifacts.ModelBlessing()
      self._set_model_blessing_bit(model_blessing_one, model_one.id, 1)
      model_blessing_two = standard_artifacts.ModelBlessing()
      self._set_model_blessing_bit(model_blessing_two, model_two.id, 1)
      m.publish_artifacts([model_blessing_one, model_blessing_two])

      m.register_execution(
          exec_properties={},
          pipeline_info=self._pipeline_info,
          component_info=self._component_info,
          contexts=contexts)
      m.publish_execution(
          component_info=self._component_info,
          output_artifacts={
              'a': [model_one, model_two, model_three],
              'b': [model_blessing_one, model_blessing_two]
          })

      resolver = latest_blessed_model_resolver.LatestBlessedModelResolver()
      resolve_result = resolver.resolve(
          pipeline_info=self._pipeline_info,
          metadata_handler=m,
          source_channels={
              'model':
                  types.Channel(
                      type=standard_artifacts.Model,
                      producer_component_id=self._component_info.component_id,
                      output_key='a'),
              'model_blessing':
                  types.Channel(
                      type=standard_artifacts.ModelBlessing,
                      producer_component_id=self._component_info.component_id,
                      output_key='b')
          })
      self.assertTrue(resolve_result.has_complete_result)
      self.assertEqual([
          a.uri
          for a in resolve_result.per_key_resolve_result['model']
      ], ['model_two'])
      self.assertTrue(resolve_result.per_key_resolve_state['model'])


if __name__ == '__main__':
  tf.test.main()
