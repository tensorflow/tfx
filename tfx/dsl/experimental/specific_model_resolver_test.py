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
"""Tests for third_party.py.tfx.dsl.experimental.specific_model_resolver."""

# TODO(b/149535307): Remove __future__ imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Standard Imports

import tensorflow as tf
from tfx import types
from tfx.components.model_validator import constants as model_validator
from tfx.dsl.experimental import specific_model_resolver
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.types import standard_artifacts

from ml_metadata.proto import metadata_store_pb2


class SpecificModelResolverTest(tf.test.TestCase):

  def setUp(self):
    super(SpecificModelResolverTest, self).setUp()
    self._connection_config = metadata_store_pb2.ConnectionConfig()
    self._connection_config.sqlite.SetInParent()
    self._pipeline_info = data_types.PipelineInfo(
        pipeline_name='my_pipeline', pipeline_root='/tmp', run_id='my_run_id')
    self._component_info = data_types.ComponentInfo(
        component_type='a.b.c',
        component_id='my_component',
        pipeline_info=self._pipeline_info)

  def _set_existing_model(self, m: metadata.Metadata):
    contexts = m.register_pipeline_contexts_if_not_exists(self._pipeline_info)
    # Model with id 1, will be blessed.
    model_one = standard_artifacts.Model()
    model_one.uri = 'model_one'
    m.publish_artifacts([model_one])

    model_blessing_one = standard_artifacts.ModelBlessing()
    self._set_model_blessing_bit(model_blessing_one, model_one.id, 1)
    m.publish_artifacts([model_blessing_one])

    m.register_execution(
        exec_properties={},
        pipeline_info=self._pipeline_info,
        component_info=self._component_info,
        contexts=contexts)
    m.publish_execution(
        component_info=self._component_info,
        output_artifacts={
            'a': [model_one],
            'b': [model_blessing_one]
        })

  def _set_model_blessing_bit(self, artifact: types.Artifact, model_id: int,
                              is_blessed: int):
    artifact.mlmd_artifact.custom_properties[
        model_validator.ARTIFACT_PROPERTY_BLESSED_KEY].int_value = is_blessed
    artifact.mlmd_artifact.custom_properties[
        model_validator
        .ARTIFACT_PROPERTY_CURRENT_MODEL_ID_KEY].int_value = model_id

  def testResolve_WithInvalidPipelineInfo_ExpectedFail(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      self._set_existing_model(m)
      resolver = specific_model_resolver.SpecificModelResolver(
          data_types.PipelineInfo(
              pipeline_name='my_pipeline',
              pipeline_root='/tmp',
              run_id='my_run_id2'))
      with self.assertRaises(ValueError):
        resolver.resolve(
            pipeline_info=None,
            metadata_handler=m,
            source_channels={
                'model':
                    types.Channel(type=standard_artifacts.Model),
                'model_blessing':
                    types.Channel(type=standard_artifacts.ModelBlessing)
            })

  def testResolve(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      self._set_existing_model(m)
      resolver = specific_model_resolver.SpecificModelResolver(
          self._pipeline_info)
      result = resolver.resolve(
          pipeline_info=None,
          metadata_handler=m,
          source_channels={
              'model':
                  types.Channel(type=standard_artifacts.Model),
              'model_blessing':
                  types.Channel(type=standard_artifacts.ModelBlessing)
          })
      self.assertTrue(result.has_complete_result)
      self.assertEqual([a.uri for a in result.per_key_resolve_result['model']],
                       ['model_one'])
      self.assertTrue(result.per_key_resolve_state['model'])


if __name__ == '__main__':
  tf.test.main()
