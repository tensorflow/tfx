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
"""Test for SpansResolver."""

from typing import Text
# Standard Imports

import tensorflow as tf
from tfx import types
from tfx.components.example_gen import utils
from tfx.dsl.experimental import spans_resolver
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.proto import range_config_pb2
from tfx.types import standard_artifacts

from ml_metadata.proto import metadata_store_pb2


class SpansResolverTest(tf.test.TestCase):

  def setUp(self):
    super(SpansResolverTest, self).setUp()
    self._connection_config = metadata_store_pb2.ConnectionConfig()
    self._connection_config.sqlite.SetInParent()
    self._pipeline_info = data_types.PipelineInfo(
        pipeline_name='my_pipeline', pipeline_root='/tmp', run_id='my_run_id')
    self._component_info = data_types.ComponentInfo(
        component_type='a.b.c',
        component_id='my_component',
        pipeline_info=self._pipeline_info)

  def _createExamples(self, span: Text) -> standard_artifacts.Examples:
    artifact = standard_artifacts.Examples()
    artifact.uri = 'uri' + span
    artifact.set_string_custom_property(utils.SPAN_PROPERTY_NAME, span)
    return artifact

  def testResolve(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      contexts = m.register_pipeline_contexts_if_not_exists(self._pipeline_info)
      artifact_one = standard_artifacts.Examples()
      artifact_one.uri = 'uri_one'
      artifact_one.set_string_custom_property(utils.SPAN_PROPERTY_NAME, '1')
      m.publish_artifacts([artifact_one])
      artifact_two = standard_artifacts.Examples()
      artifact_two.uri = 'uri_two'
      artifact_two.set_string_custom_property(utils.SPAN_PROPERTY_NAME, '2')
      m.register_execution(
          exec_properties={},
          pipeline_info=self._pipeline_info,
          component_info=self._component_info,
          contexts=contexts)
      m.publish_execution(
          component_info=self._component_info,
          output_artifacts={'key': [artifact_one, artifact_two]})

      resolver = spans_resolver.SpansResolver(
          range_config=range_config_pb2.RangeConfig(
              static_range=range_config_pb2.StaticRange(
                  start_span_number=1, end_span_number=1)))
      resolve_result = resolver.resolve(
          pipeline_info=self._pipeline_info,
          metadata_handler=m,
          source_channels={
              'input':
                  types.Channel(
                      type=artifact_one.type,
                      producer_component_id=self._component_info.component_id,
                      output_key='key')
          })

      self.assertTrue(resolve_result.has_complete_result)
      self.assertEqual([
          artifact.uri
          for artifact in resolve_result.per_key_resolve_result['input']
      ], [artifact_one.uri])
      self.assertTrue(resolve_result.per_key_resolve_state['input'])

  def testResolveArtifacts(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      artifact1 = self._createExamples('1')
      artifact2 = self._createExamples('2')
      artifact3 = self._createExamples('3')
      artifact4 = self._createExamples('4')
      artifact5 = self._createExamples('5')

      # Test StaticRange.
      resolver = spans_resolver.SpansResolver(
          range_config=range_config_pb2.RangeConfig(
              static_range=range_config_pb2.StaticRange(
                  start_span_number=2, end_span_number=3)))
      result = resolver.resolve_artifacts(
          m, {'input': [artifact1, artifact2, artifact3, artifact4, artifact5]})
      self.assertIsNotNone(result)
      self.assertEqual([a.uri for a in result['input']],
                       [artifact3.uri, artifact2.uri])

      # Test RollingRange.
      resolver = spans_resolver.SpansResolver(
          range_config=range_config_pb2.RangeConfig(
              rolling_range=range_config_pb2.RollingRange(num_spans=3)))
      result = resolver.resolve_artifacts(
          m, {'input': [artifact1, artifact2, artifact3, artifact4, artifact5]})
      self.assertIsNotNone(result)
      self.assertEqual([a.uri for a in result['input']],
                       [artifact5.uri, artifact4.uri, artifact3.uri])

      # Test RollingRange with start_span_number.
      resolver = spans_resolver.SpansResolver(
          range_config=range_config_pb2.RangeConfig(
              rolling_range=range_config_pb2.RollingRange(
                  start_span_number=4, num_spans=3)))
      result = resolver.resolve_artifacts(
          m, {'input': [artifact1, artifact2, artifact3, artifact4, artifact5]})
      self.assertIsNotNone(result)
      self.assertEqual([a.uri for a in result['input']],
                       [artifact5.uri, artifact4.uri])


if __name__ == '__main__':
  tf.test.main()
