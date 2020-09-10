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

  def testWithNonExampleChannels(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      contexts = m.register_pipeline_contexts_if_not_exists(self._pipeline_info)
      resolver = spans_resolver.SpansResolver()

      with self.assertRaisesRegexp(ValueError, 
          'Channel does not contain Example artifacts'):
        resolve_result = resolver.resolve(
            pipeline_info=self._pipeline_info,
            metadata_handler=m,
            source_channels={
                'input': types.Channel(type=standard_artifacts.Model)
            })

  def testStaticRangeConfig(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      contexts = m.register_pipeline_contexts_if_not_exists(self._pipeline_info)

      artifact_one = standard_artifacts.Examples()
      artifact_one.uri = 'span1'
      artifact_one.set_string_custom_property(utils.SPAN_PROPERTY_NAME, '1')
      m.publish_artifacts([artifact_one])

      m.register_execution(
          exec_properties={},
          pipeline_info=self._pipeline_info,
          component_info=self._component_info,
          contexts=contexts)
      m.publish_execution(
          component_info=self._component_info,
          output_artifacts={'key': [artifact_one]})

      # Resolve any spans from [1, 2], inclusive.
      resolver = spans_resolver.SpansResolver(
          range_config=range_config_pb2.RangeConfig(
             static_range=range_config_pb2.StaticRange(
                 start_span_number=1, end_span_number=2)))

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

      # Currently, only Span-1 exists, so resolver result is incomplete.
      self.assertFalse(resolve_result.has_complete_result)
      self.assertEqual([
          artifact.uri
          for artifact in resolve_result.per_key_resolve_result['input']
      ], [artifact_one.uri])
      self.assertFalse(resolve_result.per_key_resolve_state['input'])

      artifact_two = standard_artifacts.Examples()
      artifact_two.uri = 'span2'
      artifact_two.set_string_custom_property(utils.SPAN_PROPERTY_NAME, '2')
      m.publish_artifacts([artifact_two])

      m.register_execution(
          exec_properties={},
          pipeline_info=self._pipeline_info,
          component_info=self._component_info,
          contexts=contexts)
      m.publish_execution(
          component_info=self._component_info,
          output_artifacts={'key': [artifact_two]})

      resolve_result = resolver.resolve(
          pipeline_info=self._pipeline_info,
          metadata_handler=m,
          source_channels={
              'input':
                  types.Channel(
                      type=artifact_two.type,
                      producer_component_id=self._component_info.component_id,
                      output_key='key')
          })

      # Resolver now picks up both Span-1 and Span-2.
      self.assertTrue(resolve_result.has_complete_result)
      self.assertEqual([
          artifact.uri
          for artifact in resolve_result.per_key_resolve_result['input']
      ], [artifact_two.uri, artifact_one.uri])
      self.assertTrue(resolve_result.per_key_resolve_state['input'])

      artifact_three = standard_artifacts.Examples()
      artifact_three.uri = 'span3hree'
      artifact_three.set_string_custom_property(utils.SPAN_PROPERTY_NAME, '3')
      m.publish_artifacts([artifact_three])

      m.register_execution(
          exec_properties={},
          pipeline_info=self._pipeline_info,
          component_info=self._component_info,
          contexts=contexts)
      m.publish_execution(
          component_info=self._component_info,
          output_artifacts={'key': [artifact_three]})

      resolve_result = resolver.resolve(
          pipeline_info=self._pipeline_info,
          metadata_handler=m,
          source_channels={
              'input':
                  types.Channel(
                      type=artifact_three.type,
                      producer_component_id=self._component_info.component_id,
                      output_key='key')
          })

      # Resolver picks up both Span-1 and Span-2 (ignores out-of-range Span-3).
      self.assertTrue(resolve_result.has_complete_result)
      self.assertEqual([
          artifact.uri
          for artifact in resolve_result.per_key_resolve_result['input']
      ], [artifact_two.uri, artifact_one.uri])
      self.assertTrue(resolve_result.per_key_resolve_state['input'])
      
  def testRollingRangeConfig(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      contexts = m.register_pipeline_contexts_if_not_exists(self._pipeline_info)

      artifact_one = standard_artifacts.Examples()
      artifact_one.uri = 'span1'
      artifact_one.set_string_custom_property(utils.SPAN_PROPERTY_NAME, '1')
      artifact_two = standard_artifacts.Examples()
      artifact_two.uri = 'span2'
      artifact_two.set_string_custom_property(utils.SPAN_PROPERTY_NAME, '2')
      m.publish_artifacts([artifact_one, artifact_two])

      m.register_execution(
          exec_properties={},
          pipeline_info=self._pipeline_info,
          component_info=self._component_info,
          contexts=contexts)
      m.publish_execution(
          component_info=self._component_info,
          output_artifacts={'key': [artifact_one, artifact_two]})

      # Resolve two spans, looking only at spans no earlier than Span-2.
      resolver = spans_resolver.SpansResolver(
          range_config=range_config_pb2.RangeConfig(
             rolling_range=range_config_pb2.RollingRange(
                 start_span_number=2,
                 num_spans=2,
                 skip_num_recent_spans=0)))

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

      # Rolling range starts at Span-2, so only Span-2 is resolved.
      self.assertFalse(resolve_result.has_complete_result)
      self.assertEqual([
          artifact.uri
          for artifact in resolve_result.per_key_resolve_result['input']
      ], [artifact_two.uri])
      self.assertFalse(resolve_result.per_key_resolve_state['input'])

      artifact_three = standard_artifacts.Examples()
      artifact_three.uri = 'span3hree'
      artifact_three.set_string_custom_property(utils.SPAN_PROPERTY_NAME, '3')
      m.publish_artifacts([artifact_three])

      m.register_execution(
          exec_properties={},
          pipeline_info=self._pipeline_info,
          component_info=self._component_info,
          contexts=contexts)
      m.publish_execution(
          component_info=self._component_info,
          output_artifacts={'key': [artifact_three]})

      # Resolve second and third latest spans, looking only at spans no
      # earlier than Span-0.
      resolver = spans_resolver.SpansResolver(
          range_config=range_config_pb2.RangeConfig(
             rolling_range=range_config_pb2.RollingRange(
                 start_span_number=0,
                 num_spans=2,
                 skip_num_recent_spans=1)))

      resolve_result = resolver.resolve(
          pipeline_info=self._pipeline_info,
          metadata_handler=m,
          source_channels={
              'input':
                  types.Channel(
                      type=artifact_two.type,
                      producer_component_id=self._component_info.component_id,
                      output_key='key')
          })

      # Resolver now picks up Spans 1 and 2, ignoring latest Span-3.
      self.assertTrue(resolve_result.has_complete_result)
      self.assertEqual([
          artifact.uri
          for artifact in resolve_result.per_key_resolve_result['input']
      ], [artifact_two.uri, artifact_one.uri])
      self.assertTrue(resolve_result.per_key_resolve_state['input'])

  def testFailOnMultipleChannels(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      contexts = m.register_pipeline_contexts_if_not_exists(self._pipeline_info)  
      resolver = spans_resolver.SpansResolver()
    
      with self.assertRaisesRegexp(ValueError,
          'Resolver must have exactly one source channel'):
        resolve_result = resolver.resolve(
            pipeline_info=self._pipeline_info,
            metadata_handler=m,
            source_channels={
                'input1':
                    types.Channel(type=standard_artifacts.Examples),
                'input2':
                    types.Channel(type=standard_artifacts.Examples)
            })

if __name__ == '__main__':
  tf.test.main()
