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
"""Tests for tfx.dsl.components.common.resolver."""

import tensorflow as tf
from tfx import types
from tfx.dsl.components.common import resolver
from tfx.dsl.input_resolution.strategies import latest_artifact_strategy
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.types import standard_artifacts

from ml_metadata.proto import metadata_store_pb2


class ResolverTest(tf.test.TestCase):

  def testResolverDefinition(self):
    channel_to_resolve = types.Channel(type=standard_artifacts.Examples)
    rnode = resolver.Resolver(
        strategy_class=latest_artifact_strategy.LatestArtifactStrategy,
        config={'desired_num_of_artifacts': 5},
        channel_to_resolve=channel_to_resolve)
    self.assertDictEqual(
        rnode.exec_properties, {
            resolver.RESOLVER_STRATEGY_CLASS:
                latest_artifact_strategy.LatestArtifactStrategy,
            resolver.RESOLVER_CONFIG: {
                'desired_num_of_artifacts': 5
            }
        })
    self.assertEqual(rnode.inputs.get_all()['channel_to_resolve'],
                     channel_to_resolve)
    self.assertEqual(rnode.outputs.get_all()['channel_to_resolve'].type_name,
                     channel_to_resolve.type_name)

  def testResolverDefinition_BadChannel(self):
    with self.assertRaisesRegex(
        ValueError,
        'Expected extra kwarg .* to be of type .*tfx.types.Channel'):
      resolver.Resolver(
          strategy_class=latest_artifact_strategy.LatestArtifactStrategy,
          config={'desired_num_of_artifacts': 5},
          not_a_channel=object())

  def testResolverDefinition_BadStrategyClass(self):
    class NotAStrategy:
      pass

    with self.assertRaisesRegex(
        TypeError, 'strategy_class should be ResolverStrategy'):
      resolver.Resolver(strategy_class=NotAStrategy)


class ResolverDriverTest(tf.test.TestCase):

  def setUp(self):
    super(ResolverDriverTest, self).setUp()
    self.connection_config = metadata_store_pb2.ConnectionConfig()
    self.connection_config.sqlite.SetInParent()
    self.pipeline_info = data_types.PipelineInfo(
        pipeline_name='p_name', pipeline_root='p_root', run_id='run_id')
    self.component_info = data_types.ComponentInfo(
        component_type='c_type',
        component_id='c_id',
        pipeline_info=self.pipeline_info)
    self.driver_args = data_types.DriverArgs(enable_cache=True)
    self.source_channel_key = 'source_channel'
    self.source_channels = {
        self.source_channel_key: types.Channel(type=standard_artifacts.Examples)
    }

  def testResolveArtifactSuccess(self):
    existing_artifact = standard_artifacts.Examples()
    existing_artifact.uri = 'my/uri'
    with metadata.Metadata(connection_config=self.connection_config) as m:
      contexts = m.register_pipeline_contexts_if_not_exists(self.pipeline_info)
      m.publish_artifacts([existing_artifact])
      m.register_execution(
          exec_properties={},
          pipeline_info=self.pipeline_info,
          component_info=self.component_info,
          contexts=contexts)
      m.publish_execution(
          component_info=self.component_info,
          output_artifacts={'key': [existing_artifact]})
      driver = resolver._ResolverDriver(metadata_handler=m)
      output_dict = self.source_channels.copy()
      execution_result = driver.pre_execution(
          component_info=self.component_info,
          pipeline_info=self.pipeline_info,
          driver_args=self.driver_args,
          input_dict=self.source_channels,
          output_dict=output_dict,
          exec_properties={
              resolver.RESOLVER_STRATEGY_CLASS:
                  latest_artifact_strategy.LatestArtifactStrategy,
              resolver.RESOLVER_CONFIG: {
                  'desired_num_of_artifacts': 1
              }
          })
      self.assertTrue(execution_result.use_cached_results)
      self.assertEmpty(execution_result.input_dict)
      self.assertDictEqual(
          execution_result.exec_properties, {
              resolver.RESOLVER_STRATEGY_CLASS:
                  latest_artifact_strategy.LatestArtifactStrategy,
              resolver.RESOLVER_CONFIG: {
                  'desired_num_of_artifacts': 1
              }
          })
      self.assertEqual(
          execution_result.output_dict[self.source_channel_key][0].uri,
          existing_artifact.uri)
      # TODO(b/148828122): Remove this after b/148828122 resolved.
      self.assertEqual(output_dict[self.source_channel_key].get()[0].uri,
                       existing_artifact.uri)

  def testResolveArtifactFailIncompleteResult(self):
    with metadata.Metadata(connection_config=self.connection_config) as m:
      driver = resolver._ResolverDriver(metadata_handler=m)
      driver.pre_execution(
          component_info=self.component_info,
          pipeline_info=self.pipeline_info,
          driver_args=self.driver_args,
          input_dict=self.source_channels,
          output_dict=self.source_channels.copy(),
          exec_properties={
              resolver.RESOLVER_STRATEGY_CLASS:
                  latest_artifact_strategy.LatestArtifactStrategy,
              resolver.RESOLVER_CONFIG: {}
          })


if __name__ == '__main__':
  tf.test.main()
