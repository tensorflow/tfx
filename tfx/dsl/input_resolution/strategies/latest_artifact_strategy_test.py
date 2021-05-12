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
"""Test for LatestArtifactStrategy."""

import tensorflow as tf
from tfx import types
from tfx.dsl.input_resolution.strategies import latest_artifact_strategy
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.types import standard_artifacts
from tfx.utils import test_case_utils

from ml_metadata.proto import metadata_store_pb2


class LatestArtifactStrategyTest(test_case_utils.TfxTest):

  def setUp(self):
    super().setUp()
    self._connection_config = metadata_store_pb2.ConnectionConfig()
    self._connection_config.sqlite.SetInParent()
    self._metadata = self.enter_context(
        metadata.Metadata(connection_config=self._connection_config))
    self._store = self._metadata.store
    self._pipeline_info = data_types.PipelineInfo(
        pipeline_name='my_pipeline', pipeline_root='/tmp', run_id='my_run_id')
    self._component_info = data_types.ComponentInfo(
        component_type='a.b.c',
        component_id='my_component',
        pipeline_info=self._pipeline_info)

  def testStrategy(self):
    contexts = self._metadata.register_pipeline_contexts_if_not_exists(
        self._pipeline_info)
    artifact_one = standard_artifacts.Examples()
    artifact_one.uri = 'uri_one'
    self._metadata.publish_artifacts([artifact_one])
    artifact_two = standard_artifacts.Examples()
    artifact_two.uri = 'uri_two'
    self._metadata.register_execution(
        exec_properties={},
        pipeline_info=self._pipeline_info,
        component_info=self._component_info,
        contexts=contexts)
    self._metadata.publish_execution(
        component_info=self._component_info,
        output_artifacts={'key': [artifact_one, artifact_two]})
    expected_artifact = max(artifact_one, artifact_two, key=lambda a: a.id)

    strategy = latest_artifact_strategy.LatestArtifactStrategy()
    resolve_result = strategy.resolve(
        pipeline_info=self._pipeline_info,
        metadata_handler=self._metadata,
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
    ], [expected_artifact.uri])
    self.assertTrue(resolve_result.per_key_resolve_state['input'])

  def testStrategy_IrMode(self):
    artifact_one = standard_artifacts.Examples()
    artifact_one.uri = 'uri_one'
    artifact_one.id = 1
    artifact_two = standard_artifacts.Examples()
    artifact_two.uri = 'uri_two'
    artifact_one.id = 2

    expected_artifact = max(artifact_one, artifact_two, key=lambda a: a.id)

    strategy = latest_artifact_strategy.LatestArtifactStrategy()
    result = strategy.resolve_artifacts(
        self._store, {'input': [artifact_two, artifact_one]})
    self.assertIsNotNone(result)
    self.assertEqual([a.uri for a in result['input']],
                     [expected_artifact.uri])


if __name__ == '__main__':
  tf.test.main()
