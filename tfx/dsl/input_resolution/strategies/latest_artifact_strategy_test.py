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
from tfx.dsl.input_resolution.strategies import latest_artifact_strategy
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

  def testStrategy(self):
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
