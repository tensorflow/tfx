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
from tfx.dsl.experimental import latest_artifacts_resolver
from tfx.orchestration import metadata
from tfx.types import standard_artifacts


class LatestArtifactsResolverTest(tf.test.TestCase):

  def setUp(self):
    super(LatestArtifactsResolverTest, self).setUp()
    self._connection_config = metadata_store_pb2.ConnectionConfig()
    self._connection_config.sqlite.SetInParent()

  def testArtifact(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      # Publish multiple artifacts.
      artifact_one = standard_artifacts.Examples()
      artifact_one.uri = 'uri_one'
      m.publish_artifacts([artifact_one])
      artifact_two = standard_artifacts.Examples()
      artifact_two.uri = 'uri_two'
      m.publish_artifacts([artifact_two])

      resolver = latest_artifacts_resolver.LatestArtifactsResolver()
      resolve_result = resolver.resolve(
          m, {'input': types.Channel(type=artifact_one.type)})

      self.assertTrue(resolve_result.has_complete_result)
      self.assertEqual([
          artifact.uri
          for artifact in resolve_result.per_key_resolve_result['input']
      ], ['uri_two'])
      self.assertTrue(resolve_result.per_key_resolve_state['input'])


if __name__ == '__main__':
  tf.test.main()
