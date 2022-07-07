# Copyright 2022 Google LLC. All Rights Reserved.
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
"""Tests for tfx.components.util.example_statistics_utils."""

import tensorflow as tf
from tfx.components.util import example_statistics_utils
from tfx.types import artifact_utils
from tfx.types import standard_artifacts


class ExampleStatisticsUtilsTest(tf.test.TestCase):

  def testGetStatsPathEmpty(self):
    artifact = standard_artifacts.ExampleStatistics()
    artifact.uri = '/foo'
    path = example_statistics_utils.get_stats_path(artifact)
    self.assertEqual(path, '/foo')

  def testGetStatsPathWithSplit(self):
    artifact = standard_artifacts.ExampleStatistics()
    artifact.uri = '/foo'
    artifact.split_names = artifact_utils.encode_split_names(['bar', 'baz'])
    path = example_statistics_utils.get_stats_path(artifact, 'bar')
    self.assertEqual(path, '/foo/Split-bar')

  def testGetStatsPathMissingSplit(self):
    artifact = standard_artifacts.ExampleStatistics()
    artifact.uri = '/foo'
    with self.assertRaises(ValueError):
      example_statistics_utils.get_stats_path(artifact, 'bar')


if __name__ == '__main__':
  tf.test.main()
