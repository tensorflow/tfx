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
"""Tests for tfx.components.statistics_gen.stats_artifact_utils."""

import os

import tensorflow as tf
from tfx.components.statistics_gen import stats_artifact_utils
from tfx.types import artifact_utils
from tfx.types import standard_artifacts


class StatsArtifactUtilsTest(tf.test.TestCase):

  def testLoadsStatistics(self):
    source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')

    stats_artifact = standard_artifacts.ExampleStatistics()
    stats_artifact.uri = os.path.join(source_data_dir, 'statistics_gen')
    stats_artifact.split_names = artifact_utils.encode_split_names(
        ['train', 'eval', 'test'])

    self.assertGreater(
        stats_artifact_utils.load_statistics(
            stats_artifact, 'train').proto().datasets[0].num_examples, 0)
    with self.assertRaisesRegex(
        ValueError,
        'Split does not exist over all example artifacts: not_a_split'):
      stats_artifact_utils.load_statistics(stats_artifact, 'not_a_split')


if __name__ == '__main__':
  tf.test.main()
