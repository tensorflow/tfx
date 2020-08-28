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
"""Tests for tfx.components.statistics_gen.component."""
import tensorflow as tf
import tensorflow_data_validation as tfdv
from tfx.components.statistics_gen import component
from tfx.types import artifact_utils
from tfx.types import channel_utils
from tfx.types import standard_artifacts


class ComponentTest(tf.test.TestCase):

  def testConstruct(self):
    examples = standard_artifacts.Examples()
    examples.split_names = artifact_utils.encode_split_names(['train', 'eval'])
    exclude_splits = ['eval']
    statistics_gen = component.StatisticsGen(
        examples=channel_utils.as_channel([examples]),
        exclude_splits=exclude_splits)
    self.assertEqual(standard_artifacts.ExampleStatistics.TYPE_NAME,
                     statistics_gen.outputs['statistics'].type_name)
    self.assertEqual(statistics_gen.spec.exec_properties['exclude_splits'],
                     '["eval"]')

  def testConstructWithSchemaAndStatsOptions(self):
    examples = standard_artifacts.Examples()
    examples.split_names = artifact_utils.encode_split_names(['train', 'eval'])
    schema = standard_artifacts.Schema()
    stats_options = tfdv.StatsOptions(
        weight_feature='weight',
        generators=[  # generators should be dropped
            tfdv.LiftStatsGenerator(
                schema=None,
                y_path=tfdv.FeaturePath(['label']),
                x_paths=[tfdv.FeaturePath(['feature'])])
        ])
    statistics_gen = component.StatisticsGen(
        examples=channel_utils.as_channel([examples]),
        schema=channel_utils.as_channel([schema]),
        stats_options=stats_options)
    self.assertEqual(standard_artifacts.ExampleStatistics.TYPE_NAME,
                     statistics_gen.outputs['statistics'].type_name)


if __name__ == '__main__':
  tf.test.main()
