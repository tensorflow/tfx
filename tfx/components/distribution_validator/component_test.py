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
"""Tests for tfx.distribution_validator.component."""

import tensorflow as tf
from tfx.components.distribution_validator import component
from tfx.proto import distribution_validator_pb2
from tfx.types import artifact_utils
from tfx.types import channel_utils
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs

from google.protobuf import text_format


class DistributionValidatorTest(tf.test.TestCase):

  def testConstruct(self):
    statistics_artifact1 = standard_artifacts.ExampleStatistics()
    statistics_artifact1.split_names = artifact_utils.encode_split_names(
        ['train', 'eval'])
    statistics_artifact2 = standard_artifacts.ExampleStatistics()
    statistics_artifact2.split_names = artifact_utils.encode_split_names(
        ['train', 'eval'])
    include_splits = [('eval', 'train')]
    config = text_format.Parse(
        """
      default_slice_config {
        num_examples_comparator {
          min_fraction_threshold: 1.0
        }
      }
    """, distribution_validator_pb2.DistributionValidatorConfig())
    distribution_validator = component.DistributionValidator(
        statistics=channel_utils.as_channel([statistics_artifact1]),
        baseline_statistics=channel_utils.as_channel([statistics_artifact2]),
        config=config,
        include_split_pairs=include_splits)
    self.assertEqual(
        standard_artifacts.ExampleAnomalies.TYPE_NAME,
        distribution_validator.outputs[
            standard_component_specs.ANOMALIES_KEY].type_name)
    self.assertEqual(
        distribution_validator.spec.exec_properties[
            standard_component_specs.INCLUDE_SPLIT_PAIRS_KEY],
        '[["eval", "train"]]')
    restored_config = distribution_validator.exec_properties[
        standard_component_specs.DISTRIBUTION_VALIDATOR_CONFIG_KEY]
    self.assertEqual(config, restored_config)


if __name__ == '__main__':
  tf.test.main()
