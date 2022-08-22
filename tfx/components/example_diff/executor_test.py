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
"""Tests for tfx.components.example_diff.executor."""
import os
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow_data_validation as tfdv
from tensorflow_data_validation.skew import feature_skew_detector
from tfx.components.example_diff import executor
from tfx.dsl.io import fileio
from tfx.proto import example_diff_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs
from tfx.utils import json_utils

from google.protobuf import text_format

_EXECUTOR_TEST_PARAMS = [{
    'testcase_name': 'no_sharded_output',
    'sharded_output': False
}]
if tfdv.default_sharded_output_supported():
  _EXECUTOR_TEST_PARAMS.append({
      'testcase_name': 'yes_sharded_output',
      'sharded_output': True
  })

_ALL_FEATURE_NAMES = [
    'company', 'dropoff_census_tract', 'dropoff_community_area',
    'dropoff_latitude', 'dropoff_longitude', 'fare', 'payment_type',
    'pickup_census_tract', 'pickup_community_area', 'pickup_latitude',
    'pickup_longitude', 'tips', 'trip_miles', 'trip_seconds', 'trip_start_day',
    'trip_start_hour', 'trip_start_month', 'trip_start_timestamp'
]


class ExecutorTest(parameterized.TestCase):

  def get_temp_dir(self):
    return tempfile.mkdtemp()

  def _validate_skew_results(self, diff_path, expect_skew):
    # TODO(b/227361696): Validate contents and not just presence.
    self.assertNotEmpty(fileio.glob(diff_path + '*-of-*'))
    count = 0
    for _ in feature_skew_detector.skew_results_iterator(diff_path):
      count += 1
    if not expect_skew:
      self.assertEqual(count, 0)
    else:
      self.assertGreater(count, 0)

  def _validate_skew_pairs(self, diff_path, expect_skew):
    self.assertNotEmpty(fileio.glob(diff_path + '*-of-*'))
    count = 0
    for _ in feature_skew_detector.skew_pair_iterator(diff_path):
      count += 1
    if not expect_skew:
      self.assertEqual(count, 0)
    else:
      self.assertGreater(count, 0)

  @parameterized.named_parameters(
      # The identifier defined here is not actually expected to be unique, so
      # that we get output, but the rate of collision should be low enough to
      # not blow up the pipeline.
      {
          'testcase_name':
              'explicit_split_pairs_skew',
          'split_pairs': [('train', 'eval')],
          'expected_split_pair_names': ['train_eval'],
          'expect_skew':
              True,
          'identifiers': [
              'trip_start_month', 'trip_start_hour', 'trip_start_day', 'company'
          ]
      },
      {
          'testcase_name': 'explicit_split_pairs_noskew',
          'split_pairs': [('train', 'train')],
          'expected_split_pair_names': ['train_train'],
          'expect_skew': False,
          'identifiers': _ALL_FEATURE_NAMES,
      },
      {
          'testcase_name': 'implicit_split_pairs',
          'split_pairs': None,
          'expected_split_pair_names':
              ['train_train', 'train_eval', 'eval_train', 'eval_eval'],
          'expect_skew': False,
          'identifiers': _ALL_FEATURE_NAMES,
      })
  def testDo(self, split_pairs, expected_split_pair_names, expect_skew,
             identifiers):
    source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')
    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    fileio.makedirs(output_data_dir)

    # The identifier defined here is not actually expected to be unique, so that
    # we get output, but the rate of collision should be low enough to not blow
    # up the pipeline.
    config = text_format.Parse(
        """
      paired_example_skew: {
        identifier_features: 'company'
        skew_sample_size: 10
        allow_duplicate_identifiers: true
      }
    """, example_diff_pb2.ExampleDiffConfig())
    for identifier in identifiers:
      config.paired_example_skew.identifier_features.append(identifier)

    # Create input dict.
    examples = standard_artifacts.Examples()
    examples.uri = os.path.join(source_data_dir, 'csv_example_gen')
    examples.split_names = artifact_utils.encode_split_names(['train', 'eval'])

    input_dict = {
        standard_component_specs.EXAMPLES_KEY: [examples],
        standard_component_specs.BASELINE_EXAMPLES_KEY: [examples],
    }

    exec_properties = {
        # List needs to be serialized before being passed into Do function.
        standard_component_specs.INCLUDE_SPLIT_PAIRS_KEY:
            json_utils.dumps(split_pairs),
        standard_component_specs.EXAMPLE_DIFF_CONFIG_KEY:
            config,
    }

    # Create output dict.
    example_diff = standard_artifacts.ExamplesDiff()
    example_diff.uri = output_data_dir
    output_dict = {
        standard_component_specs.EXAMPLE_DIFF_RESULT_KEY: [example_diff],
    }

    # Run executor.
    example_diff_executor = executor.Executor()
    example_diff_executor.Do(input_dict, output_dict, exec_properties)

    # See tensorflow_data_validation/skew/feature_skew_detector_test.py for
    # detailed examples of feature skew pipeline output.
    for split_pair_name in expected_split_pair_names:
      self._validate_skew_pairs(
          os.path.join(example_diff.uri, 'SplitPair-' + split_pair_name,
                       'sample_pairs'), expect_skew)
      self._validate_skew_results(
          os.path.join(example_diff.uri, 'SplitPair-' + split_pair_name,
                       'skew_stats'), expect_skew)
      # Validate that no additional skew pairs exist.
      all_outputs = fileio.glob(os.path.join(example_diff.uri, 'SplitPair-*'))
      for output in all_outputs:
        split_pair = output.split('SplitPair-')[1]
        self.assertIn(split_pair, expected_split_pair_names)


if __name__ == '__main__':
  absltest.main()
