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
"""Tests for tfx.components.statistics_gen.executor."""
import os
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow_data_validation as tfdv
from tfx.components.statistics_gen import executor
from tfx.dsl.io import fileio
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs
from tfx.utils import json_utils

from tensorflow_metadata.proto.v0 import schema_pb2


_EXECUTOR_TEST_PARAMS = [{
    'testcase_name': 'no_sharded_output',
    'sharded_output': False
}]
if tfdv.default_sharded_output_supported():
  _EXECUTOR_TEST_PARAMS.append({
      'testcase_name': 'yes_sharded_output',
      'sharded_output': True
  })


# TODO(b/133421802): Investigate why tensorflow.TestCase could cause a crash
# when used with tfdv.
class ExecutorTest(parameterized.TestCase):

  def get_temp_dir(self):
    return tempfile.mkdtemp()

  def _validate_stats(self, stats):
    self.assertLen(stats.datasets, 1)
    data_set = stats.datasets[0]
    self.assertGreater(data_set.num_examples, 0)
    self.assertNotEmpty(data_set.features)
    # TODO(b/126245422): verify content of generated stats after we have stable
    # test data set.

  def _validate_stats_output(self, stats_path):
    self.assertTrue(fileio.exists(stats_path))
    stats = tfdv.load_stats_binary(stats_path)
    self._validate_stats(stats)

  def _validate_sharded_stats_output(self, stats_prefix):
    stats = tfdv.load_sharded_statistics(stats_prefix).proto()
    self._validate_stats(stats)

  @parameterized.named_parameters(*_EXECUTOR_TEST_PARAMS)
  def testDo(self, sharded_output: bool):
    source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')
    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    fileio.makedirs(output_data_dir)

    # Create input dict.
    examples = standard_artifacts.Examples()
    examples.uri = os.path.join(source_data_dir, 'csv_example_gen')
    examples.split_names = artifact_utils.encode_split_names(
        ['train', 'eval', 'test'])

    input_dict = {
        standard_component_specs.EXAMPLES_KEY: [examples],
    }

    exec_properties = {
        # List needs to be serialized before being passed into Do function.
        standard_component_specs.EXCLUDE_SPLITS_KEY:
            json_utils.dumps(['test']),
        standard_component_specs.SHARDED_STATS_OUTPUT_KEY:
            sharded_output,
    }

    # Create output dict.
    stats = standard_artifacts.ExampleStatistics()
    stats.uri = output_data_dir
    output_dict = {
        standard_component_specs.STATISTICS_KEY: [stats],
    }

    # Run executor.
    stats_gen_executor = executor.Executor()
    stats_gen_executor.Do(input_dict, output_dict, exec_properties)

    self.assertEqual(
        artifact_utils.encode_split_names(['train', 'eval']), stats.split_names)

    # Check statistics_gen outputs.
    self._validate_stats_output(
        os.path.join(stats.uri, 'Split-train', 'FeatureStats.pb'))
    self._validate_stats_output(
        os.path.join(stats.uri, 'Split-eval', 'FeatureStats.pb'))
    if sharded_output:
      self._validate_sharded_stats_output(
          os.path.join(
              stats.uri, 'Split-train',
              'FeatureStats' + tfdv.default_sharded_output_suffix()))
      self._validate_sharded_stats_output(
          os.path.join(
              stats.uri, 'Split-eval',
              'FeatureStats' + tfdv.default_sharded_output_suffix()))
    else:
      # We want to verify that attempting to load sharded stats produces an
      # error.
      with self.assertRaisesRegex(ValueError, 'No input paths found.*'):
        self._validate_sharded_stats_output(
            os.path.join(
                stats.uri, 'Split-train',
                'FeatureStats' + tfdv.default_sharded_output_suffix()))
      with self.assertRaisesRegex(ValueError, 'No input paths found.*'):
        self._validate_sharded_stats_output(
            os.path.join(
                stats.uri, 'Split-eval',
                'FeatureStats' + tfdv.default_sharded_output_suffix()))

    # Assert 'test' split is excluded.
    self.assertFalse(
        fileio.exists(os.path.join(stats.uri, 'test', 'FeatureStats.pb')))

  def testDoWithSchemaAndStatsOptions(self):
    source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')
    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    fileio.makedirs(output_data_dir)

    # Create input dict.
    examples = standard_artifacts.Examples()
    examples.uri = os.path.join(source_data_dir, 'csv_example_gen')
    examples.split_names = artifact_utils.encode_split_names(['train', 'eval'])

    schema = standard_artifacts.Schema()
    schema.uri = os.path.join(source_data_dir, 'schema_gen')

    input_dict = {
        standard_component_specs.EXAMPLES_KEY: [examples],
        standard_component_specs.SCHEMA_KEY: [schema]
    }

    exec_properties = {
        standard_component_specs.STATS_OPTIONS_JSON_KEY:
            tfdv.StatsOptions(label_feature='company').to_json(),
        standard_component_specs.EXCLUDE_SPLITS_KEY:
            json_utils.dumps([])
    }

    # Create output dict.
    stats = standard_artifacts.ExampleStatistics()
    stats.uri = output_data_dir
    output_dict = {
        standard_component_specs.STATISTICS_KEY: [stats],
    }

    # Run executor.
    stats_gen_executor = executor.Executor()
    stats_gen_executor.Do(input_dict, output_dict, exec_properties)

    # Check statistics_gen outputs.
    self._validate_stats_output(
        os.path.join(stats.uri, 'Split-train', 'FeatureStats.pb'))
    self._validate_stats_output(
        os.path.join(stats.uri, 'Split-eval', 'FeatureStats.pb'))

  def testDoWithTwoSchemas(self):
    source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')
    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    fileio.makedirs(output_data_dir)

    # Create input dict.
    examples = standard_artifacts.Examples()
    examples.uri = os.path.join(source_data_dir, 'csv_example_gen')
    examples.split_names = artifact_utils.encode_split_names(['train', 'eval'])

    schema = standard_artifacts.Schema()
    schema.uri = os.path.join(source_data_dir, 'schema_gen')

    input_dict = {
        standard_component_specs.EXAMPLES_KEY: [examples],
        standard_component_specs.SCHEMA_KEY: [schema]
    }

    exec_properties = {
        standard_component_specs.STATS_OPTIONS_JSON_KEY:
            tfdv.StatsOptions(
                label_feature='company', schema=schema_pb2.Schema()).to_json(),
        standard_component_specs.EXCLUDE_SPLITS_KEY:
            json_utils.dumps([])
    }

    # Create output dict.
    stats = standard_artifacts.ExampleStatistics()
    stats.uri = output_data_dir
    output_dict = {
        standard_component_specs.STATISTICS_KEY: [stats],
    }

    # Run executor.
    stats_gen_executor = executor.Executor()
    with self.assertRaises(ValueError):
      stats_gen_executor.Do(input_dict, output_dict, exec_properties)

  def testNoInputSplits(self):
    source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')
    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    fileio.makedirs(output_data_dir)

    # Create input dict.
    examples = standard_artifacts.Examples()
    examples.uri = os.path.join(source_data_dir, 'csv_example_gen')
    examples.split_names = artifact_utils.encode_split_names([])

    input_dict = {
        standard_component_specs.EXAMPLES_KEY: [examples],
    }

    exec_properties = {
        standard_component_specs.EXCLUDE_SPLITS_KEY:
            json_utils.dumps([])
    }

    # Create output dict.
    stats = standard_artifacts.ExampleStatistics()
    stats.uri = output_data_dir
    output_dict = {
        standard_component_specs.STATISTICS_KEY: [stats],
    }

    # Run executor.
    stats_gen_executor = executor.Executor()
    with self.assertRaises(ValueError):
      stats_gen_executor.Do(input_dict, output_dict, exec_properties)


if __name__ == '__main__':
  absltest.main()
