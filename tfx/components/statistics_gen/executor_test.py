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
import tensorflow as tf
import tensorflow_data_validation as tfdv

from tfx.components.statistics_gen import executor
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.utils import json_utils
from tensorflow_metadata.proto.v0 import schema_pb2


# TODO(b/133421802): Investigate why tensorflow.TestCase could cause a crash
# when used with tfdv.
class ExecutorTest(absltest.TestCase):

  def get_temp_dir(self):
    return tempfile.mkdtemp()

  def _validate_stats_output(self, stats_path):
    self.assertTrue(tf.io.gfile.exists(stats_path))
    stats = tfdv.load_statistics(stats_path)
    self.assertLen(stats.datasets, 1)
    data_set = stats.datasets[0]
    self.assertGreater(data_set.num_examples, 0)
    self.assertNotEmpty(data_set.features)
    # TODO(b/126245422): verify content of generated stats after we have stable
    # test data set.

  def testDo(self):
    source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')
    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    tf.io.gfile.makedirs(output_data_dir)

    # Create input dict.
    examples = standard_artifacts.Examples()
    examples.uri = os.path.join(source_data_dir, 'csv_example_gen')
    examples.split_names = artifact_utils.encode_split_names(
        ['train', 'eval', 'test'])

    input_dict = {
        executor.EXAMPLES_KEY: [examples],
    }

    exec_properties = {
        # List needs to be serialized before being passed into Do function.
        executor.EXCLUDE_SPLITS_KEY:
            json_utils.dumps(['test']),
    }

    # Create output dict.
    stats = standard_artifacts.ExampleStatistics()
    stats.uri = output_data_dir
    output_dict = {
        executor.STATISTICS_KEY: [stats],
    }

    # Run executor.
    stats_gen_executor = executor.Executor()
    stats_gen_executor.Do(input_dict, output_dict, exec_properties)

    self.assertEqual(
        artifact_utils.encode_split_names(['train', 'eval']), stats.split_names)

    # Check statistics_gen outputs.
    self._validate_stats_output(
        os.path.join(stats.uri, 'train', 'stats_tfrecord'))
    self._validate_stats_output(
        os.path.join(stats.uri, 'eval', 'stats_tfrecord'))

    # Assert 'test' split is excluded.
    self.assertFalse(
        tf.io.gfile.exists(os.path.join(stats.uri, 'test', 'stats_tfrecord')))

  def testDoWithSchemaAndStatsOptions(self):
    source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')
    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    tf.io.gfile.makedirs(output_data_dir)

    # Create input dict.
    examples = standard_artifacts.Examples()
    examples.uri = os.path.join(source_data_dir, 'csv_example_gen')
    examples.split_names = artifact_utils.encode_split_names(['train', 'eval'])

    schema = standard_artifacts.Schema()
    schema.uri = os.path.join(source_data_dir, 'schema_gen')

    input_dict = {
        executor.EXAMPLES_KEY: [examples],
        executor.SCHEMA_KEY: [schema]
    }

    exec_properties = {
        executor.STATS_OPTIONS_JSON_KEY:
            tfdv.StatsOptions(label_feature='company').to_json(),
        executor.EXCLUDE_SPLITS_KEY:
            json_utils.dumps([])
    }

    # Create output dict.
    stats = standard_artifacts.ExampleStatistics()
    stats.uri = output_data_dir
    output_dict = {
        executor.STATISTICS_KEY: [stats],
    }

    # Run executor.
    stats_gen_executor = executor.Executor()
    stats_gen_executor.Do(input_dict, output_dict, exec_properties)

    # Check statistics_gen outputs.
    self._validate_stats_output(
        os.path.join(stats.uri, 'train', 'stats_tfrecord'))
    self._validate_stats_output(
        os.path.join(stats.uri, 'eval', 'stats_tfrecord'))

  def testDoWithTwoSchemas(self):
    source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')
    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    tf.io.gfile.makedirs(output_data_dir)

    # Create input dict.
    examples = standard_artifacts.Examples()
    examples.uri = os.path.join(source_data_dir, 'csv_example_gen')
    examples.split_names = artifact_utils.encode_split_names(['train', 'eval'])

    schema = standard_artifacts.Schema()
    schema.uri = os.path.join(source_data_dir, 'schema_gen')

    input_dict = {
        executor.EXAMPLES_KEY: [examples],
        executor.SCHEMA_KEY: [schema]
    }

    exec_properties = {
        executor.STATS_OPTIONS_JSON_KEY:
            tfdv.StatsOptions(
                label_feature='company', schema=schema_pb2.Schema()).to_json(),
        executor.EXCLUDE_SPLITS_KEY:
            json_utils.dumps([])
    }

    # Create output dict.
    stats = standard_artifacts.ExampleStatistics()
    stats.uri = output_data_dir
    output_dict = {
        executor.STATISTICS_KEY: [stats],
    }

    # Run executor.
    stats_gen_executor = executor.Executor()
    with self.assertRaises(ValueError):
      stats_gen_executor.Do(input_dict, output_dict, exec_properties)


if __name__ == '__main__':
  absltest.main()
