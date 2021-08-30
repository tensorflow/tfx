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
"""Tests for tfx.components.example_gen.csv_example_gen.executor."""

import os
from absl.testing import absltest

import apache_beam as beam
from apache_beam.testing import util
from tfx.components.example_gen.csv_example_gen import executor
from tfx.dsl.io import fileio
from tfx.proto import example_gen_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs
from tfx.utils import proto_utils


class ExecutorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._input_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'testdata',
        'external')

  def testCsvToExample(self):
    with beam.Pipeline() as pipeline:
      examples = (
          pipeline
          | 'ToTFExample' >> executor._CsvToExample(
              exec_properties={
                  standard_component_specs.INPUT_BASE_KEY: self._input_data_dir
              },
              split_pattern='csv/*'))

      def check_results(results):
        # We use Python assertion here to avoid Beam serialization error.
        assert (15000 == len(results)), 'Unexpected example count {}.'.format(
            len(results))
        assert (18 == len(results[0].features.feature)), 'Example not match.'

      util.assert_that(examples, check_results)

  def testCsvToExampleWithEmptyColumn(self):
    with beam.Pipeline() as pipeline:
      examples = (
          pipeline
          | 'ToTFExample' >> executor._CsvToExample(
              exec_properties={
                  standard_component_specs.INPUT_BASE_KEY: self._input_data_dir
              },
              split_pattern='csv_empty/*'))

      def check_results(results):
        # We use Python assertion here to avoid Beam serialization error.
        assert (3 == len(results)), 'Unexpected example count {}.'.format(
            len(results))
        for example in results:
          assert (example.features.feature['A'].HasField('int64_list')
                 ), 'Column A should be int64 type.'
          assert (not example.features.feature['B'].WhichOneof('kind')
                 ), 'Column B should be empty.'
          assert (example.features.feature['C'].HasField('bytes_list')
                 ), 'Column C should be byte type.'
          assert (example.features.feature['D'].HasField('float_list')
                 ), 'Column D should be float type.'

      util.assert_that(examples, check_results)

  def testCsvToExampleMultiLineString(self):
    with beam.Pipeline() as pipeline:
      examples = (
          pipeline
          | 'ToTFExample' >> executor._CsvToExample(
              exec_properties={
                  standard_component_specs.INPUT_BASE_KEY: self._input_data_dir
              },
              split_pattern='csv_multi_line_string/*'))

      def check_results(results):
        # We use Python assertion here to avoid Beam serialization error.
        assert (3 == len(results)), 'Unexpected example count: {}.'.format(
            len(results))
        instance = results[1]
        assert (instance.features.feature['B'].HasField('bytes_list')
               ), 'Column B should be bytes type. '
        value = instance.features.feature['B'].bytes_list.value
        assert (value ==
                [b'"2,\n"3",\n4\n5"']), 'Unexpected value: {}.'.format(value)

      util.assert_that(examples, check_results)

  def testDo(self):
    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.create_tempdir()),
        self._testMethodName)

    # Create output dict.
    examples = standard_artifacts.Examples()
    examples.uri = output_data_dir
    output_dict = {standard_component_specs.EXAMPLES_KEY: [examples]}

    # Create exec proterties.
    exec_properties = {
        standard_component_specs.INPUT_BASE_KEY:
            self._input_data_dir,
        standard_component_specs.INPUT_CONFIG_KEY:
            proto_utils.proto_to_json(
                example_gen_pb2.Input(splits=[
                    example_gen_pb2.Input.Split(name='csv', pattern='csv/*'),
                ])),
        standard_component_specs.OUTPUT_CONFIG_KEY:
            proto_utils.proto_to_json(
                example_gen_pb2.Output(
                    split_config=example_gen_pb2.SplitConfig(splits=[
                        example_gen_pb2.SplitConfig.Split(
                            name='train', hash_buckets=2),
                        example_gen_pb2.SplitConfig.Split(
                            name='eval', hash_buckets=1)
                    ])))
    }

    # Run executor.
    csv_example_gen = executor.Executor()
    csv_example_gen.Do({}, output_dict, exec_properties)

    self.assertEqual(
        artifact_utils.encode_split_names(['train', 'eval']),
        examples.split_names)

    # Check CSV example gen outputs.
    train_output_file = os.path.join(examples.uri, 'Split-train',
                                     'data_tfrecord-00000-of-00001.gz')
    eval_output_file = os.path.join(examples.uri, 'Split-eval',
                                    'data_tfrecord-00000-of-00001.gz')
    self.assertTrue(fileio.exists(train_output_file))
    self.assertTrue(fileio.exists(eval_output_file))
    self.assertGreater(
        fileio.open(train_output_file).size(),
        fileio.open(eval_output_file).size())


if __name__ == '__main__':
  absltest.main()
