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
"""Tests for tfx.components.example_gen.custom_executors.parquet_executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import apache_beam as beam
from apache_beam.testing import util
import tensorflow as tf
from google.protobuf import json_format
from tfx.components.example_gen import utils
from tfx.components.example_gen.custom_executors import parquet_executor
from tfx.proto import example_gen_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts


class ExecutorTest(tf.test.TestCase):

  def setUp(self):
    super(ExecutorTest, self).setUp()
    self._input_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'testdata',
        'external')

  def testParquetToExample(self):
    with beam.Pipeline() as pipeline:
      examples = (
          pipeline
          | 'ToTFExample' >> parquet_executor._ParquetToExample(
              exec_properties={utils.INPUT_BASE_KEY: self._input_data_dir},
              split_pattern='parquet/*'))

      def check_result(got):
        # We use Python assertion here to avoid Beam serialization error in
        # pickling tf.test.TestCase.
        assert (10000 == len(got)), 'Unexpected example count'
        assert (18 == len(got[0].features.feature)), 'Example not match'

      util.assert_that(examples, check_result)

  def testDo(self):
    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    # Create output dict.
    examples = standard_artifacts.Examples()
    examples.uri = output_data_dir
    output_dict = {utils.EXAMPLES_KEY: [examples]}

    # Create exec proterties.
    exec_properties = {
        utils.INPUT_BASE_KEY:
            self._input_data_dir,
        utils.INPUT_CONFIG_KEY:
            json_format.MessageToJson(
                example_gen_pb2.Input(splits=[
                    example_gen_pb2.Input.Split(
                        name='parquet', pattern='parquet/*'),
                ]),
                preserving_proto_field_name=True),
        utils.OUTPUT_CONFIG_KEY:
            json_format.MessageToJson(
                example_gen_pb2.Output(
                    split_config=example_gen_pb2.SplitConfig(splits=[
                        example_gen_pb2.SplitConfig.Split(
                            name='train', hash_buckets=2),
                        example_gen_pb2.SplitConfig.Split(
                            name='eval', hash_buckets=1)
                    ])),
                preserving_proto_field_name=True)
    }

    # Run executor.
    parquet_example_gen = parquet_executor.Executor()
    parquet_example_gen.Do({}, output_dict, exec_properties)

    self.assertEqual(
        artifact_utils.encode_split_names(['train', 'eval']),
        examples.split_names)

    # Check Parquet example gen outputs.
    train_output_file = os.path.join(examples.uri, 'train',
                                     'data_tfrecord-00000-of-00001.gz')
    eval_output_file = os.path.join(examples.uri, 'eval',
                                    'data_tfrecord-00000-of-00001.gz')
    self.assertTrue(tf.io.gfile.exists(train_output_file))
    self.assertTrue(tf.io.gfile.exists(eval_output_file))
    self.assertGreater(
        tf.io.gfile.GFile(train_output_file).size(),
        tf.io.gfile.GFile(eval_output_file).size())


if __name__ == '__main__':
  tf.test.main()
