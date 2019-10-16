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
from tfx.components.example_gen.custom_executors import parquet_executor
from tfx.proto import example_gen_pb2
from tfx.types import standard_artifacts
from google.protobuf import json_format


class ExecutorTest(tf.test.TestCase):

  def setUp(self):
    input_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'testdata')

    # Create input dict.
    input_base = standard_artifacts.ExternalArtifact()
    input_base.uri = os.path.join(input_data_dir, 'external')
    self._input_dict = {'input_base': [input_base]}

  def testParquetToExample(self):
    with beam.Pipeline() as pipeline:
      examples = (
          pipeline
          | 'ToTFExample' >> parquet_executor._ParquetToExample(
              input_dict=self._input_dict,
              exec_properties={},
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
    train_examples = standard_artifacts.Examples(split='train')
    train_examples.uri = os.path.join(output_data_dir, 'train')
    eval_examples = standard_artifacts.Examples(split='eval')
    eval_examples.uri = os.path.join(output_data_dir, 'eval')
    output_dict = {'examples': [train_examples, eval_examples]}

    # Create exec proterties.
    exec_properties = {
        'input_config':
            json_format.MessageToJson(
                example_gen_pb2.Input(splits=[
                    example_gen_pb2.Input.Split(
                        name='parquet', pattern='parquet/*'),
                ])),
        'output_config':
            json_format.MessageToJson(
                example_gen_pb2.Output(
                    split_config=example_gen_pb2.SplitConfig(splits=[
                        example_gen_pb2.SplitConfig.Split(
                            name='train', hash_buckets=2),
                        example_gen_pb2.SplitConfig.Split(
                            name='eval', hash_buckets=1)
                    ])))
    }

    # Run executor.
    parquet_example_gen = parquet_executor.Executor()
    parquet_example_gen.Do(self._input_dict, output_dict, exec_properties)

    # Check Parquet example gen outputs.
    train_output_file = os.path.join(train_examples.uri,
                                     'data_tfrecord-00000-of-00001.gz')
    eval_output_file = os.path.join(eval_examples.uri,
                                    'data_tfrecord-00000-of-00001.gz')
    self.assertTrue(tf.io.gfile.exists(train_output_file))
    self.assertTrue(tf.io.gfile.exists(eval_output_file))
    self.assertGreater(
        tf.io.gfile.GFile(train_output_file).size(),
        tf.io.gfile.GFile(eval_output_file).size())


if __name__ == '__main__':
  tf.test.main()
