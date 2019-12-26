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
"""Tests for tfx.components.example_gen.base_example_gen_executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import apache_beam as beam
import tensorflow as tf
from google.protobuf import json_format
from tfx.components.example_gen import base_example_gen_executor
from tfx.proto import example_gen_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts


@beam.ptransform_fn
def _TestInputSourceToExamplePTransform(
    pipeline,
    input_dict,  # pylint: disable=unused-argument
    exec_properties,  # pylint: disable=unused-argument
    split_pattern):
  mock_examples = []
  size = 0
  if split_pattern == 'single/*':
    size = 30000
  elif split_pattern == 'train/*':
    size = 20000
  elif split_pattern == 'eval/*':
    size = 10000
  assert size != 0
  for i in range(size):
    feature = {}
    feature['i'] = tf.train.Feature() if random.randrange(
        10) == 0 else tf.train.Feature(
            int64_list=tf.train.Int64List(value=[i]))
    feature['f'] = tf.train.Feature() if random.randrange(
        10) == 0 else tf.train.Feature(
            float_list=tf.train.FloatList(value=[float(i)]))
    feature['s'] = tf.train.Feature() if random.randrange(
        10) == 0 else tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(str(i))]))
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    mock_examples.append(example_proto)
  return pipeline | beam.Create(mock_examples)


class TestExampleGenExecutor(base_example_gen_executor.BaseExampleGenExecutor):

  def GetInputSourceToExamplePTransform(self):
    return _TestInputSourceToExamplePTransform


class BaseExampleGenExecutorTest(tf.test.TestCase):

  def setUp(self):
    super(BaseExampleGenExecutorTest, self).setUp()
    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    # Create output dict.
    examples = standard_artifacts.Examples()
    examples.uri = output_data_dir
    examples.split_names = artifact_utils.encode_split_names(['train', 'eval'])
    self._output_dict = {'examples': [examples]}

    self._train_output_file = os.path.join(examples.uri, 'train',
                                           'data_tfrecord-00000-of-00001.gz')
    self._eval_output_file = os.path.join(examples.uri, 'eval',
                                          'data_tfrecord-00000-of-00001.gz')

  def testDoInputSplit(self):
    # Create exec proterties.
    exec_properties = {
        'input_config':
            json_format.MessageToJson(
                example_gen_pb2.Input(splits=[
                    example_gen_pb2.Input.Split(
                        name='train', pattern='train/*'),
                    example_gen_pb2.Input.Split(name='eval', pattern='eval/*')
                ]),
                preserving_proto_field_name=True),
        'output_config':
            json_format.MessageToJson(
                example_gen_pb2.Output(), preserving_proto_field_name=True)
    }

    # Run executor.
    example_gen = TestExampleGenExecutor()
    example_gen.Do({}, self._output_dict, exec_properties)

    # Check example gen outputs.
    self.assertTrue(tf.io.gfile.exists(self._train_output_file))
    self.assertTrue(tf.io.gfile.exists(self._eval_output_file))
    # Input train split is bigger than eval split.
    self.assertGreater(
        tf.io.gfile.GFile(self._train_output_file).size(),
        tf.io.gfile.GFile(self._eval_output_file).size())

  def testDoOutputSplit(self):
    # Create exec proterties.
    exec_properties = {
        'input_config':
            json_format.MessageToJson(
                example_gen_pb2.Input(splits=[
                    example_gen_pb2.Input.Split(
                        name='single', pattern='single/*'),
                ]),
                preserving_proto_field_name=True),
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
    example_gen = TestExampleGenExecutor()
    example_gen.Do({}, self._output_dict, exec_properties)

    # Check example gen outputs.
    self.assertTrue(tf.io.gfile.exists(self._train_output_file))
    self.assertTrue(tf.io.gfile.exists(self._eval_output_file))
    # Output split ratio: train:eval=2:1.
    self.assertGreater(
        tf.io.gfile.GFile(self._train_output_file).size(),
        tf.io.gfile.GFile(self._eval_output_file).size())


if __name__ == '__main__':
  tf.test.main()
