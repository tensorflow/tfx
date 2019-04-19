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
from tfx.components.example_gen import base_example_gen_executor
from tfx.proto import example_gen_pb2
from tfx.utils import types
from google.protobuf import json_format


@beam.ptransform_fn
def _TestInputSourceToExamplePTransform(
    pipeline,
    input_dict,  # pylint: disable=unused-argument
    exec_properties):  # pylint: disable=unused-argument
  mock_examples = []
  for i in range(10000):
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

  def test_do(self):
    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    # Create output dict.
    train_examples = types.TfxType(type_name='ExamplesPath', split='train')
    train_examples.uri = os.path.join(output_data_dir, 'train')
    eval_examples = types.TfxType(type_name='ExamplesPath', split='eval')
    eval_examples.uri = os.path.join(output_data_dir, 'eval')
    output_dict = {'examples': [train_examples, eval_examples]}

    # Create exec proterties.
    exec_properties = {
        'output':
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
    example_gen.Do({}, output_dict, exec_properties)

    # Check example gen outputs.
    train_output_file = os.path.join(train_examples.uri,
                                     'data_tfrecord-00000-of-00001.gz')
    eval_output_file = os.path.join(eval_examples.uri,
                                    'data_tfrecord-00000-of-00001.gz')
    self.assertTrue(tf.gfile.Exists(train_output_file))
    self.assertTrue(tf.gfile.Exists(eval_output_file))
    self.assertGreater(
        tf.gfile.GFile(train_output_file).size(),
        tf.gfile.GFile(eval_output_file).size())


if __name__ == '__main__':
  tf.test.main()
