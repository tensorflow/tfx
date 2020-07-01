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
import apache_beam as beam
import tensorflow as tf
from google.protobuf import json_format
from tfx.components.example_gen import base_example_gen_executor
from tfx.components.example_gen import utils
from tfx.proto import example_gen_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts


@beam.ptransform_fn
def _TestInputSourceToExamplePTransform(pipeline, exec_properties,
                                        split_pattern):
  mock_examples = []
  size = 0
  if split_pattern == 'single/*':
    size = 6000
  elif split_pattern == 'train/*':
    size = 4000
  elif split_pattern == 'eval/*':
    size = 2000
  assert size != 0
  has_empty = exec_properties.get('has_empty', True)
  for i in range(size):
    feature = {}
    feature['i'] = tf.train.Feature(
    ) if i % 10 == 0 and has_empty else tf.train.Feature(
        int64_list=tf.train.Int64List(value=[i]))
    feature['f'] = tf.train.Feature(
    ) if i % 10 == 0 and has_empty else tf.train.Feature(
        float_list=tf.train.FloatList(value=[float(i)]))
    feature['s'] = tf.train.Feature(
    ) if i % 10 == 0 and has_empty else tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(str(i))]))
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    mock_examples.append(example_proto)
  result = pipeline | beam.Create(mock_examples)

  if exec_properties.get('format_proto', False):
    result |= beam.Map(lambda x: x.SerializeToString(deterministic=True))

  return result


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
    self._output_dict = {utils.EXAMPLES_KEY: [examples]}

    self._train_output_file = os.path.join(examples.uri, 'train',
                                           'data_tfrecord-00000-of-00001.gz')
    self._eval_output_file = os.path.join(examples.uri, 'eval',
                                          'data_tfrecord-00000-of-00001.gz')

    # Create exec proterties for output splits.
    self._exec_properties = {
        utils.INPUT_CONFIG_KEY:
            json_format.MessageToJson(
                example_gen_pb2.Input(splits=[
                    example_gen_pb2.Input.Split(
                        name='single', pattern='single/*'),
                ]),
                preserving_proto_field_name=True)
    }

  def testDoInputSplit(self):
    # Create exec proterties.
    exec_properties = {
        utils.INPUT_CONFIG_KEY:
            json_format.MessageToJson(
                example_gen_pb2.Input(splits=[
                    example_gen_pb2.Input.Split(
                        name='train', pattern='train/*'),
                    example_gen_pb2.Input.Split(name='eval', pattern='eval/*')
                ]),
                preserving_proto_field_name=True),
        utils.OUTPUT_CONFIG_KEY:
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
    # Add output config to exec proterties.
    self._exec_properties[utils.OUTPUT_CONFIG_KEY] = json_format.MessageToJson(
        example_gen_pb2.Output(
            split_config=example_gen_pb2.SplitConfig(splits=[
                example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=2),
                example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1)
            ])))

    # Run executor.
    example_gen = TestExampleGenExecutor()
    example_gen.Do({}, self._output_dict, self._exec_properties)

    # Check example gen outputs.
    self.assertTrue(tf.io.gfile.exists(self._train_output_file))
    self.assertTrue(tf.io.gfile.exists(self._eval_output_file))

    # Output split ratio: train:eval=2:1.
    self.assertGreater(
        tf.io.gfile.GFile(self._train_output_file).size(),
        tf.io.gfile.GFile(self._eval_output_file).size())

  def testDoOutputSplitWithProto(self):
    # Add output config to exec proterties.
    self._exec_properties[utils.OUTPUT_CONFIG_KEY] = json_format.MessageToJson(
        example_gen_pb2.Output(
            split_config=example_gen_pb2.SplitConfig(splits=[
                example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=2),
                example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1)
            ])))
    self._exec_properties['format_proto'] = True

    # Run executor.
    example_gen = TestExampleGenExecutor()
    example_gen.Do({}, self._output_dict, self._exec_properties)

    # Check example gen outputs.
    self.assertTrue(tf.io.gfile.exists(self._train_output_file))
    self.assertTrue(tf.io.gfile.exists(self._eval_output_file))

    # Output split ratio: train:eval=2:1.
    self.assertGreater(
        tf.io.gfile.GFile(self._train_output_file).size(),
        tf.io.gfile.GFile(self._eval_output_file).size())

  def testFeatureBasedPartition(self):
    # Add output config to exec proterties.
    self._exec_properties[utils.OUTPUT_CONFIG_KEY] = json_format.MessageToJson(
        example_gen_pb2.Output(
            split_config=example_gen_pb2.SplitConfig(
                splits=[
                    example_gen_pb2.SplitConfig.Split(
                        name='train', hash_buckets=2),
                    example_gen_pb2.SplitConfig.Split(
                        name='eval', hash_buckets=1)
                ],
                partition_feature_name='i')))
    self._exec_properties['has_empty'] = False

    # Run executor.
    example_gen = TestExampleGenExecutor()
    example_gen.Do({}, self._output_dict, self._exec_properties)

    # Check example gen outputs.
    self.assertTrue(tf.io.gfile.exists(self._train_output_file))
    self.assertTrue(tf.io.gfile.exists(self._eval_output_file))

    # Output split ratio: train:eval=2:1.
    self.assertGreater(
        tf.io.gfile.GFile(self._train_output_file).size(),
        tf.io.gfile.GFile(self._eval_output_file).size())

  def testInvalidFeatureName(self):
    # Add output config to exec proterties.
    self._exec_properties[utils.OUTPUT_CONFIG_KEY] = json_format.MessageToJson(
        example_gen_pb2.Output(
            split_config=example_gen_pb2.SplitConfig(
                splits=[
                    example_gen_pb2.SplitConfig.Split(
                        name='train', hash_buckets=2),
                    example_gen_pb2.SplitConfig.Split(
                        name='eval', hash_buckets=1)
                ],
                partition_feature_name='invalid')))

    # Run executor.
    example_gen = TestExampleGenExecutor()
    with self.assertRaisesRegexp(RuntimeError,
                                 'Feature name `.*` does not exist.'):
      example_gen.Do({}, self._output_dict, self._exec_properties)

  def testEmptyFeature(self):
    # Add output config to exec proterties.
    self._exec_properties[utils.OUTPUT_CONFIG_KEY] = json_format.MessageToJson(
        example_gen_pb2.Output(
            split_config=example_gen_pb2.SplitConfig(
                splits=[
                    example_gen_pb2.SplitConfig.Split(
                        name='train', hash_buckets=2),
                    example_gen_pb2.SplitConfig.Split(
                        name='eval', hash_buckets=1)
                ],
                partition_feature_name='i')))

    # Run executor.
    example_gen = TestExampleGenExecutor()
    with self.assertRaisesRegexp(
        RuntimeError, 'Partition feature does not contain any value.'):
      example_gen.Do({}, self._output_dict, self._exec_properties)

  def testInvalidFloatListFeature(self):
    # Add output config to exec proterties.
    self._exec_properties[utils.OUTPUT_CONFIG_KEY] = json_format.MessageToJson(
        example_gen_pb2.Output(
            split_config=example_gen_pb2.SplitConfig(
                splits=[
                    example_gen_pb2.SplitConfig.Split(
                        name='train', hash_buckets=2),
                    example_gen_pb2.SplitConfig.Split(
                        name='eval', hash_buckets=1)
                ],
                partition_feature_name='f')))
    self._exec_properties['has_empty'] = False

    # Run executor.
    example_gen = TestExampleGenExecutor()
    with self.assertRaisesRegexp(
        RuntimeError,
        'Only `bytes_list` and `int64_list` features are supported for partition.'
    ):
      example_gen.Do({}, self._output_dict, self._exec_properties)

  def testInvalidFeatureBasedPartitionWithProtos(self):
    # Add output config to exec proterties.
    self._exec_properties[utils.OUTPUT_CONFIG_KEY] = json_format.MessageToJson(
        example_gen_pb2.Output(
            split_config=example_gen_pb2.SplitConfig(
                splits=[
                    example_gen_pb2.SplitConfig.Split(
                        name='train', hash_buckets=2),
                    example_gen_pb2.SplitConfig.Split(
                        name='eval', hash_buckets=1)
                ],
                partition_feature_name='i')))
    self._exec_properties['has_empty'] = False
    self._exec_properties['format_proto'] = True

    # Run executor.
    example_gen = TestExampleGenExecutor()
    with self.assertRaisesRegexp(
        RuntimeError, 'Split by `partition_feature_name` is only supported for '
        'FORMAT_TF_EXAMPLE payload format.'):
      example_gen.Do({}, self._output_dict, self._exec_properties)


if __name__ == '__main__':
  tf.test.main()
