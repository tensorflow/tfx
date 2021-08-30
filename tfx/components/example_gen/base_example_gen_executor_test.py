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

import os

import apache_beam as beam
import tensorflow as tf
from tfx.components.example_gen import base_example_gen_executor
from tfx.dsl.io import fileio
from tfx.proto import example_gen_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs
from tfx.utils import proto_utils


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

    if exec_properties.get('sequence_example', False):
      feature_list = {}
      feature_list['list'] = tf.train.FeatureList(feature=[feature['s']])
      example_proto = tf.train.SequenceExample(
          context=tf.train.Features(feature=feature),
          feature_lists=tf.train.FeatureLists(feature_list=feature_list))
    else:
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
    super().setUp()
    self._output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    # Create output dict.
    self._examples = standard_artifacts.Examples()
    self._examples.uri = self._output_data_dir
    self._output_dict = {
        standard_component_specs.EXAMPLES_KEY: [self._examples]
    }

    self._train_output_file = os.path.join(self._examples.uri, 'Split-train',
                                           'data_tfrecord-00000-of-00001.gz')
    self._eval_output_file = os.path.join(self._examples.uri, 'Split-eval',
                                          'data_tfrecord-00000-of-00001.gz')

    # Create exec proterties for output splits.
    self._exec_properties = {
        standard_component_specs.INPUT_CONFIG_KEY:
            proto_utils.proto_to_json(
                example_gen_pb2.Input(splits=[
                    example_gen_pb2.Input.Split(
                        name='single', pattern='single/*'),
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

  def _testDo(self):
    # Run executor.
    example_gen = TestExampleGenExecutor()
    example_gen.Do({}, self._output_dict, self._exec_properties)

    self.assertEqual(
        artifact_utils.encode_split_names(['train', 'eval']),
        self._examples.split_names)

    # Check example gen outputs.
    self.assertTrue(fileio.exists(self._train_output_file))
    self.assertTrue(fileio.exists(self._eval_output_file))

    # Output split ratio: train:eval=2:1.
    self.assertGreater(
        fileio.open(self._train_output_file).size(),
        fileio.open(self._eval_output_file).size())

  def testDoInputSplit(self):
    # Create exec proterties for input split.
    self._exec_properties = {
        standard_component_specs.INPUT_CONFIG_KEY:
            proto_utils.proto_to_json(
                example_gen_pb2.Input(splits=[
                    example_gen_pb2.Input.Split(
                        name='train', pattern='train/*'),
                    example_gen_pb2.Input.Split(name='eval', pattern='eval/*')
                ])),
        standard_component_specs.OUTPUT_CONFIG_KEY:
            proto_utils.proto_to_json(example_gen_pb2.Output())
    }

    self._testDo()

  def testDoOutputSplit(self):
    self._testDo()

  def testDoOutputSplitWithProto(self):
    # Update exec proterties.
    self._exec_properties['format_proto'] = True

    self._testDo()

  def testDoOutputSplitWithSequenceExample(self):
    # Update exec proterties.
    self._exec_properties['sequence_example'] = True

    self._testDo()

  def _testFeatureBasedPartition(self, partition_feature_name):
    self._exec_properties[
        standard_component_specs.OUTPUT_CONFIG_KEY] = proto_utils.proto_to_json(
            example_gen_pb2.Output(
                split_config=example_gen_pb2.SplitConfig(
                    splits=[
                        example_gen_pb2.SplitConfig.Split(
                            name='train', hash_buckets=2),
                        example_gen_pb2.SplitConfig.Split(
                            name='eval', hash_buckets=1)
                    ],
                    partition_feature_name=partition_feature_name)))

  def testFeatureBasedPartition(self):
    # Update exec proterties.
    self._testFeatureBasedPartition('i')
    self._exec_properties['has_empty'] = False

    self._testDo()

  def testFeatureBasedPartitionWithSequenceExample(self):
    # Update exec proterties.
    self._testFeatureBasedPartition('i')
    self._exec_properties['has_empty'] = False
    self._exec_properties['sequence_example'] = True

    self._testDo()

  def testInvalidFeatureName(self):
    # Update exec proterties.
    self._testFeatureBasedPartition('invalid')

    # Run executor.
    example_gen = TestExampleGenExecutor()
    with self.assertRaisesRegex(RuntimeError,
                                'Feature name `.*` does not exist.'):
      example_gen.Do({}, self._output_dict, self._exec_properties)

  def testEmptyFeature(self):
    # Update exec proterties.
    self._testFeatureBasedPartition('i')

    # Run executor.
    example_gen = TestExampleGenExecutor()
    with self.assertRaisesRegex(
        RuntimeError, 'Partition feature does not contain any value.'):
      example_gen.Do({}, self._output_dict, self._exec_properties)

  def testInvalidFloatListFeature(self):
    # Update exec proterties.
    self._testFeatureBasedPartition('f')
    self._exec_properties['has_empty'] = False

    # Run executor.
    example_gen = TestExampleGenExecutor()
    with self.assertRaisesRegex(
        RuntimeError,
        'Only `bytes_list` and `int64_list` features are supported for partition.'
    ):
      example_gen.Do({}, self._output_dict, self._exec_properties)

  def testInvalidFeatureBasedPartitionWithProtos(self):
    # Update exec proterties.
    self._testFeatureBasedPartition('i')
    self._exec_properties['has_empty'] = False
    self._exec_properties['format_proto'] = True

    # Run executor.
    example_gen = TestExampleGenExecutor()
    with self.assertRaisesRegex(
        RuntimeError, 'Split by `partition_feature_name` is only supported '
        'for FORMAT_TF_EXAMPLE and FORMAT_TF_SEQUENCE_EXAMPLE payload format.'):
      example_gen.Do({}, self._output_dict, self._exec_properties)


if __name__ == '__main__':
  tf.test.main()
