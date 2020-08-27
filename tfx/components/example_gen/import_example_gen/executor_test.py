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
"""Tests for tfx.components.example_gen.import_example_gen.executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import apache_beam as beam
from apache_beam.testing import util
import tensorflow as tf

from tfx.components.example_gen import utils
from tfx.components.example_gen.import_example_gen import executor
from tfx.proto import example_gen_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from google.protobuf import json_format


class ExecutorTest(tf.test.TestCase):

  def setUp(self):
    super(ExecutorTest, self).setUp()
    self._input_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'testdata',
        'external')

    # Create values in exec_properties
    self._input_config = json_format.MessageToJson(
        example_gen_pb2.Input(splits=[
            example_gen_pb2.Input.Split(name='tfrecord', pattern='tfrecord/*'),
        ]),
        preserving_proto_field_name=True)
    self._output_config = json_format.MessageToJson(
        example_gen_pb2.Output(
            split_config=example_gen_pb2.SplitConfig(splits=[
                example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=2),
                example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1)
            ])),
        preserving_proto_field_name=True)

  def testImportExample(self):
    with beam.Pipeline() as pipeline:
      examples = (
          pipeline
          | 'ToSerializedRecord' >> executor._ImportSerializedRecord(
              exec_properties={utils.INPUT_BASE_KEY: self._input_data_dir},
              split_pattern='tfrecord/*')
          | 'ToTFExample' >> beam.Map(tf.train.Example.FromString))

      def check_result(got):
        # We use Python assertion here to avoid Beam serialization error in
        # pickling tf.test.TestCase.
        assert (15000 == len(got)), 'Unexpected example count'
        assert (18 == len(got[0].features.feature)), 'Example not match'

      util.assert_that(examples, check_result)

  def _testDo(self, payload_format):
    exec_properties = {
        utils.INPUT_BASE_KEY: self._input_data_dir,
        utils.INPUT_CONFIG_KEY: self._input_config,
        utils.OUTPUT_CONFIG_KEY: self._output_config,
        utils.OUTPUT_DATA_FORMAT_KEY: payload_format,
    }

    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    # Create output dict.
    self.examples = standard_artifacts.Examples()
    self.examples.uri = output_data_dir
    output_dict = {utils.EXAMPLES_KEY: [self.examples]}

    # Run executor.
    import_example_gen = executor.Executor()
    import_example_gen.Do({}, output_dict, exec_properties)

    self.assertEqual(
        artifact_utils.encode_split_names(['train', 'eval']),
        self.examples.split_names)

    # Check import_example_gen outputs.
    train_output_file = os.path.join(self.examples.uri, 'train',
                                     'data_tfrecord-00000-of-00001.gz')
    eval_output_file = os.path.join(self.examples.uri, 'eval',
                                    'data_tfrecord-00000-of-00001.gz')

    # Check import_example_gen outputs.
    train_output_file = os.path.join(self.examples.uri, 'train',
                                     'data_tfrecord-00000-of-00001.gz')
    eval_output_file = os.path.join(self.examples.uri, 'eval',
                                    'data_tfrecord-00000-of-00001.gz')
    self.assertTrue(tf.io.gfile.exists(train_output_file))
    self.assertTrue(tf.io.gfile.exists(eval_output_file))
    self.assertGreater(
        tf.io.gfile.GFile(train_output_file).size(),
        tf.io.gfile.GFile(eval_output_file).size())

  def testDoWithExamples(self):
    self._testDo(example_gen_pb2.PayloadFormat.FORMAT_TF_EXAMPLE)
    self.assertEqual(
        example_gen_pb2.PayloadFormat.Name(
            example_gen_pb2.PayloadFormat.FORMAT_TF_EXAMPLE),
        self.examples.get_string_custom_property(
            utils.PAYLOAD_FORMAT_PROPERTY_NAME))

  def testDoWithProto(self):
    self._testDo(example_gen_pb2.PayloadFormat.FORMAT_PROTO)
    self.assertEqual(
        example_gen_pb2.PayloadFormat.Name(
            example_gen_pb2.PayloadFormat.FORMAT_PROTO),
        self.examples.get_string_custom_property(
            utils.PAYLOAD_FORMAT_PROPERTY_NAME))

  def testDoWithSequenceExamples(self):
    self._input_config = json_format.MessageToJson(
        example_gen_pb2.Input(splits=[
            example_gen_pb2.Input.Split(
                name='tfrecord_sequence', pattern='tfrecord_sequence/*'),
        ]),
        preserving_proto_field_name=True)

    self._testDo(example_gen_pb2.PayloadFormat.FORMAT_TF_SEQUENCE_EXAMPLE)
    self.assertEqual(
        example_gen_pb2.PayloadFormat.Name(
            example_gen_pb2.PayloadFormat.FORMAT_TF_SEQUENCE_EXAMPLE),
        self.examples.get_string_custom_property(
            utils.PAYLOAD_FORMAT_PROPERTY_NAME))


if __name__ == '__main__':
  tf.test.main()
