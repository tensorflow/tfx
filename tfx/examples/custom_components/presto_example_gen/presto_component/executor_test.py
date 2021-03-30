# Lint as: python3
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
"""Tests for presto_component.executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
from unittest import mock

import apache_beam as beam
from apache_beam.testing import util
import prestodb
import tensorflow as tf
from tfx.dsl.io import fileio
from tfx.examples.custom_components.presto_example_gen.presto_component import executor
from tfx.examples.custom_components.presto_example_gen.proto import presto_config_pb2
from tfx.proto import example_gen_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.utils import proto_utils


class _MockReadPrestoDoFn(beam.DoFn):

  def __init__(self, client):
    pass

  def process(self, query):
    for i in range(10000):
      yield {('i', 'integer', None if random.randrange(10) == 0 else i),
             ('f', 'double', None if random.randrange(10) == 0 else float(i)),
             ('s', 'varchar', None if random.randrange(10) == 0 else str(i))}


class _MockReadPrestoDoFn2(beam.DoFn):

  def __init__(self, client):
    pass

  def process(self, query):
    yield {('i', 'integer', 1), ('f', 'double', 2.0), ('s', 'varchar', 'abc')}


def _mock_deserialize_conn_config(input_config):  # pylint: disable=invalid-name, unused-argument
  return prestodb.dbapi.connect('localhost')


class ExecutorTest(tf.test.TestCase):

  def testDeserializeConnConfig(self):
    conn_config = presto_config_pb2.PrestoConnConfig(
        host='presto.localhost', max_attempts=10)

    deseralized_conn = executor._deserialize_conn_config(conn_config)
    truth_conn = prestodb.dbapi.connect('presto.localhost', max_attempts=10)
    self.assertEqual(truth_conn.host, deseralized_conn.host)
    self.assertEqual(truth_conn.port,
                     deseralized_conn.port)  # test for default port value
    self.assertEqual(truth_conn.auth,
                     deseralized_conn.auth)  # test for default auth value
    self.assertEqual(truth_conn.max_attempts, deseralized_conn.max_attempts)

  @mock.patch.multiple(
      executor,
      _ReadPrestoDoFn=_MockReadPrestoDoFn2,
      _deserialize_conn_config=_mock_deserialize_conn_config,
  )
  def testPrestoToExample(self):
    with beam.Pipeline() as pipeline:
      examples = (
          pipeline | 'ToTFExample' >> executor._PrestoToExample(
              exec_properties={
                  'input_config':
                      proto_utils.proto_to_json(example_gen_pb2.Input()),
                  'custom_config':
                      proto_utils.proto_to_json(example_gen_pb2.CustomConfig())
              },
              split_pattern='SELECT i, f, s FROM `fake`'))

      feature = {}
      feature['i'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[1]))
      feature['f'] = tf.train.Feature(
          float_list=tf.train.FloatList(value=[2.0]))
      feature['s'] = tf.train.Feature(
          bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes('abc')]))
      example_proto = tf.train.Example(
          features=tf.train.Features(feature=feature))
      util.assert_that(examples, util.equal_to([example_proto]))

  @mock.patch.multiple(
      executor,
      _ReadPrestoDoFn=_MockReadPrestoDoFn,
      _deserialize_conn_config=_mock_deserialize_conn_config,
  )
  def testDo(self):
    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    # Create output dict.
    examples = standard_artifacts.Examples()
    examples.uri = output_data_dir
    output_dict = {'examples': [examples]}

    # Create exe properties.
    exec_properties = {
        'input_config':
            proto_utils.proto_to_json(
                example_gen_pb2.Input(splits=[
                    example_gen_pb2.Input.Split(
                        name='bq', pattern='SELECT i, f, s FROM `fake`'),
                ])),
        'custom_config':
            proto_utils.proto_to_json(example_gen_pb2.CustomConfig()),
        'output_config':
            proto_utils.proto_to_json(
                example_gen_pb2.Output(
                    split_config=example_gen_pb2.SplitConfig(splits=[
                        example_gen_pb2.SplitConfig.Split(
                            name='train', hash_buckets=2),
                        example_gen_pb2.SplitConfig.Split(
                            name='eval', hash_buckets=1)
                    ]))),
    }

    # Run executor.
    presto_example_gen = executor.Executor()
    presto_example_gen.Do({}, output_dict, exec_properties)

    self.assertEqual(
        artifact_utils.encode_split_names(['train', 'eval']),
        examples.split_names)

    # Check Presto example gen outputs.
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
  tf.test.main()
