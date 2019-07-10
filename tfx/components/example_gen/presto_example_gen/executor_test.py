# Copyright 2019 Google LLC. All Rights Reserved.
# Copyright 2019 Naver Corp. All Rights Reserved.
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
"""Tests for tfx.components.example_gen.presto_example_gen.executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import prestodb
import random
import apache_beam as beam
from apache_beam.testing import util
import mock
import tensorflow as tf
from tfx.components.example_gen.presto_example_gen import executor
from tfx.proto import example_gen_pb2
from tfx.utils import types
from google.protobuf import json_format


@beam.ptransform_fn
def _MockReadFromPresto(pipeline, query, db_cursor):  # pylint: disable=invalid-name, unused-argument
  mock_query_results = []
  for i in range(10000):
    mock_query_result = {
        'i': None if random.randrange(10) == 0 else i,
        'f': None if random.randrange(10) == 0 else float(i),
        's': None if random.randrange(10) == 0 else str(i)
    }
    mock_query_results.append(mock_query_result)
  return pipeline | beam.Create(mock_query_results)


@beam.ptransform_fn
def _MockReadFromPresto2(pipeline, query, db_cursor):  # pylint: disable=invalid-name, unused-argument
  rows = db_cursor.fetchall()
  desc = db_cursor.description
  mock_query_results = list()
  for row in rows:
     mock_query_results.append({desc[i][0]: row[i] for i in range(len(desc))})

  return pipeline | beam.Create(mock_query_results)


class ExecutorTest(tf.test.TestCase):

  def setUp(self):
    # Mock Presto result schema.
    self._schema = [
      ('i', 'integer', None, None, None, None, None),
      ('f', 'double', None, None, None, None, None),
      ('s', 'varchar', None, None, None, None, None)
    ]
    self._rows = [[1, 2.0, 'abc']]
    self._connection_info = {
      'host': 'host.test',
      'port': 65535,
      'user': 'test_user',
      'catalog': 'test_catalog',
      'schema': 'test_schema'
    }

  @mock.patch.multiple(
      executor,
      _ReadFromPresto=_MockReadFromPresto2,  # pylint: disable=invalid-name, unused-argument
  )

  @mock.patch.object(prestodb.dbapi, 'Connection')
  def testPrestoToExample(self, mock_connection):
    # Mock query result schema for _PrestoConverter.
    mock_connection.return_value.cursor.return_value.description = self._schema
    mock_connection.return_value.cursor.return_value.fetchone.return_value = self._rows[0]
    mock_connection.return_value.cursor.return_value.fetchall.return_value = self._rows

    with beam.Pipeline() as pipeline:
      examples = (
          pipeline | 'ToTFExample' >> executor._PrestoToExample(
              input_dict={},
              exec_properties={
                'connection_info': self._connection_info
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
      _ReadFromPresto=_MockReadFromPresto,  # pylint: disable=invalid-name, unused-argument
  )
  @mock.patch.object(prestodb.dbapi, 'Connection')
  def testDo(self, mock_connection):
    # Mock query result schema for _PrestoConverter.
    mock_connection.return_value.cursor.return_value.description = self._schema
    # mock_client.return_value.query.return_value.result.return_value.schema = self._schema

    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    # Create output dict.
    train_examples = types.TfxArtifact(type_name='ExamplesPath', split='train')
    train_examples.uri = os.path.join(output_data_dir, 'train')
    eval_examples = types.TfxArtifact(type_name='ExamplesPath', split='eval')
    eval_examples.uri = os.path.join(output_data_dir, 'eval')
    output_dict = {'examples': [train_examples, eval_examples]}

    # Create exe properties.
    exec_properties = {
        'input_config':
            json_format.MessageToJson(
                example_gen_pb2.Input(splits=[
                    example_gen_pb2.Input.Split(
                        name='presto', pattern='SELECT i, f, s FROM `fake`'),
                ])),
        'output_config':
            json_format.MessageToJson(
                example_gen_pb2.Output(
                    split_config=example_gen_pb2.SplitConfig(splits=[
                        example_gen_pb2.SplitConfig.Split(
                            name='train', hash_buckets=2),
                        example_gen_pb2.SplitConfig.Split(
                            name='eval', hash_buckets=1)
                    ]))),
        'connection_info': self._connection_info
    }

    # Run executor.
    presto_example_gen = executor.Executor()
    presto_example_gen.Do({}, output_dict, exec_properties)

    # Check BigQuery example gen outputs.
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
