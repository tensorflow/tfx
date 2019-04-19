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
"""Tests for tfx.components.example_gen.big_query_example_gen.executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import apache_beam as beam
from apache_beam.testing import util
import mock
import tensorflow as tf
from google.cloud import bigquery
from tfx.components.example_gen.big_query_example_gen import executor
from tfx.proto import example_gen_pb2
from tfx.utils import types
from google.protobuf import json_format


@beam.ptransform_fn
def _MockReadFromBigQuery(pipeline, query):  # pylint: disable=invalid-name, unused-argument
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
def _MockReadFromBigQuery2(pipeline, query):  # pylint: disable=invalid-name, unused-argument
  mock_query_results = [{
      'i': 1,
      'f': 2.0,
      's': 'abc',
  }]
  return pipeline | beam.Create(mock_query_results)


class ExecutorTest(tf.test.TestCase):

  def setUp(self):
    # Mock BigQuery result schema.
    self._schema = [
        bigquery.SchemaField('i', 'INTEGER', mode='REQUIRED'),
        bigquery.SchemaField('f', 'FLOAT', mode='REQUIRED'),
        bigquery.SchemaField('s', 'STRING', mode='REQUIRED'),
    ]

    # Create exe properties.
    self._exec_properties = {
        'query':
            'SELECT i, f, s FROM `fake`',
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

  @mock.patch.multiple(
      executor,
      _ReadFromBigQuery=_MockReadFromBigQuery2,  # pylint: disable=invalid-name, unused-argument
  )
  @mock.patch.object(bigquery, 'Client')
  def testBigQueryToExample(self, mock_client):
    # Mock query result schema for _BigQueryConverter.
    mock_client.return_value.query.return_value.result.return_value.schema = self._schema

    with beam.Pipeline() as pipeline:
      examples = (
          pipeline | 'ToTFExample' >> executor._BigQueryToExample(
              {}, self._exec_properties))

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
      _ReadFromBigQuery=_MockReadFromBigQuery,  # pylint: disable=invalid-name, unused-argument
  )
  @mock.patch.object(bigquery, 'Client')
  def testDo(self, mock_client):
    # Mock query result schema for _BigQueryConverter.
    mock_client.return_value.query.return_value.result.return_value.schema = self._schema

    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    # Create output dict.
    train_examples = types.TfxType(type_name='ExamplesPath', split='train')
    train_examples.uri = os.path.join(output_data_dir, 'train')
    eval_examples = types.TfxType(type_name='ExamplesPath', split='eval')
    eval_examples.uri = os.path.join(output_data_dir, 'eval')
    output_dict = {'examples': [train_examples, eval_examples]}

    # Run executor.
    big_query_example_gen = executor.Executor()
    big_query_example_gen.Do({}, output_dict, self._exec_properties)

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
