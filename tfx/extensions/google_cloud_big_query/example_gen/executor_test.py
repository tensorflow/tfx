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
"""Tests for tfx.extensions.google_cloud_big_query.example_gen.executor."""

import os
import random
from unittest import mock

import apache_beam as beam
from apache_beam.testing import util
from google.cloud import bigquery
import tensorflow as tf
from tfx.dsl.components.base import base_beam_executor
from tfx.dsl.io import fileio
from tfx.extensions.google_cloud_big_query import utils
from tfx.extensions.google_cloud_big_query.example_gen import executor
from tfx.proto import example_gen_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.utils import proto_utils


@beam.ptransform_fn
def _MockReadFromBigQuery(pipeline, query):
  del query  # Unused arg
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
def _MockReadFromBigQuery2(pipeline, query):
  del query  # Unused arg
  mock_query_results = [{
      'i': 1,
      'i2': [2, 3],
      'b': True,
      'f': 2.0,
      'f2': [2.7, 3.8],
      's': 'abc',
      's2': ['abc', 'def']
  }]
  return pipeline | beam.Create(mock_query_results)


class ExecutorTest(tf.test.TestCase):

  def setUp(self):
    # Mock BigQuery result schema.
    self._schema = [
        bigquery.SchemaField('i', 'INTEGER', mode='REQUIRED'),
        bigquery.SchemaField('i2', 'INTEGER', mode='REPEATED'),
        bigquery.SchemaField('b', 'BOOLEAN', mode='REQUIRED'),
        bigquery.SchemaField('f', 'FLOAT', mode='REQUIRED'),
        bigquery.SchemaField('f2', 'FLOAT', mode='REPEATED'),
        bigquery.SchemaField('s', 'STRING', mode='REQUIRED'),
        bigquery.SchemaField('s2', 'STRING', mode='REPEATED'),
    ]
    super().setUp()

  @mock.patch.multiple(
      utils,
      ReadFromBigQuery=_MockReadFromBigQuery2,
  )
  @mock.patch.object(bigquery, 'Client')
  def testBigQueryToExample(self, mock_client):
    # Mock query result schema for _BigQueryConverter.
    mock_client.return_value.query.return_value.result.return_value.schema = self._schema

    with beam.Pipeline() as pipeline:
      examples = (
          pipeline | 'ToTFExample' >> executor._BigQueryToExample(
              exec_properties={'_beam_pipeline_args': []},
              split_pattern='SELECT i, i2, b, f, f2, s, s2 FROM `fake`'))

      feature = {}
      feature['i'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[1]))
      feature['i2'] = tf.train.Feature(
          int64_list=tf.train.Int64List(value=[2, 3]))
      feature['b'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[1]))
      feature['f'] = tf.train.Feature(
          float_list=tf.train.FloatList(value=[2.0]))
      feature['f2'] = tf.train.Feature(
          float_list=tf.train.FloatList(value=[2.7, 3.8]))
      feature['s'] = tf.train.Feature(
          bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes('abc')]))
      feature['s2'] = tf.train.Feature(
          bytes_list=tf.train.BytesList(
              value=[tf.compat.as_bytes('abc'),
                     tf.compat.as_bytes('def')]))
      example_proto = tf.train.Example(
          features=tf.train.Features(feature=feature))
      util.assert_that(examples, util.equal_to([example_proto]))

  @mock.patch.multiple(
      utils,
      ReadFromBigQuery=_MockReadFromBigQuery,
  )
  @mock.patch.object(bigquery, 'Client')
  def testDo(self, mock_client):
    # Mock query result schema for _BigQueryConverter.
    mock_client.return_value.query.return_value.result.return_value.schema = self._schema

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
                        name='bq', pattern='SELECT i, b, f, s FROM `fake`'),
                ])),
        'output_config':
            proto_utils.proto_to_json(
                example_gen_pb2.Output(
                    split_config=example_gen_pb2.SplitConfig(splits=[
                        example_gen_pb2.SplitConfig.Split(
                            name='train', hash_buckets=2),
                        example_gen_pb2.SplitConfig.Split(
                            name='eval', hash_buckets=1)
                    ])))
    }

    # Run executor.
    big_query_example_gen = executor.Executor(
        base_beam_executor.BaseBeamExecutor.Context(
            beam_pipeline_args=['--project=test-project']))
    big_query_example_gen.Do({}, output_dict, exec_properties)

    mock_client.assert_called_with(project='test-project')

    self.assertEqual(
        artifact_utils.encode_split_names(['train', 'eval']),
        examples.split_names)

    # Check BigQuery example gen outputs.
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
