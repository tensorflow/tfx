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
import mock
import tensorflow as tf
from google.cloud import bigquery
from tfx.components.example_gen.big_query_example_gen import executor
from tfx.utils import types


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


class ExecutorTest(tf.test.TestCase):

  @mock.patch.object(bigquery, 'Client')
  def test_do(self, mock_client):
    schema = [
        bigquery.SchemaField('i', 'INTEGER', mode='REQUIRED'),
        bigquery.SchemaField('f', 'FLOAT', mode='REQUIRED'),
        bigquery.SchemaField('s', 'STRING', mode='REQUIRED'),
    ]
    # Mock query result schema for _BigQueryConverter.
    mock_client.return_value.query.return_value.result.return_value.schema = schema

    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    # Create exe properties.
    exec_properties = {
        'query': 'SELECT i, f, s FROM `fake`',
    }

    # Create output dict.
    train_examples = types.TfxType(type_name='ExamplesPath', split='train')
    train_examples.uri = os.path.join(output_data_dir, 'train')
    eval_examples = types.TfxType(type_name='ExamplesPath', split='eval')
    eval_examples.uri = os.path.join(output_data_dir, 'eval')
    output_dict = {'examples': [train_examples, eval_examples]}

    # Run executor.
    big_query_example_gen = executor.Executor(
        big_query_ptransform_for_testing=_MockReadFromBigQuery)
    big_query_example_gen.Do({}, output_dict, exec_properties)

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
