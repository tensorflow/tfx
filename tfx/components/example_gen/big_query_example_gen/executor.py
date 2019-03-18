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
"""Generic TFX BigQueryExampleGen executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import apache_beam as beam
import tensorflow as tf
from typing import Any, Dict, List, Text
from google.cloud import bigquery
from tfx.components.example_gen import base_example_gen_executor
from tfx.utils import types


class _BigQueryConverter(object):
  """Help class for bigquery result row to tf example conversion."""

  def __init__(self, query):
    client = bigquery.Client()
    # Dummy query to get the type information for each field.
    query_job = client.query('SELECT * FROM ({}) LIMIT 0'.format(query))
    results = query_job.result()
    self._type_map = {}
    for field in results.schema:
      self._type_map[field.name] = field.field_type

  def RowToExample(self, instance):
    """Convert bigquery result row to tf example."""
    feature = {}
    for key, value in instance.items():
      data_type = self._type_map[key]
      if value is None:
        feature[key] = tf.train.Feature()
      elif data_type == 'INTEGER':
        feature[key] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[value]))
      elif data_type == 'FLOAT':
        feature[key] = tf.train.Feature(
            float_list=tf.train.FloatList(value=[value]))
      elif data_type == 'STRING':
        feature[key] = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(value)]))
      else:
        # TODO(jyzhao): support more types.
        raise RuntimeError(
            'BigQuery column type {} is not supported.'.format(data_type))
    return tf.train.Example(features=tf.train.Features(feature=feature))


# Create this instead of inline in _BigQueryToExample for test mocking purpose.
@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(beam.typehints.Dict[Text, Any])
def _ReadFromBigQuery(  # pylint: disable=invalid-name
    pipeline, query):
  return (pipeline
          | 'QueryTable' >> beam.io.Read(
              beam.io.BigQuerySource(query=query, use_standard_sql=True)))


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(tf.train.Example)
def _BigQueryToExample(  # pylint: disable=invalid-name
    pipeline,
    input_dict,  # pylint: disable=unused-argument
    exec_properties):
  """Read from BigQuery and transform to TF examples.

  Args:
    pipeline: beam pipeline.
    input_dict: Input dict from input key to a list of Artifacts.
    exec_properties: A dict of execution properties.
      - query: BigQuery sql string.

  Returns:
    PCollection of TF examples.

  Raises:
    RuntimeError: if query is missing in exec_properties.
  """
  if 'query' not in exec_properties:
    raise RuntimeError('Missing query.')
  query = exec_properties['query']
  converter = _BigQueryConverter(query)

  return (pipeline
          | 'QueryTable' >> _ReadFromBigQuery(query)  # pylint: disable=no-value-for-parameter
          | 'ToTFExample' >> beam.Map(converter.RowToExample))


class Executor(base_example_gen_executor.BaseExampleGenExecutor):
  """Generic TFX BigQueryExampleGen executor."""

  def GetInputSourceToExamplePTransform(self):
    """Returns PTransform for BigQuery to TF examples."""
    return _BigQueryToExample
