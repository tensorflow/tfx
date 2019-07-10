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
"""Generic TFX PrestoExampleGen executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime

import prestodb
import apache_beam as beam
import tensorflow as tf

from typing import Any, Dict, List, Text
from tfx.components.example_gen import base_example_gen_executor
from tfx.utils import types


def get_schema_dict(cursor: prestodb.dbapi.Cursor):
  if not cursor.description:
    return {}

  return {t[0]: t[1] for t in cursor.description}


class _PrestoConverter(object):
  """Help class for prestodb result row to tf example conversion."""

  def __init__(self, query: Text, db_conn: prestodb.dbapi.Connection):
    # Dummy query to get the type information for each field.
    cur = None
    try:
      cur = db_conn.cursor()
      cur.execute('SELECT * FROM ({}) LIMIT 0'.format(query))
      cur.fetchone()
      self._type_map = get_schema_dict(cur)
      for field in cur.description:
        self._type_map[field[0]] = field[1]
    finally:
      if cur:
        cur.close()

  def RowToExample(self, instance: Dict[Text, Any]) -> tf.train.Example:
    """Convert presto result row to tf example."""
    feature = {}
    for key, value in instance.items():
      data_type = self._type_map[key]
      if value is None:
        feature[key] = tf.train.Feature()
      elif data_type in ['tinyint', 'smallint', 'integer', 'bigint']:
        feature[key] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[value]))
      elif data_type in ['real', 'double']:
        feature[key] = tf.train.Feature(
            float_list=tf.train.FloatList(value=[value]))
      elif data_type in ['varchar']:
        if value == '':
          feature[key] = tf.train.Feature()
        else:
          feature[key] = tf.train.Feature(
              bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(value)]))
      elif data_type in ['timestamp']:
        value = int(datetime.fromisoformat(value).timestamp())
        feature[key] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[value]))
      else:
        # TODO(jyzhao): support more types.
        raise RuntimeError(
            'Presto column type {} is not supported.'.format(data_type))
    return tf.train.Example(features=tf.train.Features(feature=feature))


# Create this instead of inline in _PrestoToExample for test mocking purpose.
@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(beam.typehints.Dict[Text, Any])
def _ReadFromPresto(  # pylint: disable=invalid-name
    pipeline: beam.Pipeline, query: Text,
    db_cursor: prestodb.dbapi.Cursor) -> beam.pvalue.PCollection:
  rows = db_cursor.fetchall()
  desc = db_cursor.description
  query_results = list()
  for row in rows:
    query_results.append({desc[i][0]: row[i] for i in range(len(desc))})

  return (pipeline | beam.Create(query_results))


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(tf.train.Example)
def _PrestoToExample(  # pylint: disable=invalid-name
    pipeline: beam.Pipeline,
    input_dict: Dict[Text, List[types.TfxArtifact]],  # pylint: disable=unused-argument
    exec_properties: Dict[Text, Any],  # pylint: disable=unused-argument
    split_pattern: Text) -> beam.pvalue.PCollection:
  """Read from Presto and transform to TF examples.

  Args:
    pipeline: beam pipeline.
    input_dict: Input dict from input key to a list of Artifacts.
    exec_properties: A dict of execution properties.
    split_pattern: Split.pattern in Input config, a Presto sql string.

  Returns:
    PCollection of TF examples.
  """
  db_conn_info = exec_properties['connection_info']
  db_conn = prestodb.dbapi.connect(**db_conn_info)
  converter = _PrestoConverter(split_pattern, db_conn)
  db_cursor = db_conn.cursor()
  db_cursor.execute(split_pattern)

  return (pipeline
          | 'QueryTable' >> _ReadFromPresto(split_pattern, db_cursor)  # pylint: disable=no-value-for-parameter
          | 'ToTFExample' >> beam.Map(converter.RowToExample))


class Executor(base_example_gen_executor.BaseExampleGenExecutor):
  """Generic TFX BigQueryExampleGen executor."""

  def GetInputSourceToExamplePTransform(self) -> beam.PTransform:
    """Returns PTransform for BigQuery to TF examples."""
    return _PrestoToExample
