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
"""Generic TFX PrestoExampleGen executor."""

import datetime
from typing import Any, Dict, Iterable, Text, Tuple

import apache_beam as beam
import prestodb
import tensorflow as tf
from tfx.components.example_gen import base_example_gen_executor
from tfx.examples.custom_components.presto_example_gen.proto import presto_config_pb2
from tfx.proto import example_gen_pb2
from tfx.utils import proto_utils


@beam.typehints.with_input_types(Text)
@beam.typehints.with_output_types(beam.typehints.Iterable[Tuple[Text, Text,
                                                                Any]])
class _ReadPrestoDoFn(beam.DoFn):
  """Beam DoFn class that reads from Presto.

  Attributes:
    cursor: A prestodb.dbapi.Cursor object that reads records from Presto table.
  """

  def __init__(self, client: prestodb.dbapi.Connection):
    self.cursor = client.cursor()

  def process(self, query: Text) -> Iterable[Tuple[Text, Text, Any]]:
    """Yields rows from query results.

    Args:
      query: A SQL query used to return results from Presto table.

    Yields:
      One row from the query result, represented by a list of tuples. Each tuple
      contains information on column name, column data type, data.
    """
    self.cursor.execute(query)
    rows = self.cursor.fetchall()
    if rows:
      cols = []
      col_types = []
      # Returns a list of (column_name, column_type, None, ...)
      # https://github.com/prestodb/presto-python-client/blob/master/prestodb/dbapi.py#L199
      for metadata in self.cursor.description:
        cols.append(metadata[0])
        col_types.append(metadata[1])

      for r in rows:
        yield zip(cols, col_types, r)

  def teardown(self):
    if self.cursor:
      self.cursor.close()


def _deserialize_conn_config(
    conn_config: presto_config_pb2.PrestoConnConfig
) -> prestodb.dbapi.Connection:
  """Deserializes Presto connection config to Presto client.

  Args:
    conn_config: Protobuf-encoded connection config for Presto client.

  Returns:
    A prestodb.dbapi.Connection instance initialized with user-supplied
    parameters.
  """
  params = {'host': conn_config.host}  # Required field
  # Only deserialize rest of parameters if set by user
  if conn_config.HasField('port'):
    params['port'] = conn_config.port
  if conn_config.HasField('user'):
    params['user'] = conn_config.user
  if conn_config.HasField('source'):
    params['source'] = conn_config.source
  if conn_config.HasField('catalog'):
    params['catalog'] = conn_config.catalog
  if conn_config.HasField('schema'):
    params['schema'] = conn_config.schema
  if conn_config.HasField('http_scheme'):
    params['http_scheme'] = conn_config.http_scheme
  if conn_config.WhichOneof('opt_auth'):
    params['auth'] = _deserialize_auth_config(conn_config)
  if conn_config.HasField('max_attempts'):
    params['max_attempts'] = conn_config.max_attempts
  if conn_config.HasField('request_timeout'):
    params['request_timeout'] = conn_config.request_timeout

  return prestodb.dbapi.connect(**params)


def _deserialize_auth_config(
    conn_config: presto_config_pb2.PrestoConnConfig
) -> prestodb.auth.Authentication:
  """Extracts from conn config the deserialized Presto Authentication class.

  Args:
    conn_config: Protobuf-encoded connection config for Presto client.

  Returns:
    A prestodb.auth.Authentication instance initialized with user-supplied
    parameters.

  Raises:
    RuntimeError: if authentication type is not currently supported.
  """
  if conn_config.HasField('basic_auth'):
    return prestodb.auth.BasicAuthentication(conn_config.basic_auth.username,
                                             conn_config.basic_auth.password)
    # TODO(b/140266796): Support KerberosAuth.
  else:
    raise RuntimeError('Authentication type not supported.')


def _row_to_example(
    instance: Iterable[Tuple[Text, Text, Any]]) -> tf.train.Example:
  """Convert presto result row to tf example."""
  feature = {}
  for key, data_type, value in instance:
    if value is None:
      feature[key] = tf.train.Feature()
    elif data_type in {'tinyint', 'smallint', 'integer', 'bigint'}:
      feature[key] = tf.train.Feature(
          int64_list=tf.train.Int64List(value=[value]))
    elif data_type in {'real', 'double', 'decimal'}:
      feature[key] = tf.train.Feature(
          float_list=tf.train.FloatList(value=[value]))
    elif data_type in {'varchar', 'char'}:
      feature[key] = tf.train.Feature(
          bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(value)]))
    elif data_type in {'timestamp'}:
      value = int(datetime.datetime.fromisoformat(value).timestamp())
      feature[key] = tf.train.Feature(
          int64_list=tf.train.Int64List(value=[value]))
    else:
      # TODO(b/140266796): support more types
      # https://prestodb.github.io/docs/current/language/types
      raise RuntimeError(
          'Presto column type {} is not supported.'.format(data_type))
  return tf.train.Example(features=tf.train.Features(feature=feature))


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(tf.train.Example)
def _PrestoToExample(  # pylint: disable=invalid-name
    pipeline: beam.Pipeline,
    exec_properties: Dict[Text, Any],
    split_pattern: Text) -> beam.pvalue.PCollection:
  """Read from Presto and transform to TF examples.

  Args:
    pipeline: beam pipeline.
    exec_properties: A dict of execution properties.
    split_pattern: Split.pattern in Input config, a Presto sql string.

  Returns:
    PCollection of TF examples.
  """
  conn_config = example_gen_pb2.CustomConfig()
  proto_utils.json_to_proto(exec_properties['custom_config'], conn_config)
  presto_config = presto_config_pb2.PrestoConnConfig()
  conn_config.custom_config.Unpack(presto_config)

  client = _deserialize_conn_config(presto_config)
  return (pipeline
          | 'Query' >> beam.Create([split_pattern])
          | 'QueryTable' >> beam.ParDo(_ReadPrestoDoFn(client))
          | 'ToTFExample' >> beam.Map(_row_to_example))


class Executor(base_example_gen_executor.BaseExampleGenExecutor):
  """Generic TFX PrestoExampleGen executor."""

  def GetInputSourceToExamplePTransform(self) -> beam.PTransform:
    """Returns PTransform for Presto to TF examples."""
    return _PrestoToExample
