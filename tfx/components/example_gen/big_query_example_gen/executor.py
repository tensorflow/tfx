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

import hashlib
import os
import apache_beam as beam
import tensorflow as tf
from typing import Any, Callable, Dict, List, Optional, Text
from google.cloud import bigquery
from tfx.components.base import base_executor
from tfx.utils import types

# Default file name for TFRecord output file prefix.
DEFAULT_FILE_NAME = 'data_tfrecord'


def _partition_fn(record, num_partitions):  # pylint: disable=unused-argument
  # TODO(jyzhao): support custom split.
  # Splits data, train(partition=0) : eval(partition=1) = 2 : 1
  return 1 if int(hashlib.sha256(record).hexdigest(), 16) % 3 == 0 else 0


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(Dict[Text, Any])
def _ReadFromBigQuery(  # pylint: disable=invalid-name
    pipeline, query):
  return (pipeline
          | 'QueryTable' >> beam.io.Read(
              beam.io.BigQuerySource(query=query, use_standard_sql=True)))


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

  def row_to_serialized_example(self, instance):
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
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    # Returns deterministic result as downstream partition based on it.
    return example_proto.SerializeToString(deterministic=True)


# TODO(jyzhao): BaseExampleGen for common stuff sharing.
class Executor(base_executor.BaseExecutor):
  """Generic TFX BigQueryExampleGen executor."""

  def __init__(self,
               beam_pipeline_args = None,
               big_query_ptransform_for_testing = None):
    """Construct a BigQueryExampleGen Executor.

    Args:
      beam_pipeline_args: beam pipeline args.
      big_query_ptransform_for_testing: for testing use only.
    """
    super(Executor, self).__init__(beam_pipeline_args)
    self._big_query_ptransform = (
        big_query_ptransform_for_testing or _ReadFromBigQuery)

  def Do(self, input_dict,
         output_dict,
         exec_properties):
    """Take BigQuery sql and generates train and eval tf examples.

    Args:
      input_dict: Input dict from input key to a list of Artifacts.
      output_dict: Output dict from output key to a list of Artifacts.
        - examples: train and eval split of tf examples.
      exec_properties: A dict of execution properties.
        - query: BigQuery sql string.

    Returns:
      None

    Raises:
      RuntimeError: if query is missing in exec_properties.
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    training_tfrecord = types.get_split_uri(output_dict['examples'], 'train')
    eval_tfrecord = types.get_split_uri(output_dict['examples'], 'eval')

    if 'query' not in exec_properties:
      raise RuntimeError('Missing query.')
    query = exec_properties['query']

    tf.logging.info('Generating examples from BigQuery.')
    with beam.Pipeline(argv=self._get_beam_pipeline_args()) as pipeline:
      converter = _BigQueryConverter(query)
      example_splits = (
          pipeline
          | 'QueryTable' >> self._big_query_ptransform(query)
          | 'ToSerializedTFExample' >> beam.Map(
              converter.row_to_serialized_example)
          | 'SplitData' >> beam.Partition(_partition_fn, 2))
      # TODO(jyzhao): make shuffle optional.
      # pylint: disable=expression-not-assigned
      (example_splits[0]
       | 'ShuffleTrainSplit' >> beam.transforms.Reshuffle()
       | 'OutputTrainSplit' >> beam.io.WriteToTFRecord(
           os.path.join(training_tfrecord, DEFAULT_FILE_NAME),
           file_name_suffix='.gz'))
      (example_splits[1]
       | 'ShuffleEvalSplit' >> beam.transforms.Reshuffle()
       | 'OutputEvalSplit' >> beam.io.WriteToTFRecord(
           os.path.join(eval_tfrecord, DEFAULT_FILE_NAME),
           file_name_suffix='.gz'))
      # pylint: enable=expression-not-assigned
    tf.logging.info('Examples generated.')
