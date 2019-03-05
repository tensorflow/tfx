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
"""Generic TFX CSV example gen executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import os
import apache_beam as beam
import numpy
import tensorflow as tf
import tensorflow_data_validation as tfdv
from tensorflow_data_validation.coders import csv_decoder
from typing import Any, Dict, List, Text
from tfx.components.base import base_executor
from tfx.utils import io_utils
from tfx.utils import types

# Default file name for TFRecord output file prefix.
DEFAULT_FILE_NAME = 'data_tfrecord'


def _partition_fn(record, num_partitions):  # pylint: disable=unused-argument
  # TODO(jyzhao): support custom split.
  # Splits data, train(partition=0) : eval(partition=1) = 2 : 1
  return 1 if int(hashlib.sha256(record).hexdigest(), 16) % 3 == 0 else 0


def _dict_to_example(instance):
  """Decoded CSV to tf example."""
  feature = {}
  for key, value in instance.items():
    if value is None:
      feature[key] = tf.train.Feature()
    elif value.dtype == numpy.integer:
      feature[key] = tf.train.Feature(
          int64_list=tf.train.Int64List(value=value.tolist()))
    elif value.dtype == numpy.float32:
      feature[key] = tf.train.Feature(
          float_list=tf.train.FloatList(value=value.tolist()))
    else:
      feature[key] = tf.train.Feature(
          bytes_list=tf.train.BytesList(value=value.tolist()))
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  # Returns deterministic result as downstream partition based on it.
  return example_proto.SerializeToString(deterministic=True)


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(bytes)
def _CsvToSerializedExample(  # pylint: disable=invalid-name
    pipeline, csv_uri):
  """Read csv file and transform to tf examples."""
  return (pipeline
          |
          'ReadFromText' >> beam.io.ReadFromText(csv_uri, skip_header_lines=1)
          | 'ParseCSV' >> csv_decoder.DecodeCSV(
              io_utils.load_csv_column_names(csv_uri))
          | 'ToSerializedTFExample' >> beam.Map(_dict_to_example))


# TODO(jyzhao): BaseExampleGen for common stuff sharing.
class Executor(base_executor.BaseExecutor):
  """Generic TFX CSV example gen executor."""

  def Do(self, input_dict,
         output_dict,
         exec_properties):
    """Take input csv data and generates train and eval tf examples.

    Args:
      input_dict: Input dict from input key to a list of Artifacts.
        - input-base: input dir that contains csv data. csv files must have
          header line.
      output_dict: Output dict from output key to a list of Artifacts.
        - examples: train and eval split of tf examples.
      exec_properties: A dict of execution properties.

    Returns:
      None
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    training_tfrecord = types.get_split_uri(output_dict['examples'], 'train')
    eval_tfrecord = types.get_split_uri(output_dict['examples'], 'eval')

    input_base = types.get_single_instance(input_dict['input-base'])
    input_base_uri = input_base.uri

    tf.logging.info('Generating examples.')

    raw_data = io_utils.get_only_uri_in_dir(input_base_uri)
    tf.logging.info('No split {}.'.format(raw_data))

    with beam.Pipeline(argv=self._get_beam_pipeline_args()) as pipeline:
      example_splits = (
          pipeline
          # pylint: disable=no-value-for-parameter
          | 'CsvToSerializedExample' >> _CsvToSerializedExample(raw_data)
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
