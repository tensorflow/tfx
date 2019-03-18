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

import apache_beam as beam
import numpy
import tensorflow as tf
import tensorflow_data_validation as tfdv
from tensorflow_data_validation.coders import csv_decoder
from typing import Any, Dict, List, Text
from tfx.components.example_gen import base_example_gen_executor
from tfx.utils import io_utils
from tfx.utils import types


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
  return tf.train.Example(features=tf.train.Features(feature=feature))


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(tf.train.Example)
def _CsvToExample(  # pylint: disable=invalid-name
    pipeline, input_dict,
    exec_properties):  # pylint: disable=unused-argument
  """Read CSV file and transform to TF examples.

  Args:
    pipeline: beam pipeline.
    input_dict: Input dict from input key to a list of Artifacts.
      - input-base: input dir that contains csv data. csv files must have header
        line.
    exec_properties: A dict of execution properties.

  Returns:
    PCollection of TF examples.
  """
  input_base = types.get_single_instance(input_dict['input-base'])
  input_base_uri = input_base.uri
  csv_uri = io_utils.get_only_uri_in_dir(input_base_uri)
  tf.logging.info('Processing input csv data {} to TFExample.'.format(csv_uri))

  return (pipeline
          |
          'ReadFromText' >> beam.io.ReadFromText(csv_uri, skip_header_lines=1)
          | 'ParseCSV' >> csv_decoder.DecodeCSV(
              io_utils.load_csv_column_names(csv_uri))
          | 'ToTFExample' >> beam.Map(_dict_to_example))


class Executor(base_example_gen_executor.BaseExampleGenExecutor):
  """Generic TFX CSV example gen executor."""

  def GetInputSourceToExamplePTransform(self):
    """Returns PTransform for CSV to TF examples."""
    return _CsvToExample
