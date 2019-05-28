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
"""Parquet based TFX example gen executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import apache_beam as beam
import six
import tensorflow as tf
from typing import Any, Dict, List, Text
from tfx.components.example_gen import base_example_gen_executor
from tfx.utils import types


def _dict_to_example(instance: Dict[Text, Any]) -> tf.train.Example:
  """Decoded parquet to tf example."""
  # Note that when convert to tf.Feature, Parquet data might lose precision.
  feature = {}
  for key, value in instance.items():
    # TODO(jyzhao): support more types.
    if value is None:
      feature[key] = tf.train.Feature()
    elif isinstance(value, six.integer_types):
      feature[key] = tf.train.Feature(
          int64_list=tf.train.Int64List(value=[value]))
    elif isinstance(value, float):
      feature[key] = tf.train.Feature(
          float_list=tf.train.FloatList(value=[value]))
    elif isinstance(value, six.text_type) or isinstance(value, str):
      feature[key] = tf.train.Feature(
          bytes_list=tf.train.BytesList(value=[value.encode('utf-8')]))
    elif isinstance(value, list):
      if not value:
        feature[key] = tf.train.Feature()
      elif isinstance(value[0], six.integer_types):
        feature[key] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=value))
      elif isinstance(value[0], float):
        feature[key] = tf.train.Feature(
            float_list=tf.train.FloatList(value=value))
      elif isinstance(value[0], six.text_type) or isinstance(value[0], str):
        feature[key] = tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[v.encode('utf-8') for v in value]))
      else:
        raise RuntimeError(
            'Parquet column type `list of {}` is not supported.'.format(
                type(value[0])))
    else:
      raise RuntimeError('Parquet column type {} is not supported.'.format(
          type(value)))
  return tf.train.Example(features=tf.train.Features(feature=feature))


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(tf.train.Example)
def _ParquetToExample(  # pylint: disable=invalid-name
    pipeline: beam.Pipeline,
    input_dict: Dict[Text, List[types.TfxArtifact]],
    exec_properties: Dict[Text, Any],  # pylint: disable=unused-argument
    split_pattern: Text) -> beam.pvalue.PCollection:
  """Read Parquet files and transform to TF examples.

  Note that each input split will be transformed by this function separately.

  Args:
    pipeline: beam pipeline.
    input_dict: Input dict from input key to a list of Artifacts.
      - input_base: input dir that contains Parquet data.
    exec_properties: A dict of execution properties.
    split_pattern: Split.pattern in Input config, glob relative file pattern
      that maps to input files with root directory given by input_base.

  Returns:
    PCollection of TF examples.
  """
  input_base_uri = types.get_single_uri(input_dict['input_base'])
  parquet_pattern = os.path.join(input_base_uri, split_pattern)
  tf.logging.info(
      'Processing input parquet data {} to TFExample.'.format(parquet_pattern))

  return (pipeline
          # TODO(jyzhao): support per column read by input_config.
          | 'ReadFromParquet' >> beam.io.ReadFromParquet(parquet_pattern)
          | 'ToTFExample' >> beam.Map(_dict_to_example))


class Executor(base_example_gen_executor.BaseExampleGenExecutor):
  """TFX example gen executor for processing parquet format.

  Data type conversion:
    integer types will be converted to tf.train.Feature with tf.train.Int64List.
    float types will be converted to tf.train.Feature with tf.train.FloatList.
    string types will be converted to tf.train.Feature with tf.train.BytesList
      and utf-8 encoding.

    Note that,
      Single value will be converted to a list of that single value.
      Missing value will be converted to empty tf.train.Feature().
      Parquet data might lose precision, e.g., int96.

    For details, check the _dict_to_example function above.


  Example usage:

    from tfx.components.example_gen.component import
    ExampleGen
    from tfx.components.example_gen.custom_executors import
    parquet_executor
    from tfx.utils.dsl_utils import external_input

    example_gen = ExampleGen(executor=parquet_executor.Executor,
                             input_base=external_input(parquet_dir_path))
  """

  def GetInputSourceToExamplePTransform(self) -> beam.PTransform:
    """Returns PTransform for parquet to TF examples."""
    return _ParquetToExample
