# Lint as: python2, python3
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
from typing import Any, Dict, Text

from absl import logging
import apache_beam as beam
import tensorflow as tf

from tfx.components.example_gen import utils
from tfx.components.example_gen.base_example_gen_executor import BaseExampleGenExecutor


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(tf.train.Example)
def _ParquetToExample(  # pylint: disable=invalid-name
    pipeline: beam.Pipeline, exec_properties: Dict[Text, Any],
    split_pattern: Text) -> beam.pvalue.PCollection:
  """Read Parquet files and transform to TF examples.

  Note that each input split will be transformed by this function separately.

  Args:
    pipeline: beam pipeline.
    exec_properties: A dict of execution properties.
      - input_base: input dir that contains Parquet data.
    split_pattern: Split.pattern in Input config, glob relative file pattern
      that maps to input files with root directory given by input_base.

  Returns:
    PCollection of TF examples.
  """
  input_base_uri = exec_properties[utils.INPUT_BASE_KEY]
  parquet_pattern = os.path.join(input_base_uri, split_pattern)
  logging.info('Processing input parquet data %s to TFExample.',
               parquet_pattern)

  return (pipeline
          # TODO(jyzhao): support per column read by input_config.
          | 'ReadFromParquet' >> beam.io.ReadFromParquet(parquet_pattern)
          | 'ToTFExample' >> beam.Map(utils.dict_to_example))


class Executor(BaseExampleGenExecutor):
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

    For details, check the dict_to_example function in example_gen.utils.


  Example usage:

    from tfx.components.example_gen.component import
    FileBasedExampleGen
    from tfx.components.example_gen.custom_executors import
    parquet_executor
    from tfx.utils.dsl_utils import external_input

    example_gen = FileBasedExampleGen(
        input=external_input(parquet_dir_path),
        executor_class=parquet_executor.Executor)
  """

  def GetInputSourceToExamplePTransform(self) -> beam.PTransform:
    """Returns PTransform for parquet to TF examples."""
    return _ParquetToExample
