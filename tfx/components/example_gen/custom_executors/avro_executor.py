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
"""Avro based TFX example gen executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import absl
import apache_beam as beam
import tensorflow as tf
from typing import Any, Dict, List, Text
from tfx import types
from tfx.components.example_gen import base_example_gen_executor
from tfx.components.example_gen.utils import dict_to_example
from tfx.types import artifact_utils


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(tf.train.Example)
def _AvroToExample(  # pylint: disable=invalid-name
    pipeline: beam.Pipeline,
    input_dict: Dict[Text, List[types.Artifact]],
    exec_properties: Dict[Text, Any],  # pylint: disable=unused-argument
    split_pattern: Text) -> beam.pvalue.PCollection:
  """Read Avro files and transform to TF examples.

  Note that each input split will be transformed by this function separately.

  Args:
    pipeline: beam pipeline.
    input_dict: Input dict from input key to a list of Artifacts.
      - input_base: input dir that contains Avro data.
    exec_properties: A dict of execution properties.
    split_pattern: Split.pattern in Input config, glob relative file pattern
      that maps to input files with root directory given by input_base.

  Returns:
    PCollection of TF examples.
  """
  input_base_uri = artifact_utils.get_single_uri(input_dict['input_base'])
  avro_pattern = os.path.join(input_base_uri, split_pattern)
  absl.logging.info(
      'Processing input avro data {} to TFExample.'.format(avro_pattern))

  return (pipeline
          | 'ReadFromAvro' >> beam.io.ReadFromAvro(avro_pattern)
          | 'ToTFExample' >> beam.Map(dict_to_example))


class Executor(base_example_gen_executor.BaseExampleGenExecutor):
  """TFX example gen executor for processing avro format.

  Data type conversion:
    integer types will be converted to tf.train.Feature with tf.train.Int64List.
    float types will be converted to tf.train.Feature with tf.train.FloatList.
    string types will be converted to tf.train.Feature with tf.train.BytesList
      and utf-8 encoding.

    Note that,
      Single value will be converted to a list of that single value.
      Missing value will be converted to empty tf.train.Feature().

    For details, check the dict_to_example function in example_gen.utils.


  Example usage:

    from tfx.components.example_gen.component import
    FileBasedExampleGen
    from tfx.components.example_gen.custom_executors import
    avro_executor
    from tfx.utils.dsl_utils import external_input

    example_gen = FileBasedExampleGen(
        input=external_input(avro_dir_path),
        executor_class=avro_executor.Executor)
  """

  def GetInputSourceToExamplePTransform(self) -> beam.PTransform:
    """Returns PTransform for avro to TF examples."""
    return _AvroToExample
