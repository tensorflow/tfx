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
"""Generic TFX parquet example gen executor."""

import apache_beam as beam
import tensorflow as tf
from tfx.components.example_gen import base_example_gen_executor
from tfx.utils import io_utils
from tfx.utils import types

def _dict_to_example(instance):
  """Decoded parquet to tf example."""
  # Ideally this would use feature types from the parquet schema, but that isn't
  # readily available from beam
  feature = {}
  for key, value in instance.items():
    if value is None:
      feature[key] = tf.train.Feature()
    elif isinstance(value, int):
      feature[key] = tf.train.Feature(
          int64_list=tf.train.Int64List(value=[value]))
    elif isinstance(value, float):
      feature[key] = tf.train.Feature(
          float_list=tf.train.FloatList(value=[value]))
    else:
      feature[key] = tf.train.Feature(
          bytes_list=tf.train.BytesList(value=[value.encode('utf-8')]))
  return tf.train.Example(features=tf.train.Features(feature=feature))

@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(tf.train.Example)
def _ParquetToExample(
    pipeline, input_dict,
    exec_properties):
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
  base_uri = input_base.uri
  tf.logging.info('Processing input parquet data {} to TFExample.'.format(base_uri))

  return (pipeline
         | 'ReadFromParquet' >> beam.io.ReadFromParquet(base_uri)
         | 'ToTFExample' >> beam.Map(_dict_to_example))



class Executor(base_example_gen_executor.BaseExampleGenExecutor):
  """Generic TFX parquet example gen executor."""

  def GetInputSourceToExamplePTransform(self):
    """Returns PTransform for parquet to TF examples."""
    return _ParquetToExample
