# Copyright 2021 Google LLC. All Rights Reserved.
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
"""PTransform for write split."""
import os
from typing import Union, Dict, Any

import apache_beam as beam
import pyarrow as pa
import tensorflow as tf
from tfx.proto import example_gen_pb2
from tfx_bsl.telemetry import util

# Default file name for TFRecord output file prefix.
from tfx.types import standard_component_specs

DEFAULT_PARQUET_FILE_NAME = 'data_parquet'
DEFAULT_TF_RECORD_FILE_NAME = 'data_tfrecord'

# Metrics namespace for ExampleGen.
METRICS_NAMESPACE = util.MakeTfxNamespace(['ExampleGen'])


@beam.typehints.with_input_types(
    Union[tf.train.Example, tf.train.SequenceExample, bytes, pa.Table])
@beam.typehints.with_output_types(bytes)
class MaybeSerialize(beam.DoFn):
  """Serializes the proto if needed."""

  def __init__(self):
    self._num_instances = beam.metrics.Metrics.counter(METRICS_NAMESPACE,
                                                       'num_instances')

  def process(self, e: Union[tf.train.Example, tf.train.SequenceExample,
                             bytes]):
    self._num_instances.inc(1)
    if isinstance(e, (tf.train.Example, tf.train.SequenceExample)):
      yield e.SerializeToString()  # pytype: disable=attribute-error
    else:
      yield e


@beam.ptransform_fn
@beam.typehints.with_input_types(
    Union[tf.train.Example, tf.train.SequenceExample, bytes, Dict[str, Any]])
def WriteSplit(
    example_split: beam.pvalue.PCollection,
    output_split_path: str,
    output_format: str,
    exec_properties: Dict[str, Any]
) -> beam.pvalue.PDone:
  """
  Shuffles and writes output split as serialized records in TFRecord or Parquet
  """
  del output_format

  output_payload_format = exec_properties.get(
    standard_component_specs.OUTPUT_DATA_FORMAT_KEY)

  if output_payload_format == example_gen_pb2.PayloadFormat.FORMAT_PARQUET:
    # TODO: propagate schema.
    schema = exec_properties.get("pyarrow_schema")
    return (example_split
            # TODO(jyzhao): make shuffle optional.
            | 'Shuffle' >> beam.transforms.Reshuffle()
            | "Write Parquet" >> beam.io.WriteToParquet(
              os.path.join(output_split_path, DEFAULT_PARQUET_FILE_NAME),
              schema, file_name_suffix=".parquet", codec='snappy'))

  return (example_split
          | 'MaybeSerialize' >> beam.ParDo(MaybeSerialize())
          # TODO(jyzhao): make shuffle optional.
          | 'Shuffle' >> beam.transforms.Reshuffle()
          | 'Write' >> beam.io.WriteToTFRecord(
              os.path.join(output_split_path, DEFAULT_TF_RECORD_FILE_NAME),
              file_name_suffix='.gz'))


def to_file_format_str(file_format: example_gen_pb2.FileFormat) -> str:  # pylint: disable=invalid-name
  if (file_format == example_gen_pb2.FILE_FORMAT_UNSPECIFIED or
      file_format == example_gen_pb2.FORMAT_TFRECORDS_GZIP):
    return 'tfrecords_gzip'
  raise ValueError('File format is not valid.')
