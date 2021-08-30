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
"""Generic TFX ImportExampleGen executor."""

import os
from typing import Any, Dict, Union

from absl import logging
import apache_beam as beam
import tensorflow as tf

from tfx.components.example_gen import base_example_gen_executor
from tfx.proto import example_gen_pb2
from tfx.types import standard_component_specs


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(bytes)
def _ImportSerializedRecord(  # pylint: disable=invalid-name
    pipeline: beam.Pipeline, exec_properties: Dict[str, Any],
    split_pattern: str) -> beam.pvalue.PCollection:
  """Read TFRecord files to PCollection of records.

  Note that each input split will be transformed by this function separately.

  Args:
    pipeline: Beam pipeline.
    exec_properties: A dict of execution properties.
      - input_base: input dir that contains input data.
    split_pattern: Split.pattern in Input config, glob relative file pattern
      that maps to input files with root directory given by input_base.

  Returns:
    PCollection of records (tf.Example, tf.SequenceExample, or bytes).
  """
  input_base_uri = exec_properties[standard_component_specs.INPUT_BASE_KEY]
  input_split_pattern = os.path.join(input_base_uri, split_pattern)
  logging.info('Reading input TFRecord data %s.', input_split_pattern)

  # TODO(jyzhao): profile input examples.
  return (pipeline
          # TODO(jyzhao): support multiple input container format.
          | 'ReadFromTFRecord' >>
          beam.io.ReadFromTFRecord(file_pattern=input_split_pattern))


class Executor(base_example_gen_executor.BaseExampleGenExecutor):
  """Generic TFX import example gen executor."""

  def GetInputSourceToExamplePTransform(self) -> beam.PTransform:
    """Returns PTransform for importing records."""

    @beam.ptransform_fn
    @beam.typehints.with_input_types(beam.Pipeline)
    @beam.typehints.with_output_types(Union[tf.train.Example,
                                            tf.train.SequenceExample, bytes])
    def ImportRecord(pipeline: beam.Pipeline, exec_properties: Dict[str, Any],
                     split_pattern: str) -> beam.pvalue.PCollection:
      """PTransform to import records.

      The records are tf.train.Example, tf.train.SequenceExample,
      or serialized proto.

      Args:
        pipeline: Beam pipeline.
        exec_properties: A dict of execution properties.
          - input_base: input dir that contains input data.
        split_pattern: Split.pattern in Input config, glob relative file pattern
          that maps to input files with root directory given by input_base.

      Returns:
        PCollection of records (tf.Example, tf.SequenceExample, or bytes).
      """
      output_payload_format = exec_properties.get(
          standard_component_specs.OUTPUT_DATA_FORMAT_KEY)

      serialized_records = (
          pipeline
          # pylint: disable=no-value-for-parameter
          | _ImportSerializedRecord(exec_properties, split_pattern))
      if output_payload_format == example_gen_pb2.PayloadFormat.FORMAT_PROTO:
        return serialized_records
      elif (output_payload_format ==
            example_gen_pb2.PayloadFormat.FORMAT_TF_EXAMPLE):
        return (serialized_records
                | 'ToTFExample' >> beam.Map(tf.train.Example.FromString))
      elif (output_payload_format ==
            example_gen_pb2.PayloadFormat.FORMAT_TF_SEQUENCE_EXAMPLE):
        return (serialized_records
                | 'ToTFSequenceExample' >> beam.Map(
                    tf.train.SequenceExample.FromString))

      raise ValueError('output_payload_format must be one of FORMAT_TF_EXAMPLE,'
                       ' FORMAT_TF_SEQUENCE_EXAMPLE or FORMAT_PROTO')

    return ImportRecord
