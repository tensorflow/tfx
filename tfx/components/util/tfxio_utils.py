# Lint as: python2, python3
# Copyright 2020 Google LLC. All Rights Reserved.
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
"""TFXIO (standardized TFX inputs) related utilities."""

# TODO(b/149535307): Remove __future__ imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Callable, List, Optional, Text, Union

from tfx.components.experimental.data_view import constants
from tfx.components.util import examples_utils
from tfx.proto import example_gen_pb2
from tfx.types import artifact
from tfx.types import standard_artifacts
import tfx_bsl
from tfx_bsl.tfxio import raw_tf_record
from tfx_bsl.tfxio import tf_example_record
from tfx_bsl.tfxio import tf_sequence_example_record
from tfx_bsl.tfxio import tfxio
from tensorflow_metadata.proto.v0 import schema_pb2

# TODO(b/161449255): clean this up after a release post tfx_bsl 0.22.1.
if getattr(tfx_bsl, 'HAS_TF_GRAPH_RECORD_DECODER', False):
  from tfx_bsl.tfxio import record_to_tensor_tfxio  # pylint: disable=g-import-not-at-top
else:
  record_to_tensor_tfxio = None


def get_tfxio_factory_from_artifact(
    examples: artifact.Artifact,
    telemetry_descriptors: List[Text],
    schema: Optional[schema_pb2.Schema] = None,
    read_as_raw_records: bool = False,
    raw_record_column_name: Optional[Text] = None
) -> Callable[[Text], tfxio.TFXIO]:
  """Returns a factory function that creates a proper TFXIO.

  Args:
    examples: The Examples artifact that the TFXIO is intended to access.
    telemetry_descriptors: A set of descriptors that identify the component
      that is instantiating the TFXIO. These will be used to construct the
      namespace to contain metrics for profiling and are therefore expected to
      be identifiers of the component itself and not individual instances of
      source use.
    schema: TFMD schema. Note that without a schema, some TFXIO interfaces
      in certain TFXIO implementations might not be available.
    read_as_raw_records: If True, ignore the payload type of `examples`. Always
      use RawTfRecord TFXIO.
    raw_record_column_name: If provided, the arrow RecordBatch produced by
      the TFXIO will contain a string column of the given name, and the contents
      of that column will be the raw records. Note that not all TFXIO supports
      this option, and an error will be raised in that case. Required if
      read_as_raw_records == True.

  Returns:
    A function that takes a file pattern as input and returns a TFXIO
    instance.

  Raises:
    NotImplementedError: when given an unsupported example payload type.
  """
  assert examples.type is standard_artifacts.Examples, (
      'examples must be of type standard_artifacts.Examples')
  # In case that the payload format custom property is not set.
  # Assume tf.Example.
  payload_format = examples_utils.get_payload_format(examples)
  data_view_uri = None
  if payload_format == example_gen_pb2.PayloadFormat.FORMAT_PROTO:
    data_view_uri = examples.get_string_custom_property(
        constants.DATA_VIEW_URI_PROPERTY_KEY)
    if not data_view_uri:
      data_view_uri = None
  return lambda file_pattern: make_tfxio(  # pylint:disable=g-long-lambda
      file_pattern=file_pattern,
      telemetry_descriptors=telemetry_descriptors,
      payload_format=payload_format,
      data_view_uri=data_view_uri,
      schema=schema,
      read_as_raw_records=read_as_raw_records,
      raw_record_column_name=raw_record_column_name)


def make_tfxio(file_pattern: Text,
               telemetry_descriptors: List[Text],
               payload_format: Union[Text, int],
               data_view_uri: Optional[Text] = None,
               schema: Optional[schema_pb2.Schema] = None,
               read_as_raw_records: bool = False,
               raw_record_column_name: Optional[Text] = None):
  """Creates a TFXIO instance that reads `file_pattern`.

  Args:
    file_pattern: the file pattern for the TFXIO to access.
    telemetry_descriptors: A set of descriptors that identify the component
      that is instantiating the TFXIO. These will be used to construct the
      namespace to contain metrics for profiling and are therefore expected to
      be identifiers of the component itself and not individual instances of
      source use.
    payload_format: one of the enums from example_gen_pb2.PayloadFormat (may
      be in string or int form). If None, default to FORMAT_TF_EXAMPLE.
    data_view_uri: uri to a DataView artifact. A DataView is needed in order
      to create a TFXIO for certain payload formats.
    schema: TFMD schema. Note: although optional, some payload formats need a
      schema in order for all TFXIO interfaces (e.g. TensorAdapter()) to work.
      Unless you know what you are doing, always supply a schema.
    read_as_raw_records: If True, ignore the payload type of `examples`. Always
      use RawTfRecord TFXIO.
    raw_record_column_name: If provided, the arrow RecordBatch produced by
      the TFXIO will contain a string column of the given name, and the contents
      of that column will be the raw records. Note that not all TFXIO supports
      this option, and an error will be raised in that case. Required if
      read_as_raw_records == True.

  Returns:
    a TFXIO instance.
  """
  if not isinstance(payload_format, int):
    payload_format = example_gen_pb2.PayloadFormat.Value(payload_format)

  if read_as_raw_records:
    assert raw_record_column_name is not None, (
        'read_as_raw_records is specified - '
        'must provide raw_record_column_name')
    return raw_tf_record.RawTfRecordTFXIO(
        file_pattern=file_pattern,
        raw_record_column_name=raw_record_column_name,
        telemetry_descriptors=telemetry_descriptors)

  if payload_format == example_gen_pb2.PayloadFormat.FORMAT_TF_EXAMPLE:
    return tf_example_record.TFExampleRecord(
        file_pattern=file_pattern,
        schema=schema,
        raw_record_column_name=raw_record_column_name,
        telemetry_descriptors=telemetry_descriptors)

  if (payload_format ==
      example_gen_pb2.PayloadFormat.FORMAT_TF_SEQUENCE_EXAMPLE):
    return tf_sequence_example_record.TFSequenceExampleRecord(
        file_pattern=file_pattern,
        schema=schema,
        raw_record_column_name=raw_record_column_name,
        telemetry_descriptors=telemetry_descriptors)

  if payload_format == example_gen_pb2.PayloadFormat.FORMAT_PROTO:
    assert data_view_uri is not None, (
        'Accessing FORMAT_PROTO requires a DataView to parse the proto.')
    return record_to_tensor_tfxio.TFRecordToTensorTFXIO(
        file_pattern=file_pattern,
        saved_decoder_path=data_view_uri,
        telemetry_descriptors=telemetry_descriptors,
        raw_record_column_name=raw_record_column_name)

  raise NotImplementedError(
      'Unsupport payload format: {}'.format(payload_format))
