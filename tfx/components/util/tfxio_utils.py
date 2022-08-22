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

from typing import Any, Callable, Dict, List, Iterator, Optional, Tuple, Union

import pyarrow as pa
import tensorflow as tf
from tfx.components.experimental.data_view import constants
from tfx.components.util import examples_utils
from tfx.proto import example_gen_pb2
from tfx.types import artifact
from tfx.types import standard_artifacts
from tfx_bsl.tfxio import dataset_options
from tfx_bsl.tfxio import parquet_tfxio
from tfx_bsl.tfxio import raw_tf_record
from tfx_bsl.tfxio import record_to_tensor_tfxio
from tfx_bsl.tfxio import tf_example_record
from tfx_bsl.tfxio import tf_sequence_example_record
from tfx_bsl.tfxio import tfxio
from tensorflow_metadata.proto.v0 import schema_pb2

_SUPPORTED_FILE_FORMATS = (example_gen_pb2.FileFormat.FILE_FORMAT_PARQUET,
                           example_gen_pb2.FileFormat.FORMAT_TFRECORDS_GZIP)
# TODO(b/162532479): switch to support List[str] exclusively, once tfx-bsl
# post-0.22 is released.
OneOrMorePatterns = Union[str, List[str]]


def resolve_payload_format_and_data_view_uri(
    examples: List[artifact.Artifact]) -> Tuple[int, Optional[str]]:
  """Resolves the payload format and a DataView URI for given artifacts.

  This routine make sure that the provided list of Examples artifacts are of
  the same payload type, and if their payload type is FORMAT_PROTO, it resolves
  one DataView (if applicable) to be used to access the data in all the
  artifacts in a consistent way (i.e. the RecordBatches from those artifacts
  will have the same schema).

  Args:
    examples: A list of Examples artifact.
  Returns:
    A pair. The first term is the payload format (a value in
      example_gen_pb2.PayloadFormat enum); the second term is the URI to the
      resolved DataView (could be None, if the examples are not FORMAT_PROTO,
      or they are all FORMAT_PROTO, but all do not have a DataView attached).
  Raises:
    ValueError: if not all artifacts are of the same payload format, or
      if they are all of FORMAT_PROTO but some (but not all) of them do not
      have a DataView attached.
  """
  assert examples, 'At least one Examples artifact is needed.'
  payload_format = _get_payload_format(examples)

  if payload_format != example_gen_pb2.PayloadFormat.FORMAT_PROTO:
    # Only FORMAT_PROTO may have DataView attached.
    return payload_format, None

  data_view_infos = []
  for examples_artifact in examples:
    data_view_infos.append(_get_data_view_info(examples_artifact))
  # All the artifacts do not have DataView attached -- this is allowed. The
  # caller may be requesting to read the data as raw string records.
  if all([i is None for i in data_view_infos]):
    return payload_format, None

  # All the artifacts have a DataView attached -- resolve to the latest
  # DataView (the one with the largest create time). This will guarantee that
  # the RecordBatch read from each artifact will share the same Arrow schema
  # (and thus Tensors fed to TF graphs, if applicable). The DataView will need
  # to guarantee backward compatibilty with older spans. Usually the DataView
  # is a struct2tensor query, so such guarantee is provided by protobuf
  # (as long as the user follows the basic principles of making changes to
  # the proto).
  if all([i is not None for i in data_view_infos]):
    return payload_format, max(data_view_infos, key=lambda pair: pair[1])[0]

  violating_artifacts = [
      e for e, i in zip(examples, data_view_infos) if i is None]
  raise ValueError(
      'Unable to resolve a DataView for the Examples Artifacts '
      'provided -- some Artifacts did not have DataView attached: {}'
      .format(violating_artifacts))


def get_tfxio_factory_from_artifact(
    examples: List[artifact.Artifact],
    telemetry_descriptors: List[str],
    schema: Optional[schema_pb2.Schema] = None,
    read_as_raw_records: bool = False,
    raw_record_column_name: Optional[str] = None
) -> Callable[[OneOrMorePatterns], tfxio.TFXIO]:
  """Returns a factory function that creates a proper TFXIO.

  Args:
    examples: The Examples artifacts that the TFXIO is intended to access.
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

  payload_format, data_view_uri = resolve_payload_format_and_data_view_uri(
      examples)
  return lambda file_pattern: make_tfxio(  # pylint:disable=g-long-lambda
      file_pattern=file_pattern,
      telemetry_descriptors=telemetry_descriptors,
      payload_format=payload_format,
      data_view_uri=data_view_uri,
      schema=schema,
      read_as_raw_records=read_as_raw_records,
      raw_record_column_name=raw_record_column_name)


def get_tf_dataset_factory_from_artifact(
    examples: List[artifact.Artifact],
    telemetry_descriptors: List[str],
) -> Callable[[
    List[str],
    dataset_options.TensorFlowDatasetOptions,
    Optional[schema_pb2.Schema],
], tf.data.Dataset]:
  """Returns a factory function that creates a tf.data.Dataset.

  Args:
    examples: The Examples artifacts that the TFXIO from which the Dataset is
      created from is intended to access.
    telemetry_descriptors: A set of descriptors that identify the component
      that is instantiating the TFXIO. These will be used to construct the
      namespace to contain metrics for profiling and are therefore expected to
      be identifiers of the component itself and not individual instances of
      source use.
  """
  payload_format, data_view_uri = resolve_payload_format_and_data_view_uri(
      examples)

  def dataset_factory(file_pattern: List[str],
                      options: dataset_options.TensorFlowDatasetOptions,
                      schema: Optional[schema_pb2.Schema]) -> tf.data.Dataset:
    return make_tfxio(
        file_pattern=file_pattern,
        telemetry_descriptors=telemetry_descriptors,
        payload_format=payload_format,
        data_view_uri=data_view_uri,
        schema=schema).TensorFlowDataset(
            options)

  return dataset_factory


def get_record_batch_factory_from_artifact(
    examples: List[artifact.Artifact],
    telemetry_descriptors: List[str],
) -> Callable[[
    List[str],
    dataset_options.RecordBatchesOptions,
    Optional[schema_pb2.Schema],
], Iterator[pa.RecordBatch]]:
  """Returns a factory function that creates Iterator[pa.RecordBatch].

  Args:
    examples: The Examples artifacts that the TFXIO from which the Dataset is
      created from is intended to access.
    telemetry_descriptors: A set of descriptors that identify the component that
      is instantiating the TFXIO. These will be used to construct the namespace
      to contain metrics for profiling and are therefore expected to be
      identifiers of the component itself and not individual instances of source
      use.
  """
  payload_format, data_view_uri = resolve_payload_format_and_data_view_uri(
      examples)

  def record_batch_factory(
      file_pattern: List[str], options: dataset_options.RecordBatchesOptions,
      schema: Optional[schema_pb2.Schema]) -> Iterator[pa.RecordBatch]:
    return make_tfxio(
        file_pattern=file_pattern,
        telemetry_descriptors=telemetry_descriptors,
        payload_format=payload_format,
        data_view_uri=data_view_uri,
        schema=schema).RecordBatches(options)

  return record_batch_factory


def get_data_view_decode_fn_from_artifact(
    examples: List[artifact.Artifact],
    telemetry_descriptors: List[str],
) -> Optional[Callable[[tf.Tensor], Dict[str, Any]]]:
  """Returns the decode function wrapped in the examples' Data View.

  Args:
    examples: The Examples artifacts from which the data view is resolved.
    telemetry_descriptors: A set of descriptors that identify the component that
      is instantiating the TFXIO. These will be used to construct the namespace
      to contain metrics for profiling and are therefore expected to be
      identifiers of the component itself and not individual instances of source
      use.
  Returns:
    If a Data View can be resolved from `examples`, then it returns
    a TF Function that takes a 1-D string tensor (example records) and returns
    decoded (composite) tensors. Otherwise returns None.
  """
  payload_format, data_view_uri = resolve_payload_format_and_data_view_uri(
      examples)
  if (payload_format != example_gen_pb2.PayloadFormat.FORMAT_PROTO or
      data_view_uri is None):
    return None

  return record_to_tensor_tfxio.BeamRecordToTensorTFXIO(
      saved_decoder_path=data_view_uri,
      telemetry_descriptors=telemetry_descriptors,
      physical_format='data_view_access_only',
      raw_record_column_name=None).DecodeFunction()


# TODO(b/216604827): Deprecate str file format.
def _file_format_from_string(file_format: str) -> example_gen_pb2.FileFormat:
  if file_format == 'tfrecords_gzip':
    return example_gen_pb2.FileFormat.FORMAT_TFRECORDS_GZIP
  else:
    return example_gen_pb2.FileFormat.Value(file_format)


def make_tfxio(
    file_pattern: OneOrMorePatterns,
    telemetry_descriptors: List[str],
    payload_format: int,
    data_view_uri: Optional[str] = None,
    schema: Optional[schema_pb2.Schema] = None,
    read_as_raw_records: bool = False,
    raw_record_column_name: Optional[str] = None,
    file_format: Optional[Union[int, List[int], str, List[str]]] = None
) -> tfxio.TFXIO:
  """Creates a TFXIO instance that reads `file_pattern`.

  Args:
    file_pattern: the file pattern for the TFXIO to access.
    telemetry_descriptors: A set of descriptors that identify the component that
      is instantiating the TFXIO. These will be used to construct the namespace
      to contain metrics for profiling and are therefore expected to be
      identifiers of the component itself and not individual instances of source
      use.
    payload_format: one of the enums from example_gen_pb2.PayloadFormat (may be
      in string or int form). If None, default to FORMAT_TF_EXAMPLE.
    data_view_uri: uri to a DataView artifact. A DataView is needed in order to
      create a TFXIO for certain payload formats.
    schema: TFMD schema. Note: although optional, some payload formats need a
      schema in order for all TFXIO interfaces (e.g. TensorAdapter()) to work.
      Unless you know what you are doing, always supply a schema.
    read_as_raw_records: If True, ignore the payload type of `examples`. Always
      use RawTfRecord TFXIO.
    raw_record_column_name: If provided, the arrow RecordBatch produced by the
      TFXIO will contain a string column of the given name, and the contents of
      that column will be the raw records. Note that not all TFXIO supports this
      option, and an error will be raised in that case. Required if
      read_as_raw_records == True.
    file_format: file format for each file_pattern. Only 'tfrecords_gzip' and
      'parquet' are supported for now.

  Returns:
    a TFXIO instance.
  """
  if not isinstance(payload_format, int):
    payload_format = example_gen_pb2.PayloadFormat.Value(payload_format)

  if file_format is not None:
    if type(file_format) is not type(file_pattern):
      raise ValueError(
          f'The type of file_pattern and file_formats should be the same.'
          f'Given: file_pattern={file_pattern}, file_format={file_format}')
    if isinstance(file_format, list):
      if len(file_format) != len(file_pattern):
        raise ValueError(
            f'The length of file_pattern and file_formats should be the same.'
            f'Given: file_pattern={file_pattern}, file_format={file_format}')
      else:
        file_format = [_file_format_from_string(item) for item in file_format]
        if any(item not in _SUPPORTED_FILE_FORMATS for item in file_format):
          raise NotImplementedError(f'{file_format} is not supported yet.')
    else:  # file_format is str type.
      file_format = _file_format_from_string(file_format)
      if file_format not in _SUPPORTED_FILE_FORMATS:
        raise NotImplementedError(f'{file_format} is not supported yet.')

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

  if payload_format == example_gen_pb2.PayloadFormat.FORMAT_PARQUET:
    return parquet_tfxio.ParquetTFXIO(
        file_pattern=file_pattern,
        schema=schema,
        telemetry_descriptors=telemetry_descriptors)

  raise NotImplementedError(
      'Unsupport payload format: {}'.format(payload_format))


def _get_payload_format(examples: List[artifact.Artifact]) -> int:
  payload_formats = set(
      [examples_utils.get_payload_format(e) for e in examples])
  if len(payload_formats) != 1:
    raise ValueError('Unable to read example artifacts of different payload '
                     'formats: {}'.format(payload_formats))
  return payload_formats.pop()


def _get_data_view_info(
    examples: artifact.Artifact) -> Optional[Tuple[str, int]]:
  """Returns the payload format and data view URI and ID from examples."""
  assert examples.type is standard_artifacts.Examples, (
      'examples must be of type standard_artifacts.Examples')
  payload_format = examples_utils.get_payload_format(examples)
  if payload_format == example_gen_pb2.PayloadFormat.FORMAT_PROTO:
    data_view_uri = examples.get_string_custom_property(
        constants.DATA_VIEW_URI_PROPERTY_KEY)
    if data_view_uri:
      assert examples.has_custom_property(constants.DATA_VIEW_CREATE_TIME_KEY)
      # The creation time could be an int or str. Legacy artifacts will contain
      # an int custom property.
      data_view_create_time = examples.get_custom_property(
          constants.DATA_VIEW_CREATE_TIME_KEY)
      data_view_create_time = int(data_view_create_time)
      return data_view_uri, data_view_create_time

  return None
