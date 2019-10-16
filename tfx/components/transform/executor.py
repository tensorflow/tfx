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
"""Executor for TensorFlow Transform."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import absl
import apache_beam as beam
import numpy as np
import six
import tensorflow as tf
import tensorflow_data_validation as tfdv
from tensorflow_data_validation.utils import batch_util
import tensorflow_transform as tft
from tensorflow_transform import impl_helper
import tensorflow_transform.beam as tft_beam
from tensorflow_transform.beam import analyzer_cache
from tensorflow_transform.beam import common as tft_beam_common
from tensorflow_transform.saved import saved_transform_io
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import metadata_io
from tensorflow_transform.tf_metadata import schema_utils
from typing import Any, Dict, Generator, List, Mapping, Sequence, Text, Tuple, Union, Optional
# pylint: disable=g-direct-tensorflow-import
from tensorflow.core.example import example_pb2
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2
# pylint: enable=g-direct-tensorflow-import
from tfx import types
from tfx.components.base import base_executor
from tfx.components.transform import labels
from tfx.components.transform import messages
from tfx.components.util import value_utils
from tfx.types import artifact_utils
from tfx.utils import import_utils
from tfx.utils import io_utils


RAW_EXAMPLE_KEY = 'raw_example'

# Schema to use if the input data should be decoded as raw example.
_RAW_EXAMPLE_SCHEMA = schema_utils.schema_from_feature_spec(
    {RAW_EXAMPLE_KEY: tf.io.FixedLenFeature([], tf.string)})

# TODO(b/123519698): Simplify the code by removing the key structure.
_TRANSFORM_INTERNAL_FEATURE_FOR_KEY = '__TFT_PASS_KEY__'

# Default file name prefix for transformed_examples.
_DEFAULT_TRANSFORMED_EXAMPLES_PREFIX = 'transformed_examples'

# Temporary path inside transform_output used for tft.beam
# TODO(b/125451545): Provide a safe temp path from base executor instead.
_TEMP_DIR_IN_TRANSFORM_OUTPUT = '.temp_path'


# TODO(b/122478841): Move it to a common place that is shared across components.
class _Status(object):
  """Status that reports success or error status of an execution."""

  def __init__(self, is_error, error_message=None):
    self._is_error = is_error
    self._error_message = error_message

  @classmethod
  def OK(cls):
    """Returns an ok Status."""

    return _Status(False)

  @classmethod
  def Error(cls, error_message):
    """Returns an error Status with error message."""

    return _Status(True, error_message)

  @property
  def error_message(self):
    return self._error_message


class _Dataset(object):
  """Dataset to be analyzed and/or transformed.

  It also contains bundle of stages of a single dataset through the transform
  pipeline.
  """
  # TODO(b/37788560): This seems like a brittle way of creating dataset keys.
  # In particular there are no guarantees that there won't be colissions.
  # A better approach might be something like ArtifactID, or perhaps
  # SHA256(file_pattern) which might also be a lot less verbose (even if it
  # might not be as self-describing).
  _FILE_PATTERN_SUFFIX_LENGTH = 6

  def __init__(self, file_pattern: Text, file_format: Text, data_format: Text,
               metadata: dataset_metadata.DatasetMetadata,
               stats_output_path: Optional[Text] = None,
               materialize_output_path: Optional[Text] = None):
    """Initialize a Dataset.

    Args:
      file_pattern: The file pattern of the dataset.
      file_format: The file format of the dataset.
      data_format: The data format of the dataset.
      metadata: A DatasetMetadata object describing the dataset.
      stats_output_path: The file path where to write stats for the dataset.
      materialize_output_path: The file path where to write the dataset.
    """
    self._file_pattern = file_pattern
    file_pattern_suffix = os.path.join(
        *file_pattern.split(os.sep)[-self._FILE_PATTERN_SUFFIX_LENGTH:])
    self._dataset_key = analyzer_cache.make_dataset_key(file_pattern_suffix)
    self._file_format = file_format
    self._data_format = data_format
    self._metadata = metadata
    self._stats_output_path = stats_output_path
    self._materialize_output_path = materialize_output_path
    self._index = None

  @property
  def file_pattern(self):
    assert self._file_pattern
    return self._file_pattern

  @property
  def stats_output_path(self):
    assert self._stats_output_path
    return self._stats_output_path

  @property
  def materialize_output_path(self):
    assert self._materialize_output_path
    return self._materialize_output_path

  @property
  def index(self):
    assert self._index is not None
    return self._index

  @property
  def dataset_key(self):
    assert self._dataset_key
    return self._dataset_key

  @property
  def data_format(self):
    assert self._data_format
    return self._data_format

  @property
  def file_format(self):
    assert self._file_format
    return self._file_format

  @property
  def metadata(self):
    return self._metadata

  @property
  def encoded(self):
    return self._encoded

  @property
  def decoded(self):
    return self._decoded

  @property
  def transformed(self):
    return self._transformed

  # TODO(b/65115913): Remove this and the setter and instead chain the
  # "encoding" only to the "Materialize" parts of the computation, just
  # before (or within) _WriteExamples.
  @property
  def transformed_and_encoded(self):
    return self._transformed_and_encoded

  @index.setter
  def index(self, val):
    self._index = val

  @encoded.setter
  def encoded(self, val):
    self._encoded = val

  @decoded.setter
  def decoded(self, val):
    self._decoded = val

  @transformed.setter
  def transformed(self, val):
    self._transformed = val

  @transformed_and_encoded.setter
  def transformed_and_encoded(self, val):
    self._transformed_and_encoded = val


def _GetSchemaProto(
    metadata: dataset_metadata.DatasetMetadata) -> schema_pb2.Schema:
  """Gets the schema proto associated with a DatasetMetadata.

  This is needed because tensorflow_transform 0.13 and tensorflow_transform 0.14
  have a different API for DatasetMetadata.

  Args:
    metadata: A dataset_metadata.DatasetMetadata.

  Returns:
    A schema_pb2.Schema.
  """
  # `schema` is either a Schema proto or dataset_schema.Schema.
  schema = metadata.schema
  # In the case where it's a dataset_schema.Schema, fetch the schema proto.
  return getattr(schema, '_schema_proto', schema)


class Executor(base_executor.BaseExecutor):
  """Transform executor."""

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """TensorFlow Transform executor entrypoint.

    This implements BaseExecutor.Do() and is invoked by orchestration systems.
    This is not inteded for manual usage or further customization. Please use
    the Transform() function which takes an input format with no artifact
    dependency.

    Args:
      input_dict: Input dict from input key to a list of artifacts, including:
        - input_data: A list of 'ExamplesPath' type which should contain two
          splits 'train' and 'eval'.
        - schema: A list of 'SchemaPath' type which should contain a single
          schema artifact.
      output_dict: Output dict from key to a list of artifacts, including:
        - transform_output: Output of 'tf.Transform', which includes an exported
          Tensorflow graph suitable for both training and serving;
        - transformed_examples: Materialized transformed examples, which
          includes both 'train' and 'eval' splits.
      exec_properties: A dict of execution properties, including either one of:
        - module_file: The file path to a python module file, from which the
          'preprocessing_fn' function will be loaded.
        - preprocessing_fn: The module path to a python function that
          implements 'preprocessing_fn'.

    Returns:
      None
    """
    self._log_startup(input_dict, output_dict, exec_properties)
    train_data_uri = artifact_utils.get_split_uri(input_dict['input_data'],
                                                  'train')
    eval_data_uri = artifact_utils.get_split_uri(input_dict['input_data'],
                                                 'eval')
    schema_file = io_utils.get_only_uri_in_dir(
        artifact_utils.get_single_uri(input_dict['schema']))
    transform_output = artifact_utils.get_single_uri(
        output_dict['transform_output'])
    transformed_train_output = artifact_utils.get_split_uri(
        output_dict['transformed_examples'], 'train')
    transformed_eval_output = artifact_utils.get_split_uri(
        output_dict['transformed_examples'], 'eval')
    temp_path = os.path.join(transform_output, _TEMP_DIR_IN_TRANSFORM_OUTPUT)
    absl.logging.debug('Using temp path %s for tft.beam', temp_path)

    def _GetCachePath(label, params_dict):
      if label not in params_dict:
        return None
      else:
        return artifact_utils.get_single_uri(params_dict[label])

    label_inputs = {
        labels.COMPUTE_STATISTICS_LABEL:
            False,
        labels.SCHEMA_PATH_LABEL:
            schema_file,
        labels.EXAMPLES_DATA_FORMAT_LABEL:
            labels.FORMAT_TF_EXAMPLE,
        labels.ANALYZE_DATA_PATHS_LABEL:
            io_utils.all_files_pattern(train_data_uri),
        labels.ANALYZE_PATHS_FILE_FORMATS_LABEL:
            labels.FORMAT_TFRECORD,
        labels.TRANSFORM_DATA_PATHS_LABEL: [
            io_utils.all_files_pattern(train_data_uri),
            io_utils.all_files_pattern(eval_data_uri)
        ],
        labels.TRANSFORM_PATHS_FILE_FORMATS_LABEL: [
            labels.FORMAT_TFRECORD, labels.FORMAT_TFRECORD
        ],
        labels.TFT_STATISTICS_USE_TFDV_LABEL:
            True,
        labels.MODULE_FILE:
            exec_properties.get('module_file', None),
        labels.PREPROCESSING_FN:
            exec_properties.get('preprocessing_fn', None),
    }
    cache_input = _GetCachePath('cache_input_path', input_dict)
    if cache_input is not None:
      label_inputs[labels.CACHE_INPUT_PATH_LABEL] = cache_input

    label_outputs = {
        labels.TRANSFORM_METADATA_OUTPUT_PATH_LABEL: transform_output,
        labels.TRANSFORM_MATERIALIZE_OUTPUT_PATHS_LABEL: [
            os.path.join(transformed_train_output,
                         _DEFAULT_TRANSFORMED_EXAMPLES_PREFIX),
            os.path.join(transformed_eval_output,
                         _DEFAULT_TRANSFORMED_EXAMPLES_PREFIX),
        ],
        labels.TEMP_OUTPUT_LABEL: str(temp_path),
    }
    cache_output = _GetCachePath('cache_output_path', output_dict)
    if cache_output is not None:
      label_outputs[labels.CACHE_OUTPUT_PATH_LABEL] = cache_output
    status_file = 'status_file'  # Unused
    self.Transform(label_inputs, label_outputs, status_file)
    absl.logging.debug('Cleaning up temp path %s on executor success',
                       temp_path)
    io_utils.delete_dir(temp_path)

  @staticmethod
  @beam.ptransform_fn
  @beam.typehints.with_input_types(beam.Pipeline)
  @beam.typehints.with_output_types(beam.pvalue.PDone)
  def _IncrementColumnUsageCounter(pipeline: beam.Pipeline,
                                   total_columns_count: int,
                                   analyze_columns_count: int,
                                   transform_columns_count: int):
    """A beam PTransform to increment counters of column usage."""

    def _MakeAndIncrementCounters(unused_element):
      """Increment column usage counters."""
      del unused_element
      beam.metrics.Metrics.counter(
          tft_beam_common.METRICS_NAMESPACE,
          'total_columns_count').inc(total_columns_count)
      beam.metrics.Metrics.counter(
          tft_beam_common.METRICS_NAMESPACE,
          'analyze_columns_count').inc(analyze_columns_count)
      beam.metrics.Metrics.counter(
          tft_beam_common.METRICS_NAMESPACE,
          'transform_columns_count').inc(transform_columns_count)
      return None

    return (pipeline
            | 'CreateSole' >> beam.Create([None])
            | 'Count' >> beam.Map(_MakeAndIncrementCounters))

  @staticmethod
  @beam.ptransform_fn
  @beam.typehints.with_input_types(beam.Pipeline)
  # TODO(b/122478841): Obviate the bytes (key part).
  @beam.typehints.with_output_types(
      beam.typehints.KV[bytes, beam.typehints.Union[bytes, example_pb2.Example]]
  )
  def _ReadExamples(pipeline: beam.Pipeline,
                    dataset: _Dataset) -> beam.pvalue.PCollection:
    """Reads examples from the given `dataset`.

    Args:
      pipeline: beam pipeline.
      dataset: A `_Dataset` object that represents the data to read.

    Returns:
      A PCollection containing KV pairs of exapmles.
    """
    if dataset.file_format != labels.FORMAT_TFRECORD:
      raise ValueError('Unsupported input file format: {}'.format(
          dataset.file_format))

    result = (
        pipeline
        | 'Read' >> beam.io.ReadFromTFRecord(
            dataset.file_pattern,
            coder=beam.coders.BytesCoder(),
            # TODO(b/114938612): Eventually remove this override.
            validate=False)
        | 'AddKey' >> beam.Map(lambda x: (None, x)))
    if dataset.data_format == labels.FORMAT_TF_EXAMPLE:
      result |= (
          'ParseExamples' >>
          beam.Map(lambda kv: (kv[0], example_pb2.Example.FromString(kv[1]))))
    # TODO(b/122478841): Figure out telemetry in beam.
    return result

  @staticmethod
  @beam.ptransform_fn
  @beam.typehints.with_input_types(
      beam.typehints.KV[bytes, example_pb2.Example])
  @beam.typehints.with_output_types(beam.pvalue.PDone)
  def _WriteExamples(pcollection: beam.pvalue.PCollection, file_format: Text,
                     transformed_example_path: Text) -> beam.pvalue.PDone:
    """Writes transformed examples compressed in gzip format.

    Args:
      pcollection: PCollection of transformed examples.
      file_format: The output file format.
      transformed_example_path: path to write to.

    Returns:
      beam.pvalue.PDone.
    """
    if file_format != labels.FORMAT_TFRECORD:
      raise ValueError('Unsupported output file format: {}'.format(file_format))

    return (pcollection
            | 'DropNoneKeys' >> beam.Values()
            | 'Write' >> beam.io.WriteToTFRecord(
                transformed_example_path,
                file_name_suffix='.gz',
                coder=beam.coders.ProtoCoder(example_pb2.Example)))

  def _GetSchema(self, schema_path: Text) -> schema_pb2.Schema:
    """Gets a tf.metadata schema.

    Args:
      schema_path: Path to schema file.

    Returns:
      A tf.metadata schema.
    """
    schema_reader = io_utils.SchemaReader()
    return schema_reader.read(schema_path)

  def _ReadMetadata(self, data_format: Text,
                    schema_path: Text) -> dataset_metadata.DatasetMetadata:
    """Returns a dataset_metadata.DatasetMetadata for the input data.

    Args:
      data_format: name of the input data format.
      schema_path: path to schema file.

    Returns:
      A dataset_metadata.DatasetMetadata representing the provided set of
          columns.
    """

    if self._ShouldDecodeAsRawExample(data_format):
      return dataset_metadata.DatasetMetadata(_RAW_EXAMPLE_SCHEMA)
    schema_proto = self._GetSchema(schema_path)
    # For compatibility with tensorflow_transform 0.13 and 0.14, we create and
    # then update a DatasetMetadata.
    result = dataset_metadata.DatasetMetadata(dataset_schema.Schema({}))
    _GetSchemaProto(result).CopyFrom(schema_proto)
    return result

  @staticmethod
  @beam.ptransform_fn
  @beam.typehints.with_input_types(
      beam.typehints.Dict[str, beam.typehints.Any])  # TFDV format.
  @beam.typehints.with_output_types(beam.pvalue.PDone)
  def _GenerateStats(
      pcollection: beam.pvalue.PCollection,
      stats_output_path: Text,
      schema: schema_pb2.Schema,
      use_tfdv=True,
      use_deep_copy_optimization=False  # pylint: disable=unused-argument
  ) -> beam.pvalue.PDone:
    """Generates statistics.

    Args:
      pcollection: PCollection of examples.
      stats_output_path: path where statistics is written to.
      schema: schema.
      use_tfdv: whether use TFDV for computing statistics.
      use_deep_copy_optimization: whether use deep copy optimization.

    Returns:
      beam.pvalue.PDone.
    """
    if not use_tfdv:
      raise ValueError(
          'TFDV is not used for stats. Please provide althernatives.')

    # pylint: disable=no-value-for-parameter
    return (pcollection
            | 'ComputeTFDVStats' >> Executor._ComputeTFDVStats(schema)
            | 'WriteStats' >> Executor._WriteStats(stats_output_path))

  @staticmethod
  @beam.ptransform_fn
  @beam.typehints.with_input_types(beam.typehints.Dict[str, beam.typehints.Any])
  @beam.typehints.with_output_types(statistics_pb2.DatasetFeatureStatisticsList)
  def _ComputeTFDVStats(pcollection: beam.pvalue.PCollection,
                        schema: schema_pb2.Schema) -> beam.pvalue.PCollection:
    """Cmoputes Statistics with TFDV.

    Args:
      pcollection: pcollection of examples.
      schema: schema.

    Returns:
      PCollection of `DatasetFeatureStatisticsList`.
    """
    feature_specs_from_schema = schema_utils.schema_as_feature_spec(
        schema).feature_spec

    def EncodeTFDV(element, feature_specs):
      """Encodes element in an in-memory format that TFDV expects."""
      if _TRANSFORM_INTERNAL_FEATURE_FOR_KEY not in element:
        raise ValueError(
            'Expected _TRANSFORM_INTERNAL_FEATURE_FOR_KEY ({}) to exist in the '
            'input but not found.'.format(_TRANSFORM_INTERNAL_FEATURE_FOR_KEY))

      # TODO(b/123549935): Obviate the numpy array conversions by
      # allowing TFDV to accept primitives in general, and TFT's
      # input/output format in particular.
      result = {}
      for feature_name, feature_spec in six.iteritems(feature_specs):
        feature_value = element.get(feature_name)
        if feature_value is None:
          result[feature_name] = None
        elif isinstance(feature_value, np.ndarray):
          result[feature_name] = np.asarray(
              feature_value, feature_spec.dtype.as_numpy_dtype).reshape(-1)
        elif isinstance(feature_value, list):
          result[feature_name] = np.asarray(
              feature_value, feature_spec.dtype.as_numpy_dtype)
        else:
          result[feature_name] = np.asarray(
              [feature_value], dtype=feature_spec.dtype.as_numpy_dtype)
      return result

    # TODO(pachristopher): Remove encoding and batching steps once TFT
    # supports Arrow tables.
    # TODO(pachristopher): Explore if encoding TFT dict into serialized examples
    # and then converting them to Arrow tables is cheaper than converting to
    # TFDV dict and then to Arrow tables.
    result = (pcollection
              | 'EncodeTFDV' >> beam.Map(
                  EncodeTFDV, feature_specs=feature_specs_from_schema))

    return (
        result
        |
        'BatchExamplesToArrowTables' >> batch_util.BatchExamplesToArrowTables()
        | 'ComputeFeatureStatisticsTFDV' >> tfdv.GenerateStatistics(
            tfdv.StatsOptions(schema=schema)))

  @staticmethod
  @beam.ptransform_fn
  @beam.typehints.with_input_types(statistics_pb2.DatasetFeatureStatisticsList)
  @beam.typehints.with_output_types(beam.pvalue.PDone)
  def _WriteStats(pcollection_stats: beam.pvalue.PCollection,
                  stats_output_path: Text) -> beam.pvalue.PDone:
    """Writs Statistics outputs.

    Args:
      pcollection_stats: pcollection of statistics.
      stats_output_path: path to write statistics.

    Returns:
      beam.pvalue.PDone.
    """

    # TODO(b/68765333): Investigate if this can be avoided.
    tf.io.gfile.makedirs(os.path.dirname(stats_output_path))
    # TODO(b/117601471): Replace with utility method to write stats.
    return (pcollection_stats | 'Write' >> beam.io.WriteToText(
        stats_output_path,
        append_trailing_newlines=False,
        shard_name_template='',  # To force unsharded output.
        coder=beam.coders.ProtoCoder(
            statistics_pb2.DatasetFeatureStatisticsList)))

  @staticmethod
  @beam.ptransform_fn
  @beam.typehints.with_input_types(
      beam.typehints.KV[bytes, beam.typehints.Union[bytes, example_pb2.Example]]
  )
  @beam.typehints.with_output_types(
      beam.typehints.Dict[str, beam.typehints.Any])
  def _DecodeInputs(pcol: beam.pvalue.PCollection,
                    decode_fn: Any) -> beam.pvalue.PCollection:
    """Decodes the given PCollection while handling KV data.

    Args:
      pcol: PCollection of data.
      decode_fn: Function used to decode data.

    Returns:
      PCollection of decoded data.
    """

    def decode_example(
        kv_pair: Mapping[bytes, Union[bytes, example_pb2.Example]]
    ) -> Mapping[Text, Any]:  # pylint: disable=invalid-name
      """Decodes a single example."""
      (key, elem) = kv_pair
      result = decode_fn(elem)
      if _TRANSFORM_INTERNAL_FEATURE_FOR_KEY in result:
        raise ValueError('"{}" is a reserved feature name, '
                         'it should not be present in the dataset.'.format(
                             _TRANSFORM_INTERNAL_FEATURE_FOR_KEY))
      result[_TRANSFORM_INTERNAL_FEATURE_FOR_KEY] = key
      return result

    return pcol | 'ApplyDecodeFn' >> beam.Map(decode_example)

  @beam.typehints.with_input_types(
      beam.typehints.Dict[str, beam.typehints.Any], metadata=beam.typehints.Any)
  @beam.typehints.with_output_types(
      beam.typehints.KV[beam.typehints.Union[None, bytes], example_pb2.Example])
  class _EncodeAsExamples(beam.DoFn):
    """Encodes data as tf.Examples based on the given metadata."""

    def __init__(self):
      self._coder = None

    def process(self, element: Dict[Text, Any],
                metadata: Any) -> Generator[Tuple[Any, Any], None, None]:
      if self._coder is None:
        self._coder = tft.coders.ExampleProtoCoder(
            metadata.schema, serialized=False)

      # Make sure that the synthetic key feature doesn't get encoded.
      assert _TRANSFORM_INTERNAL_FEATURE_FOR_KEY in element
      key = element[_TRANSFORM_INTERNAL_FEATURE_FOR_KEY]
      element_copy = element.copy()
      del element_copy[_TRANSFORM_INTERNAL_FEATURE_FOR_KEY]
      yield (key, self._coder.encode(element_copy))

  @staticmethod
  @beam.ptransform_fn
  @beam.typehints.with_input_types(beam.Pipeline)
  def _OptimizeRun(
      pipeline: beam.Pipeline, input_cache_dir: Text, output_cache_dir: Text,
      analyze_data_list: List[_Dataset], feature_spec: Mapping[Text, Any],
      preprocessing_fn: Any, cache_source: beam.PTransform
  ) -> Tuple[Dict[Text, Optional[_Dataset]], Dict[Text, Dict[
      Text, beam.pvalue.PCollection]], bool]:
    """Utilizes TFT cache if applicable and removes unused datasets."""

    dataset_keys_list = [dataset.dataset_key for dataset in analyze_data_list]
    if input_cache_dir is not None:
      input_cache = (
          pipeline
          | 'ReadCache' >> analyzer_cache.ReadAnalysisCacheFromFS(
              input_cache_dir, dataset_keys_list, source=cache_source))
    elif output_cache_dir is not None:
      input_cache = {}
    else:
      # Using None here to indicate that this pipeline will not read or write
      # cache.
      input_cache = None

    if input_cache is None:
      # Cache is disabled so we won't be filtering out any datasets, and will
      # always perform a flatten over all of them.
      filtered_analysis_dataset_keys = dataset_keys_list
      flat_data_required = True
    else:
      filtered_analysis_dataset_keys, flat_data_required = (
          tft_beam.analysis_graph_builder.get_analysis_dataset_keys(
              preprocessing_fn, feature_spec, dataset_keys_list, input_cache))

    new_analyze_data_dict = {}
    for dataset in analyze_data_list:
      if dataset.dataset_key in filtered_analysis_dataset_keys:
        new_analyze_data_dict[dataset.dataset_key] = dataset
      else:
        new_analyze_data_dict[dataset.dataset_key] = None

    return (new_analyze_data_dict, input_cache, flat_data_required)

  def _GetPreprocessingFn(self, inputs: Mapping[Text, Any],
                          unused_outputs: Mapping[Text, Any]) -> Any:
    """Returns a user defined preprocessing_fn.

    Args:
      inputs: A dictionary of labelled input values.
      unused_outputs: A dictionary of labelled output values.

    Returns:
      User defined function.

    Raises:
      ValueError: When neither or both of MODULE_FILE and PREPROCESSING_FN
        are present in inputs.
    """
    has_module_file = bool(
        value_utils.GetSoleValue(inputs, labels.MODULE_FILE, strict=False))
    has_preprocessing_fn = bool(
        value_utils.GetSoleValue(inputs, labels.PREPROCESSING_FN, strict=False))

    if has_module_file == has_preprocessing_fn:
      raise ValueError(
          'Neither or both of MODULE_FILE and PREPROCESSING_FN have been '
          'supplied in inputs.')

    if has_module_file:
      return import_utils.import_func_from_source(
          value_utils.GetSoleValue(inputs, labels.MODULE_FILE),
          'preprocessing_fn')

    preprocessing_fn_path_split = value_utils.GetSoleValue(
        inputs, labels.PREPROCESSING_FN).split('.')
    return import_utils.import_func_from_module(
        '.'.join(preprocessing_fn_path_split[0:-1]),
        preprocessing_fn_path_split[-1])

  # TODO(b/122478841): Refine this API in following cls.
  # Note: This API is up to change.
  def Transform(self, inputs: Mapping[Text, Any], outputs: Mapping[Text, Any],
                status_file: Text) -> None:
    """Executes on request.

    This is the implementation part of transform executor. This is intended for
    using or extending the executor without artifact dependency.

    Args:
      inputs: A dictionary of labelled input values, including:
        - labels.COMPUTE_STATISTICS_LABEL: Whether compute statistics.
        - labels.SCHEMA_PATH_LABEL: Path to schema file.
        - labels.EXAMPLES_DATA_FORMAT_LABEL: Example data format.
        - labels.ANALYZE_DATA_PATHS_LABEL: Paths or path patterns to analyze
          data.
        - labels.ANALYZE_PATHS_FILE_FORMATS_LABEL: File formats of paths to
          analyze data.
        - labels.TRANSFORM_DATA_PATHS_LABEL: Paths or path patterns to transform
          data.
        - labels.TRANSFORM_PATHS_FILE_FORMATS_LABEL: File formats of paths to
          transform data.
        - labels.TFT_STATISTICS_USE_TFDV_LABEL: Whether use tfdv to compute
          statistics.
        - labels.MODULE_FILE: Path to a Python module that contains the
          preprocessing_fn, optional.
        - labels.PREPROCESSING_FN: Path to a Python function that implements
          preprocessing_fn, optional.
      outputs: A dictionary of labelled output values, including:
        - labels.PER_SET_STATS_OUTPUT_PATHS_LABEL: Paths to statistics output,
          optional.
        - labels.TRANSFORM_METADATA_OUTPUT_PATH_LABEL: A path to
          TFTransformOutput output.
        - labels.TRANSFORM_MATERIALIZE_OUTPUT_PATHS_LABEL: Paths to transform
          materialization.
        - labels.TEMP_OUTPUT_LABEL: A path to temporary directory.
      status_file: Where the status should be written (not yet implemented)
    """

    del status_file  # unused
    compute_statistics = value_utils.GetSoleValue(
        inputs, labels.COMPUTE_STATISTICS_LABEL)
    transform_output_path = value_utils.GetSoleValue(
        outputs, labels.TRANSFORM_METADATA_OUTPUT_PATH_LABEL)
    raw_examples_data_format = value_utils.GetSoleValue(
        inputs, labels.EXAMPLES_DATA_FORMAT_LABEL)
    schema = value_utils.GetSoleValue(inputs, labels.SCHEMA_PATH_LABEL)
    input_dataset_metadata = self._ReadMetadata(raw_examples_data_format,
                                                schema)

    absl.logging.debug(
        'Inputs to executor.Transform function: {}'.format(inputs))
    absl.logging.debug(
        'Outputs to executor.Transform function: {}'.format(outputs))

    feature_spec = schema_utils.schema_as_feature_spec(
        _GetSchemaProto(input_dataset_metadata)).feature_spec

    # NOTE: We disallow an empty schema, which we detect by testing the
    # number of columns.  While in principal an empty schema is valid, in
    # practice this is a sign of a user error, and this is a convenient
    # place to catch that error.
    if (not feature_spec and
        not self._ShouldDecodeAsRawExample(raw_examples_data_format)):
      raise ValueError(messages.SCHEMA_EMPTY)

    preprocessing_fn = self._GetPreprocessingFn(inputs, outputs)

    materialize_output_paths = value_utils.GetValues(
        outputs, labels.TRANSFORM_MATERIALIZE_OUTPUT_PATHS_LABEL)

    # Inspecting the preprocessing_fn even if we know we need a full pass in
    # order to fail faster if it fails.
    analyze_input_columns = tft.get_analyze_input_columns(
        preprocessing_fn, feature_spec)

    if not compute_statistics and not materialize_output_paths:
      if analyze_input_columns:
        absl.logging.warning(
            'Not using the in-place Transform because the following features '
            'require analyzing: {}'.format(
                tuple(c for c in analyze_input_columns)))
      else:
        absl.logging.warning(
            'Using the in-place Transform since compute_statistics=False, '
            'it does not materialize transformed data, and the configured '
            'preprocessing_fn appears to not require analyzing the data.')
        self._RunInPlaceImpl(preprocessing_fn, input_dataset_metadata,
                             transform_output_path)
        # TODO(b/122478841): Writes status to status file.
        return
    self._RunBeamImpl(inputs, outputs, preprocessing_fn, input_dataset_metadata,
                      raw_examples_data_format, transform_output_path,
                      compute_statistics, materialize_output_paths)
    # TODO(b/122478841): Writes status to status file.

  def _RunBeamImpl(self, inputs: Mapping[Text, Any],
                   outputs: Mapping[Text, Any], preprocessing_fn: Any,
                   input_dataset_metadata: dataset_metadata.DatasetMetadata,
                   raw_examples_data_format: Text, transform_output_path: Text,
                   compute_statistics: bool,
                   materialize_output_paths: Sequence[Text]) -> _Status:
    """Perform data preprocessing with FlumeC++ runner.

    Args:
      inputs: A dictionary of labelled input values.
      outputs: A dictionary of labelled output values.
      preprocessing_fn: The tf.Transform preprocessing_fn.
      input_dataset_metadata: A DatasetMetadata object for the input data.
      raw_examples_data_format: A string describing the raw data format.
      transform_output_path: An absolute path to write the output to.
      compute_statistics: A bool indicating whether or not compute statistics.
      materialize_output_paths: Paths to materialized outputs.

    Raises:
      RuntimeError: If reset() is not being invoked between two run().
      ValueError: If the schema is empty.

    Returns:
      Status of the execution.
    """
    analyze_data_paths = value_utils.GetValues(inputs,
                                               labels.ANALYZE_DATA_PATHS_LABEL)
    analyze_paths_file_formats = value_utils.GetValues(
        inputs, labels.ANALYZE_PATHS_FILE_FORMATS_LABEL)
    transform_data_paths = value_utils.GetValues(
        inputs, labels.TRANSFORM_DATA_PATHS_LABEL)
    transform_paths_file_formats = value_utils.GetValues(
        inputs, labels.TRANSFORM_PATHS_FILE_FORMATS_LABEL)
    stats_use_tfdv = value_utils.GetSoleValue(
        inputs, labels.TFT_STATISTICS_USE_TFDV_LABEL)
    per_set_stats_output_paths = value_utils.GetValues(
        outputs, labels.PER_SET_STATS_OUTPUT_PATHS_LABEL)
    temp_path = value_utils.GetSoleValue(outputs, labels.TEMP_OUTPUT_LABEL)

    input_cache_dir = value_utils.GetSoleValue(
        inputs, labels.CACHE_INPUT_PATH_LABEL, strict=False)
    output_cache_dir = value_utils.GetSoleValue(
        outputs, labels.CACHE_OUTPUT_PATH_LABEL, strict=False)

    absl.logging.debug('Analyze data patterns: %s',
                       list(enumerate(analyze_data_paths)))
    absl.logging.debug('Transform data patterns: %s',
                       list(enumerate(transform_data_paths)))
    absl.logging.debug('Transform materialization output paths: %s',
                       list(enumerate(materialize_output_paths)))
    absl.logging.debug('Transform output path: %s', transform_output_path)

    if len(analyze_data_paths) != len(analyze_paths_file_formats):
      return _Status.Error(
          'size of analyze_data_paths and '
          'analyze_paths_file_formats do not match: {} v.s {}'.format(
              len(analyze_data_paths), len(analyze_paths_file_formats)))
    if len(transform_data_paths) != len(transform_paths_file_formats):
      return _Status.Error(
          'size of transform_data_paths and '
          'transform_paths_file_formats do not match: {} v.s {}'.format(
              len(transform_data_paths), len(transform_paths_file_formats)))

    feature_spec = schema_utils.schema_as_feature_spec(
        _GetSchemaProto(input_dataset_metadata)).feature_spec
    analyze_input_columns = (
        tft.get_analyze_input_columns(preprocessing_fn, feature_spec))
    transform_input_columns = (
        tft.get_transform_input_columns(preprocessing_fn, feature_spec))
    # Use the same dataset (same columns) for AnalyzeDataset and computing
    # pre-transform stats so that the data will only be read once for these
    # two operations.
    if compute_statistics:
      analyze_input_columns = list(
          set(list(analyze_input_columns) + list(transform_input_columns)))
    if input_dataset_metadata.schema is _RAW_EXAMPLE_SCHEMA:
      analyze_input_dataset_metadata = input_dataset_metadata
      transform_input_dataset_metadata = input_dataset_metadata
    else:
      analyze_input_dataset_metadata = dataset_metadata.DatasetMetadata(
          schema_utils.schema_from_feature_spec({
              feature: feature_spec[feature]
              for feature in analyze_input_columns
          }))
      transform_input_dataset_metadata = dataset_metadata.DatasetMetadata(
          schema_utils.schema_from_feature_spec({
              feature: feature_spec[feature]
              for feature in transform_input_columns
          }))

    can_process_analysis_jointly = not bool(output_cache_dir)
    analyze_data_list = self._MakeDatasetList(analyze_data_paths,
                                              analyze_paths_file_formats,
                                              raw_examples_data_format,
                                              analyze_input_dataset_metadata,
                                              can_process_analysis_jointly)

    can_process_transform_jointly = not bool(per_set_stats_output_paths or
                                             materialize_output_paths)
    transform_data_list = self._MakeDatasetList(
        transform_data_paths, transform_paths_file_formats,
        raw_examples_data_format, transform_input_dataset_metadata,
        can_process_transform_jointly, per_set_stats_output_paths,
        materialize_output_paths)

    desired_batch_size = self._GetDesiredBatchSize(raw_examples_data_format)

    with self._CreatePipeline(outputs) as p:
      with tft_beam.Context(
          temp_dir=temp_path,
          desired_batch_size=desired_batch_size,
          passthrough_keys={_TRANSFORM_INTERNAL_FEATURE_FOR_KEY},
          use_deep_copy_optimization=True):
        # pylint: disable=expression-not-assigned
        # pylint: disable=no-value-for-parameter
        _ = (
            p |
            'IncrementColumnUsageCounter' >> self._IncrementColumnUsageCounter(
                len(feature_spec.keys()), len(analyze_input_columns),
                len(transform_input_columns)))

        (new_analyze_data_dict, input_cache, flat_data_required) = (
            p | 'OptimizeRun' >> self._OptimizeRun(
                input_cache_dir, output_cache_dir, analyze_data_list,
                feature_spec, preprocessing_fn, self._GetCacheSource()))

        if input_cache:
          absl.logging.debug('Analyzing data with cache.')

        full_analyze_dataset_keys_list = [
            dataset.dataset_key for dataset in analyze_data_list
        ]

        # Removing unneeded datasets if they won't be needed for statistics or
        # materialization.
        if not materialize_output_paths and not compute_statistics:
          if None in new_analyze_data_dict.values():
            absl.logging.debug(
                'Not reading the following datasets due to cache: %s', [
                    dataset.file_pattern
                    for dataset in analyze_data_list
                    if new_analyze_data_dict[dataset.dataset_key] is None
                ])
          analyze_data_list = [
              d for d in new_analyze_data_dict.values() if d is not None
          ]

        analyze_decode_fn = (
            self._GetDecodeFunction(raw_examples_data_format,
                                    analyze_input_dataset_metadata.schema))

        for dataset in analyze_data_list:
          infix = 'AnalysisIndex{}'.format(dataset.index)
          dataset.encoded = (
              p
              | 'ReadDataset[{}]'.format(infix) >> self._ReadExamples(dataset))
          dataset.decoded = (
              dataset.encoded | 'Decode[{}]'.format(infix) >>
              self._DecodeInputs(analyze_decode_fn))

        input_analysis_data = {}
        for key, dataset in six.iteritems(new_analyze_data_dict):
          if dataset is None:
            input_analysis_data[key] = None
          else:
            input_analysis_data[key] = dataset.decoded

        if flat_data_required:
          flat_input_analysis_data = (
              [dataset.decoded for dataset in analyze_data_list]
              | 'FlattenAnalysisDatasetsBecauseItIsRequired'
              >> beam.Flatten(pipeline=p))
        else:
          flat_input_analysis_data = None

        transform_fn, cache_output = (
            (flat_input_analysis_data, input_analysis_data, input_cache,
             input_dataset_metadata)
            | 'Analyze' >> tft_beam.AnalyzeDatasetWithCache(
                preprocessing_fn, pipeline=p))

        # Write the raw/input metadata.
        (input_dataset_metadata
         | 'WriteMetadata' >> tft_beam.WriteMetadata(
             os.path.join(transform_output_path,
                          tft.TFTransformOutput.RAW_METADATA_DIR), p))

        # WriteTransformFn writes transform_fn and metadata to subdirectories
        # tensorflow_transform.SAVED_MODEL_DIR and
        # tensorflow_transform.TRANSFORMED_METADATA_DIR respectively.
        (transform_fn |
         'WriteTransformFn' >> tft_beam.WriteTransformFn(transform_output_path))

        if output_cache_dir is not None and cache_output is not None:
          # TODO(b/37788560): Possibly make this part of the beam graph.
          tf.io.gfile.makedirs(output_cache_dir)
          absl.logging.debug('Using existing cache in: %s', input_cache_dir)
          if input_cache_dir is not None:
            # Only copy cache that is relevant to this iteration. This is
            # assuming that this pipeline operates on rolling ranges, so those
            # cache entries may also be relevant for future iterations.
            for span_cache_dir in input_analysis_data:
              full_span_cache_dir = os.path.join(input_cache_dir,
                                                 span_cache_dir)
              if tf.io.gfile.isdir(full_span_cache_dir):
                self._CopyCache(full_span_cache_dir,
                                os.path.join(output_cache_dir, span_cache_dir))

          # TODO(b/138934800): Use dataset_keys directly once TFT 0.15 is
          # released.
          write_analysis_cache_kwargs = dict(
              pipeline=p,
              cache_base_dir=output_cache_dir,
              sink=self._GetCacheSink())
          if tft.__version__ > '0.14.0':
            write_analysis_cache_kwargs['dataset_keys'] = (
                full_analyze_dataset_keys_list)
          (cache_output
           | 'WriteCache' >>
           analyzer_cache.WriteAnalysisCacheToFS(**write_analysis_cache_kwargs))

        if compute_statistics or materialize_output_paths:
          # Do not compute pre-transform stats if the input format is raw proto,
          # as StatsGen would treat any input as tf.Example.
          if (compute_statistics and
              not self._IsDataFormatProto(raw_examples_data_format)):
            # Aggregated feature stats before transformation.
            pre_transform_feature_stats_path = os.path.join(
                transform_output_path,
                tft.TFTransformOutput.PRE_TRANSFORM_FEATURE_STATS_PATH)

            schema_proto = _GetSchemaProto(analyze_input_dataset_metadata)
            ([
                dataset.decoded if stats_use_tfdv else dataset.encoded
                for dataset in analyze_data_list
            ]
             | 'FlattenAnalysisDatasets' >> beam.Flatten(pipeline=p)
             | 'GenerateStats[FlattenedAnalysisDatasets]' >>
             self._GenerateStats(
                 pre_transform_feature_stats_path,
                 schema_proto,
                 use_deep_copy_optimization=True,
                 use_tfdv=stats_use_tfdv))

          transform_decode_fn = (
              self._GetDecodeFunction(raw_examples_data_format,
                                      transform_input_dataset_metadata.schema))
          # transform_data_list is a superset of analyze_data_list, we pay the
          # cost to read the same dataset (analyze_data_list) again here to
          # prevent certain beam runner from doing large temp materialization.
          for dataset in transform_data_list:
            infix = 'TransformIndex{}'.format(dataset.index)
            dataset.encoded = (
                p |
                'ReadDataset[{}]'.format(infix) >> self._ReadExamples(dataset))
            dataset.decoded = (
                dataset.encoded | 'Decode[{}]'.format(infix) >>
                self._DecodeInputs(transform_decode_fn))
            (dataset.transformed, metadata) = (
                ((dataset.decoded, transform_input_dataset_metadata),
                 transform_fn)
                | 'Transform[{}]'.format(infix) >> tft_beam.TransformDataset())

            if materialize_output_paths or not stats_use_tfdv:
              dataset.transformed_and_encoded = (
                  dataset.transformed | 'Encode[{}]'.format(infix) >>
                  beam.ParDo(self._EncodeAsExamples(), metadata))

          if compute_statistics:
            # Aggregated feature stats after transformation.
            _, metadata = transform_fn
            post_transform_feature_stats_path = os.path.join(
                transform_output_path,
                tft.TFTransformOutput.POST_TRANSFORM_FEATURE_STATS_PATH)

            # TODO(b/70392441): Retain tf.Metadata (e.g., IntDomain) in
            # schema. Currently input dataset schema only contains dtypes,
            # and other metadata is dropped due to roundtrip to tensors.
            transformed_schema_proto = _GetSchemaProto(metadata)

            ([(dataset.transformed
               if stats_use_tfdv else dataset.transformed_and_encoded)
              for dataset in transform_data_list]
             | 'FlattenTransformedDatasets' >> beam.Flatten()
             | 'GenerateStats[FlattenedTransformedDatasets]' >>
             self._GenerateStats(
                 post_transform_feature_stats_path,
                 transformed_schema_proto,
                 use_tfdv=stats_use_tfdv))

            if per_set_stats_output_paths:
              # TODO(b/67632871): Remove duplicate stats gen compute that is
              # done both on a flattened view of the data, and on each span
              # below.
              for dataset in transform_data_list:
                infix = 'TransformIndex{}'.format(dataset.index)
                if stats_use_tfdv:
                  data = dataset.transformed
                else:
                  data = dataset.transformed_and_encoded
                data | 'GenerateStats[{}]'.format(infix) >> self._GenerateStats(
                    dataset.stats_output_path,
                    transformed_schema_proto,
                    use_tfdv=stats_use_tfdv)

          if materialize_output_paths:
            for dataset in transform_data_list:
              infix = 'TransformIndex{}'.format(dataset.index)
              (dataset.transformed_and_encoded
               | 'Materialize[{}]'.format(infix) >> self._WriteExamples(
                   transform_paths_file_formats[-1],
                   dataset.materialize_output_path))

    return _Status.OK()

  def _RunInPlaceImpl(self, preprocessing_fn: Any,
                      metadata: dataset_metadata.DatasetMetadata,
                      transform_output_path: Text) -> _Status:
    """Runs a transformation iteration in-place without looking at the data.

    Args:
      preprocessing_fn: The tf.Transform preprocessing_fn.
      metadata: A DatasetMetadata object for the input data.
      transform_output_path: An absolute path to write the output to.

    Returns:
      Status of the execution.
    """

    absl.logging.debug('Processing an in-place transform')

    raw_metadata_dir = os.path.join(transform_output_path,
                                    tft.TFTransformOutput.RAW_METADATA_DIR)
    metadata_io.write_metadata(metadata, raw_metadata_dir)

    with tf.compat.v1.Graph().as_default() as graph:
      with tf.compat.v1.Session(graph=graph) as sess:

        input_signature = impl_helper.feature_spec_as_batched_placeholders(
            schema_utils.schema_as_feature_spec(
                _GetSchemaProto(metadata)).feature_spec)

        # In order to avoid a bug where import_graph_def fails when the
        # input_map and return_elements of an imported graph are the same
        # (b/34288791), we avoid using the placeholder of an input column as an
        # output of a graph. We do this by applying tf.identity to all inputs of
        # the preprocessing_fn.  Note this applies at the level of raw tensors.
        # TODO(b/34288791): Remove this workaround and use a shallow copy of
        # inputs instead.  A shallow copy is needed in case
        # self._preprocessing_fn mutates its input.
        copied_inputs = impl_helper.copy_tensors(input_signature)

        output_signature = preprocessing_fn(copied_inputs)
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.tables_initializer())
        transform_fn_path = os.path.join(transform_output_path,
                                         tft.TFTransformOutput.TRANSFORM_FN_DIR)
        saved_transform_io.write_saved_transform_from_session(
            sess, input_signature, output_signature, transform_fn_path)

        transformed_metadata = dataset_metadata.DatasetMetadata(
            schema=tft.schema_inference.infer_feature_schema(
                output_signature, graph, sess))

    transformed_metadata_dir = os.path.join(
        transform_output_path, tft.TFTransformOutput.TRANSFORMED_METADATA_DIR)
    metadata_io.write_metadata(transformed_metadata, transformed_metadata_dir)

    return _Status.OK()

  def _CreatePipeline(self,
                      unused_outputs: Mapping[Text, Any]) -> beam.Pipeline:
    """Creates beam pipeline.

    Args:
      unused_outputs: A dictionary of labelled output values.

    Returns:
      Beam pipeline.
    """
    return self._make_beam_pipeline()

  # TODO(b/114444977): Remove the unused can_process_jointly argument.
  def _MakeDatasetList(
      self,
      file_patterns: Sequence[Text],
      file_formats: Sequence[Text],
      data_format: Text,
      metadata: dataset_metadata.DatasetMetadata,
      can_process_jointly: bool,
      stats_output_paths: Optional[Sequence[Text]] = None,
      materialize_output_paths: Optional[Sequence[Text]] = None
  ) -> List[_Dataset]:
    """Makes a list of Dataset from the given `file_patterns`.

    Args:
      file_patterns: A list of file patterns where each pattern corresponds to
        one `_Dataset`.
      file_formats: A list of file format where each format corresponds to one
        `_Dataset`. Must have the same size as `file_patterns`.
      data_format: The data format of the datasets.
      metadata: A DatasetMetadata object for the datasets.
      can_process_jointly: Whether paths can be processed jointly, unused.
      stats_output_paths: The statistics output paths, if applicable.
      materialize_output_paths: The materialization output paths, if applicable.

    Returns:
      A list of `_Dataset` sorted by their dataset_key property.
    """
    assert len(file_patterns) == len(file_formats)
    if stats_output_paths:
      assert len(file_patterns) == len(stats_output_paths)
    else:
      stats_output_paths = [None] * len(file_patterns)
    if materialize_output_paths:
      assert len(file_patterns) == len(materialize_output_paths)
    else:
      materialize_output_paths = [None] * len(file_patterns)

    datasets = [
        _Dataset(p, f, data_format, metadata, s, m)
        for p, f, s, m in zip(file_patterns, file_formats,
                              stats_output_paths, materialize_output_paths)
    ]
    result = sorted(datasets, key=lambda dataset: dataset.dataset_key)
    for index, dataset in enumerate(result):
      dataset.index = index
    return result

  @staticmethod
  def _ShouldDecodeAsRawExample(data_format: Text) -> bool:
    """Returns true if data format should be decoded as raw example.

    Args:
      data_format: name of data format.

    Returns:
      True if data format should be decoded as raw example.
    """
    return (Executor._IsDataFormatSequenceExample(data_format) or
            Executor._IsDataFormatProto(data_format))

  @staticmethod
  def _IsDataFormatSequenceExample(data_format: Text) -> bool:
    """Returns true if data format is sequence example.

    Args:
      data_format: name of data format.

    Returns:
      True if data format is sequence example.
    """
    return data_format == labels.FORMAT_TF_SEQUENCE_EXAMPLE

  @staticmethod
  def _IsDataFormatProto(data_format: Text) -> bool:
    """Returns true if data format is protocol buffer.

    Args:
      data_format: name of data format.

    Returns:
      True if data format is protocol buffer.
    """
    return data_format == labels.FORMAT_PROTO

  def _GetDesiredBatchSize(self, data_format: Text) -> Any:
    """Returns batch size.

    Args:
      data_format: name of data format.

    Returns:
      Batch size or None.
    """
    if self._IsDataFormatSequenceExample(data_format):
      return 1
    return None

  @staticmethod
  def _DecodeAsRawExample(serialized_examples):
    return {RAW_EXAMPLE_KEY: serialized_examples}

  def _GetDecodeFunction(self, data_format: Text,
                         schema: dataset_schema.Schema) -> Any:
    """Returns the decode function for `data_format`.

    Args:
      data_format: name of data format.
      schema: a dataset_schema.Schema for the data.

    Returns:
      Function for decoding examples.
    """

    if self._ShouldDecodeAsRawExample(data_format):
      if self._IsDataFormatSequenceExample(data_format):
        absl.logging.warning(
            'TFX Transform doesn\'t officially support tf.SequenceExample, '
            'follow b/38235367 to track official support progress. We do not '
            'guarantee not to break your pipeline if you use Transform with a '
            'tf.SequenceExample data type. Use at your own risk.')
      return self._DecodeAsRawExample

    # TODO(b/122478841): Eventually make it always serialize.
    return tft.coders.ExampleProtoCoder(schema, serialized=False).decode

  @staticmethod
  def _GetCacheSource():
    return None

  @staticmethod
  def _GetCacheSink():
    return None

  @staticmethod
  def _CopyCache(src, dst):
    # TODO(b/37788560): Make this more efficient.
    io_utils.copy_dir(src, dst)
