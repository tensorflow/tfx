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

import copy
import os
import apache_beam as beam
import numpy as np
import six
import tensorflow as tf
import tensorflow_data_validation as tfdv
import tensorflow_transform as tft
from tensorflow_transform import impl_helper
import tensorflow_transform.beam as tft_beam
from tensorflow_transform.saved import saved_transform_io
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import metadata_io
from tensorflow_transform.tf_metadata import schema_utils
from typing import Any, Dict, Generator, List, Mapping, Sequence, Text, Tuple, Union
# pylint: disable=g-direct-tensorflow-import
from tensorflow.core.example import example_pb2
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2
# pylint: enable=g-direct-tensorflow-import
from tfx.components.base import base_executor
from tfx.components.transform import common
from tfx.components.transform import labels
from tfx.components.transform import messages
from tfx.utils import io_utils
from tfx.utils import types


RAW_EXAMPLE_KEY = 'raw_example'

# Schema to use if the input data should be decoded as raw example.
_RAW_EXAMPLE_SCHEMA = dataset_schema.from_feature_spec(
    {RAW_EXAMPLE_KEY: tf.FixedLenFeature([], tf.string)})

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

  def __init__(self, file_pattern, file_format, data_format,
               metadata):
    """Initialize a Dataset.

    Args:
      file_pattern: The file pattern of the dataset.
      file_format: The file format of the dataset.
      data_format: The data format of the dataset.
      metadata: A DatasetMetadata object describing the dataset.
    """
    self._file_pattern = file_pattern
    self._file_format = file_format
    self._data_format = data_format
    self._metadata = metadata

  @property
  def file_pattern(self):
    return self._file_pattern

  @property
  def data_format(self):
    return self._data_format

  @property
  def file_format(self):
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


class Executor(base_executor.BaseExecutor):
  """Transform executor."""

  def Do(self, input_dict,
         output_dict,
         exec_properties):
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
      exec_properties: A dict of execution properties, including:
        - module_file: The file path to a python module file, from which the
          'preprocessing_fn' function will be loaded.

    Returns:
      None
    """
    self._log_startup(input_dict, output_dict, exec_properties)
    train_data_uri = types.get_split_uri(input_dict['input_data'], 'train')
    eval_data_uri = types.get_split_uri(input_dict['input_data'], 'eval')
    schema_file = io_utils.get_only_uri_in_dir(
        types.get_single_uri(input_dict['schema']))

    transform_output = types.get_single_uri(output_dict['transform_output'])
    if tf.gfile.Exists(transform_output):
      io_utils.delete_dir(transform_output)

    transformed_train_output = types.get_split_uri(
        output_dict['transformed_examples'], 'train')
    if tf.gfile.Exists(transformed_train_output):
      io_utils.delete_dir(transformed_train_output)

    transformed_eval_output = types.get_split_uri(
        output_dict['transformed_examples'], 'eval')
    if tf.gfile.Exists(transformed_eval_output):
      io_utils.delete_dir(transformed_eval_output)

    temp_path = os.path.join(transform_output, _TEMP_DIR_IN_TRANSFORM_OUTPUT)
    tf.logging.debug('Using temp path %s for tft.beam', temp_path)

    label_inputs = {
        labels.COMPUTE_STATISTICS_LABEL:
            False,
        labels.SCHEMA_PATH_LABEL:
            schema_file,
        labels.EXAMPLES_DATA_FORMAT_LABEL:
            labels.FORMAT_TF_EXAMPLE,
        labels.ANALYZE_AND_TRANSFORM_DATA_PATHS_LABEL:
            io_utils.all_files_pattern(train_data_uri),
        labels.TRANSFORM_ONLY_DATA_PATHS_LABEL:
            io_utils.all_files_pattern(eval_data_uri),
        labels.TFT_STATISTICS_USE_TFDV_LABEL:
            True,
        labels.PREPROCESSING_FN:
            exec_properties['module_file'],
    }

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
    status_file = 'status_file'  # Unused
    self.Transform(label_inputs, label_outputs, status_file)
    tf.logging.info('Cleaning up temp path %s on executor success', temp_path)
    io_utils.delete_dir(temp_path)

  @staticmethod
  @beam.ptransform_fn
  @beam.typehints.with_input_types(beam.Pipeline)
  # TODO(b/122478841): Obviate the bytes (key part).
  @beam.typehints.with_output_types(
      beam.typehints.KV[bytes, beam.typehints.Union[bytes, example_pb2.Example]]
  )
  def _ReadExamples(pipeline,
                    dataset):
    """Reads examples from the given `dataset`.

    Args:
      pipeline: beam pipeline.
      dataset: A `_Dataset` object that represents the data to read.

    Returns:
      A PCollection containing KV pairs of exapmles.
    """

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
  def _WriteExamples(pcollection,
                     unused_file_format,
                     transformed_example_path):
    """Writes transformed examples compressed in gzip format.

    Args:
      pcollection: PCollection of transformed examples.
      unused_file_format: file format, unused.
      transformed_example_path: path to write to.

    Returns:
      beam.pvalue.PDone.
    """
    return (pcollection
            | 'DropNoneKeys' >> beam.Values()
            | 'Write' >> beam.io.WriteToTFRecord(
                transformed_example_path,
                file_name_suffix='.gz',
                coder=beam.coders.ProtoCoder(example_pb2.Example)))

  def _GetSchema(self, schema_path):
    """Gets a tf.metadata schema.

    Args:
      schema_path: Path to schema file.

    Returns:
      A tf.metadata schema.
    """
    schema_reader = io_utils.SchemaReader()
    return schema_reader.read(schema_path)

  def _ReadSchema(self, data_format,
                  schema_path):
    """Returns a TFT schema for the input data.

    Args:
      data_format: name of the input data format.
      schema_path: path to schema file.

    Returns:
      A schema representing the provided set of columns.
    """

    if self._ShouldDecodeAsRawExample(data_format):
      return _RAW_EXAMPLE_SCHEMA
    schema = self._GetSchema(schema_path)
    # TODO(b/77351671): Remove this conversion to tf.Transform's internal
    # schema format.
    feature_spec = schema_utils.schema_as_feature_spec(schema).feature_spec
    return dataset_schema.from_feature_spec(feature_spec)

  @staticmethod
  @beam.ptransform_fn
  @beam.typehints.with_input_types(
      beam.typehints.Dict[str, beam.typehints.Any])  # TFDV format.
  @beam.typehints.with_output_types(beam.pvalue.PDone)
  def _GenerateStats(
      pcollection,
      stats_output_path,
      schema,
      use_tfdv=True,
      use_deep_copy_optimization=False  # pylint: disable=unused-argument
  ):
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
  def _ComputeTFDVStats(pcollection,
                        schema):
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
        elif isinstance(feature_value, (np.ndarray, list)):
          result[feature_name] = np.asarray(
              feature_value, feature_spec.dtype.as_numpy_dtype)
        else:
          result[feature_name] = np.asarray(
              [feature_value], dtype=feature_spec.dtype.as_numpy_dtype)

      return result

    return (pcollection
            | 'EncodeTFDV' >> beam.Map(
                EncodeTFDV, feature_specs=feature_specs_from_schema)
            | 'ComputeFeatureStatisticsTFDV' >> tfdv.GenerateStatistics(
                tfdv.StatsOptions(schema=schema)))

  @staticmethod
  @beam.ptransform_fn
  @beam.typehints.with_input_types(statistics_pb2.DatasetFeatureStatisticsList)
  @beam.typehints.with_output_types(beam.pvalue.PDone)
  def _WriteStats(pcollection_stats,
                  stats_output_path):
    """Writs Statistics outputs.

    Args:
      pcollection_stats: pcollection of statistics.
      stats_output_path: path to write statistics.

    Returns:
      beam.pvalue.PDone.
    """

    # TODO(b/68765333): Investigate if this can be avoided.
    tf.gfile.MakeDirs(os.path.dirname(stats_output_path))
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
  def _DecodeInputs(pcol,
                    decode_fn):
    """Decodes the given PCollection while handling KV data.

    Args:
      pcol: PCollection of data.
      decode_fn: Function used to decode data.

    Returns:
      PCollection of decoded data.
    """

    def decode_example(
        kv_pair
    ):  # pylint: disable=invalid-name
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

    def process(self, element,
                metadata):
      if self._coder is None:
        self._coder = tft.coders.ExampleProtoCoder(
            metadata.schema, serialized=False)

      # Make sure that the synthetic key feature doesn't get encoded.
      assert _TRANSFORM_INTERNAL_FEATURE_FOR_KEY in element
      key = element[_TRANSFORM_INTERNAL_FEATURE_FOR_KEY]
      element_copy = element.copy()
      del element_copy[_TRANSFORM_INTERNAL_FEATURE_FOR_KEY]
      yield (key, self._coder.encode(element_copy))

  def _GetPreprocessingFn(self, inputs,
                          unused_outputs):
    """Returns a user defined preprocessing_fn.

    Args:
      inputs: A dictionary of labelled input values.
      unused_outputs: A dictionary of labelled output values.

    Returns:
      User defined function.
    """
    return io_utils.import_func(
        common.GetSoleValue(inputs, labels.PREPROCESSING_FN),
        'preprocessing_fn')

  # TODO(b/122478841): Refine this API in following cls.
  # Note: This API is up to change.
  def Transform(self, inputs, outputs,
                status_file):
    """Executes on request.

    This is the implementation part of transform executor. This is intended for
    using or extending the executor without artifact dependency.

    Args:
      inputs: A dictionary of labelled input values, including:
        - labels.COMPUTE_STATISTICS_LABEL: Whether compute statistics.
        - labels.SCHEMA_PATH_LABEL: Path to schema file.
        - labels.EXAMPLES_FILE_FORMAT_LABEL: Example file format, optional.
        - labels.EXAMPLES_DATA_FORMAT_LABEL: Example data format.
        - labels.ANALYZE_AND_TRANSFORM_DATA_PATHS_LABEL: Paths or path patterns
          to analyze and transform data.
        - labels.TRANSFORM_DATA_PATHS_LABEL: Paths or path patterns to transform
          only data.
        - labels.TFT_STATISTICS_USE_TFDV_LABEL: Whether use tfdv to compute
          statistics.
        - labels.PREPROCESSING_FN: Path to a Python module that contains the
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
    compute_statistics = common.GetSoleValue(inputs,
                                             labels.COMPUTE_STATISTICS_LABEL)
    transform_output_path = common.GetSoleValue(
        outputs, labels.TRANSFORM_METADATA_OUTPUT_PATH_LABEL)
    raw_examples_data_format = common.GetSoleValue(
        inputs, labels.EXAMPLES_DATA_FORMAT_LABEL)
    schema = common.GetSoleValue(inputs, labels.SCHEMA_PATH_LABEL)
    input_dataset_schema = self._ReadSchema(raw_examples_data_format, schema)
    input_dataset_metadata = dataset_metadata.DatasetMetadata(
        input_dataset_schema)

    tf.logging.info('Inputs to executor.Transform function: {}'.format(inputs))
    tf.logging.info(
        'Outputs to executor.Transform function: {}'.format(outputs))

    # NOTE: We disallow an empty schema, which we detect by testing the
    # number of columns.  While in principal an empty schema is valid, in
    # practice this is a sign of a user error, and this is a convenient
    # place to catch that error.
    if (not input_dataset_metadata.schema.as_feature_spec() and
        not self._ShouldDecodeAsRawExample(raw_examples_data_format)):
      raise ValueError(messages.SCHEMA_EMPTY)

    preprocessing_fn = self._GetPreprocessingFn(inputs, outputs)

    materialize_output_paths = common.GetValues(
        outputs, labels.TRANSFORM_MATERIALIZE_OUTPUT_PATHS_LABEL)

    feature_spec = input_dataset_metadata.schema.as_feature_spec()

    # Inspecting the preprocessing_fn even if we know we need a full pass in
    # order to fail faster if it fails.
    try:
      analyze_input_columns = tft.get_analyze_input_columns(
          preprocessing_fn, feature_spec)
    except AttributeError:
      # If using TFT 1.12, fall back to assuming all features are used.
      analyze_input_columns = feature_spec.keys()

    if not compute_statistics and not materialize_output_paths:
      if analyze_input_columns:
        tf.logging.warning(
            'Not using the in-place Transform because the following features '
            'require analyzing: {}'.format(
                tuple(c for c in analyze_input_columns)))
      else:
        tf.logging.warning(
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

  def _RunBeamImpl(self, inputs,
                   outputs, preprocessing_fn,
                   input_dataset_metadata,
                   raw_examples_data_format, transform_output_path,
                   compute_statistics,
                   materialize_output_paths):
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
    raw_examples_file_format = common.GetSoleValue(
        inputs, labels.EXAMPLES_FILE_FORMAT_LABEL, strict=False)
    analyze_and_transform_data_paths = common.GetValues(
        inputs, labels.ANALYZE_AND_TRANSFORM_DATA_PATHS_LABEL)
    transform_only_data_paths = common.GetValues(
        inputs, labels.TRANSFORM_ONLY_DATA_PATHS_LABEL)
    stats_use_tfdv = common.GetSoleValue(inputs,
                                         labels.TFT_STATISTICS_USE_TFDV_LABEL)
    per_set_stats_output_paths = common.GetValues(
        outputs, labels.PER_SET_STATS_OUTPUT_PATHS_LABEL)
    temp_path = common.GetSoleValue(outputs, labels.TEMP_OUTPUT_LABEL)

    tf.logging.info('Analyze and transform data patterns: %s',
                    list(enumerate(analyze_and_transform_data_paths)))
    tf.logging.info('Transform data patterns: %s',
                    list(enumerate(transform_only_data_paths)))
    tf.logging.info('Transform materialization output paths: %s',
                    list(enumerate(materialize_output_paths)))
    tf.logging.info('Transform output path: %s', transform_output_path)

    feature_spec = input_dataset_metadata.schema.as_feature_spec()
    try:
      analyze_input_columns = tft.get_analyze_input_columns(
          preprocessing_fn, feature_spec)
      transform_input_columns = (
          tft.get_transform_input_columns(preprocessing_fn, feature_spec))
    except AttributeError:
      # If using TFT 1.12, fall back to assuming all features are used.
      analyze_input_columns = feature_spec.keys()
      transform_input_columns = feature_spec.keys()
    # Use the same dataset (same columns) for AnalyzeDataset and computing
    # pre-transform stats so that the data will only be read once for these
    # two operations.
    if compute_statistics:
      analyze_input_columns = list(
          set(list(analyze_input_columns) + list(transform_input_columns)))
    analyze_input_dataset_metadata = copy.deepcopy(input_dataset_metadata)
    transform_input_dataset_metadata = copy.deepcopy(input_dataset_metadata)
    if input_dataset_metadata.schema is not _RAW_EXAMPLE_SCHEMA:
      analyze_input_dataset_metadata.schema = dataset_schema.from_feature_spec(
          {feature: feature_spec[feature] for feature in analyze_input_columns})
      transform_input_dataset_metadata.schema = (
          dataset_schema.from_feature_spec({
              feature: feature_spec[feature]
              for feature in transform_input_columns
          }))

    can_process_jointly = not bool(per_set_stats_output_paths or
                                   materialize_output_paths)
    analyze_data_list = self._MakeDatasetList(
        analyze_and_transform_data_paths, raw_examples_file_format,
        raw_examples_data_format, analyze_input_dataset_metadata,
        can_process_jointly)
    transform_data_list = self._MakeDatasetList(
        list(analyze_and_transform_data_paths) +
        list(transform_only_data_paths), raw_examples_file_format,
        raw_examples_data_format, transform_input_dataset_metadata,
        can_process_jointly)

    desired_batch_size = self._GetDesiredBatchSize(raw_examples_data_format)

    with self._CreatePipeline(outputs) as p:
      with tft_beam.Context(
          temp_dir=temp_path,
          desired_batch_size=desired_batch_size,
          passthrough_keys={_TRANSFORM_INTERNAL_FEATURE_FOR_KEY},
          use_deep_copy_optimization=True):
        # pylint: disable=expression-not-assigned
        # pylint: disable=no-value-for-parameter

        analyze_decode_fn = (
            self._GetDecodeFunction(raw_examples_data_format,
                                    analyze_input_dataset_metadata.schema))

        for (idx, dataset) in enumerate(analyze_data_list):
          dataset.encoded = (
              p | 'ReadAnalysisDataset[{}]'.format(idx) >>
              self._ReadExamples(dataset))
          dataset.decoded = (
              dataset.encoded
              | 'DecodeAnalysisDataset[{}]'.format(idx) >>
              self._DecodeInputs(analyze_decode_fn))

        input_analysis_data = (
            [dataset.decoded for dataset in analyze_data_list]
            | 'FlattenAnalysisDatasets' >> beam.Flatten())
        transform_fn = (
            (input_analysis_data, input_dataset_metadata)
            | 'AnalyzeDataset' >> tft_beam.AnalyzeDataset(preprocessing_fn))
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

        if compute_statistics or materialize_output_paths:
          # Do not compute pre-transform stats if the input format is raw proto,
          # as StatsGen would treat any input as tf.Example.
          if (compute_statistics and
              not self._IsDataFormatProto(raw_examples_data_format)):
            # Aggregated feature stats before transformation.
            pre_transform_feature_stats_path = os.path.join(
                transform_output_path,
                tft.TFTransformOutput.PRE_TRANSFORM_FEATURE_STATS_PATH)

            # TODO(b/70392441): Retain tf.Metadata (e.g., IntDomain) in
            # schema. Currently input dataset schema only contains dtypes,
            # and other metadata is dropped due to roundtrip to tensors.
            schema_proto = schema_utils.schema_from_feature_spec(
                analyze_input_dataset_metadata.schema.as_feature_spec())
            ([
                dataset.decoded if stats_use_tfdv else dataset.encoded
                for dataset in analyze_data_list
            ]
             | 'FlattenPreTransformAnalysisDatasets' >> beam.Flatten()
             | 'GenerateAggregatePreTransformAnalysisStats' >>
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
          for (idx, dataset) in enumerate(transform_data_list):
            dataset.encoded = (
                p
                | 'ReadTransformDataset[{}]'.format(idx) >>
                self._ReadExamples(dataset))
            dataset.decoded = (
                dataset.encoded
                | 'DecodeTransformDataset[{}]'.format(idx) >>
                self._DecodeInputs(transform_decode_fn))
            (dataset.transformed,
             metadata) = (((dataset.decoded, transform_input_dataset_metadata),
                           transform_fn)
                          | 'TransformDataset[{}]'.format(idx) >>
                          tft_beam.TransformDataset())

            if materialize_output_paths or not stats_use_tfdv:
              dataset.transformed_and_encoded = (
                  dataset.transformed
                  | 'EncodeTransformedDataset[{}]'.format(idx) >> beam.ParDo(
                      self._EncodeAsExamples(), metadata))

          if compute_statistics:
            # Aggregated feature stats after transformation.
            _, metadata = transform_fn
            post_transform_feature_stats_path = os.path.join(
                transform_output_path,
                tft.TFTransformOutput.POST_TRANSFORM_FEATURE_STATS_PATH)

            # TODO(b/70392441): Retain tf.Metadata (e.g., IntDomain) in
            # schema. Currently input dataset schema only contains dtypes,
            # and other metadata is dropped due to roundtrip to tensors.
            transformed_schema_proto = schema_utils.schema_from_feature_spec(
                metadata.schema.as_feature_spec())

            ([(dataset.transformed
               if stats_use_tfdv else dataset.transformed_and_encoded)
              for dataset in transform_data_list]
             | 'FlattenPostTransformAnalysisDatasets' >> beam.Flatten()
             | 'GenerateAggregatePostTransformAnalysisStats' >>
             self._GenerateStats(
                 post_transform_feature_stats_path,
                 transformed_schema_proto,
                 use_tfdv=stats_use_tfdv))

            if per_set_stats_output_paths:
              assert len(transform_data_list) == len(per_set_stats_output_paths)
              # TODO(b/67632871): Remove duplicate stats gen compute that is
              # done both on a flattened view of the data, and on each span
              # below.
              bundles = zip(transform_data_list, per_set_stats_output_paths)
              for (idx, (dataset, output_path)) in enumerate(bundles):
                if stats_use_tfdv:
                  data = dataset.transformed
                else:
                  data = dataset.transformed_and_encoded
                (data
                 | 'GeneratePostTransformStats[{}]'.format(idx) >>
                 self._GenerateStats(
                     output_path,
                     transformed_schema_proto,
                     use_tfdv=stats_use_tfdv))

          if materialize_output_paths:
            assert len(transform_data_list) == len(materialize_output_paths)
            bundles = zip(transform_data_list, materialize_output_paths)
            for (idx, (dataset, output_path)) in enumerate(bundles):
              (dataset.transformed_and_encoded
               | 'Materialize[{}]'.format(idx) >> self._WriteExamples(
                   raw_examples_file_format, output_path))

    return _Status.OK()

  def _RunInPlaceImpl(self, preprocessing_fn,
                      metadata,
                      transform_output_path):
    """Runs a transformation iteration in-place without looking at the data.

    Args:
      preprocessing_fn: The tf.Transform preprocessing_fn.
      metadata: A DatasetMetadata object for the input data.
      transform_output_path: An absolute path to write the output to.

    Returns:
      Status of the execution.
    """

    tf.logging.info('Processing an in-place transform')

    raw_metadata_dir = os.path.join(transform_output_path,
                                    tft.TFTransformOutput.RAW_METADATA_DIR)
    metadata_io.write_metadata(metadata, raw_metadata_dir)

    with tf.Graph().as_default() as graph:
      with tf.Session(graph=graph) as sess:

        input_signature = impl_helper.feature_spec_as_batched_placeholders(
            metadata.schema.as_feature_spec())

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
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
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
                      unused_outputs):
    """Creates beam pipeline.

    Args:
      unused_outputs: A dictionary of labelled output values.

    Returns:
      Beam pipeline.
    """

    # TODO(b/122478841): Consider making beam pipeline part of context to
    # support fusion.
    return beam.Pipeline(argv=self._get_beam_pipeline_args())

  # TODO(b/114444977): Remove the unused_can_process_jointly argument and
  # perhaps the need for this entire function.
  def _MakeDatasetList(self, file_patterns, file_format,
                       data_format,
                       metadata,
                       unused_can_process_jointly):
    """Makes a list of Dataset from the given `file_patterns`.

    Args:
      file_patterns: A list of file patterns where each pattern corresponds to
        one `_Dataset`.
      file_format: The file format of the datasets.
      data_format: The data format of the datasets.
      metadata: A DatasetMetadata object for the datasets.
      unused_can_process_jointly: Whether paths can be processed jointly,
        unused.

    Returns:
      A list of `_Dataset`.
    """

    # File patterns will need to be processed independently.
    return [
        _Dataset(p, file_format, data_format, metadata) for p in file_patterns
    ]

  @staticmethod
  def _ShouldDecodeAsRawExample(data_format):
    """Returns true if data format should be decoded as raw example.

    Args:
      data_format: name of data format.

    Returns:
      True if data format should be decoded as raw example.
    """
    return (Executor._IsDataFormatSequenceExample(data_format) or
            Executor._IsDataFormatProto(data_format))

  @staticmethod
  def _IsDataFormatSequenceExample(data_format):
    """Returns true if data format is sequence example.

    Args:
      data_format: name of data format.

    Returns:
      True if data format is sequence example.
    """
    return data_format == labels.FORMAT_TF_SEQUENCE_EXAMPLE

  @staticmethod
  def _IsDataFormatProto(data_format):
    """Returns true if data format is protocol buffer.

    Args:
      data_format: name of data format.

    Returns:
      True if data format is protocol buffer.
    """
    return data_format == labels.FORMAT_PROTO

  def _GetDesiredBatchSize(self, data_format):
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

  def _GetDecodeFunction(self, data_format,
                         schema):
    """Returns the decode function for `data_format`.

    Args:
      data_format: name of data format.
      schema: a dataset_schema.Schema for the data.

    Returns:
      Function for decoding examples.
    """

    if self._ShouldDecodeAsRawExample(data_format):
      if self._IsDataFormatSequenceExample(data_format):
        tf.logging.warning(
            'TFX Transform doesn\'t officially support tf.SequenceExample, '
            'follow b/38235367 to track official support progress. We do not '
            'guarantee not to break your pipeline if you use Transform with a '
            'tf.SequenceExample data type. Use at your own risk.')
      return self._DecodeAsRawExample

    # TODO(b/122478841): Eventually make it always serialize.
    return tft.coders.ExampleProtoCoder(schema, serialized=False).decode
