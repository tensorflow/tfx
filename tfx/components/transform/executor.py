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

import hashlib
import os
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

from absl import logging
import apache_beam as beam
import pyarrow as pa
import tensorflow as tf
import tensorflow_data_validation as tfdv
import tensorflow_transform as tft
from tensorflow_transform import impl_helper
import tensorflow_transform.beam as tft_beam
from tensorflow_transform.beam import analyzer_cache
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import metadata_io
from tensorflow_transform.tf_metadata import schema_utils
from tfx import types
from tfx.components.transform import executor_utils
from tfx.components.transform import labels
from tfx.components.transform import stats_options_util
from tfx.components.util import examples_utils
from tfx.components.util import udf_utils
from tfx.components.util import value_utils
from tfx.components.util import tfxio_utils
from tfx.dsl.components.base import base_beam_executor
from tfx.dsl.components.base import base_executor
from tfx.dsl.io import fileio
from tfx.proto import example_gen_pb2
from tfx.types import artifact_utils
from tfx.types import standard_component_specs
from tfx.utils import io_utils
import tfx_bsl
from tfx_bsl.tfxio import tfxio as tfxio_module

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import anomalies_pb2
from tensorflow_metadata.proto.v0 import schema_pb2


# Key for temp path, for internal use only.
TEMP_PATH_KEY = 'temp_path'

RAW_EXAMPLE_KEY = 'raw_example'

# Schema to use if the input data should be decoded as raw example.
_RAW_EXAMPLE_SCHEMA = schema_utils.schema_from_feature_spec(
    {RAW_EXAMPLE_KEY: tf.io.FixedLenFeature([], tf.string)})

# TODO(b/123519698): Simplify the code by removing the key structure.
_TRANSFORM_INTERNAL_FEATURE_FOR_KEY = '__TFT_PASS_KEY__'

# Temporary path inside transform_output used for tft.beam
# TODO(b/125451545): Provide a safe temp path from base executor instead.
_TEMP_DIR_IN_TRANSFORM_OUTPUT = '.temp_path'

_TRANSFORM_COMPONENT_DESCRIPTOR = 'Transform'
_TELEMETRY_DESCRIPTORS = [_TRANSFORM_COMPONENT_DESCRIPTOR]

# TODO(b/37788560): Increase this max, based on results of experimentation with
# many non-packable analyzers on our benchmarks.
_MAX_ESTIMATED_STAGES_COUNT = 20000

# Beam extra pip package prefix.
_BEAM_EXTRA_PACKAGE_PREFIX = '--extra_package='

# Stats output filename keys.
_ANOMALIES_FILE = 'SchemaDiff.pb'
STATS_FILE = 'FeatureStats.pb'
SAMPLE_FILE_NAME = 'Sample.rio'
# TODO(b/215448985): Move these to a shared location with StatsGen.
_SHARDED_OUTPUT_PARTITIONS = 10
SHARDED_STATS_PREFIX = 'FeatureStats.rio'

_SCHEMA_FILE = 'schema.pbtxt'

_ANOMALIES_KEY = 'anomalies'
_SCHEMA_KEY = 'schema'
_STATS_KEY = 'stats'
_SHARDED_STATS_KEY = 'sharded_stats'

_FILE_FORMAT_PARQUET = example_gen_pb2.FileFormat.Name(
    example_gen_pb2.FileFormat.FILE_FORMAT_PARQUET)


# TODO(b/122478841): Move it to a common place that is shared across components.
class _Status:
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


class _Dataset:
  """Dataset to be analyzed and/or transformed.

  It also contains bundle of stages of a single dataset through the transform
  pipeline.
  """
  _FILE_PATTERN_SUFFIX_LENGTH = 6

  def __init__(self,
               file_pattern: str,
               file_format: Union[str, int],
               data_format: int,
               data_view_uri: Optional[str],
               stats_output_path: Optional[str] = None,
               materialize_output_path: Optional[str] = None):
    """Initialize a Dataset.

    Args:
      file_pattern: The file pattern of the dataset.
      file_format: The file format of the dataset.
      data_format: The data format of the dataset. One of the enums from
        example_gen_pb2.PayloadFormat.
      data_view_uri: URI to the DataView used to parse the data.
      stats_output_path: The file path where to write stats for the dataset.
      materialize_output_path: The file path where to write the dataset.
    """
    self._file_pattern = file_pattern
    file_pattern_suffix = os.path.join(
        *file_pattern.split(os.sep)[-self._FILE_PATTERN_SUFFIX_LENGTH:])
    dataset_identifier = file_pattern_suffix + '-' + hashlib.sha256(
        file_pattern.encode()).hexdigest()
    self._dataset_key = analyzer_cache.DatasetKey(dataset_identifier)
    self._file_format = file_format
    self._data_format = data_format
    self._data_view_uri = data_view_uri
    self._stats_output_path = stats_output_path
    self._materialize_output_path = materialize_output_path
    self._index = None
    self._standardized = None
    self._transformed = None
    self._transformed_and_standardized = None
    self._tfxio = None

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
  def data_view_uri(self):
    return self._data_view_uri

  @property
  def file_format(self):
    assert self._file_format
    return self._file_format

  @property
  def standardized(self):
    assert self._standardized is not None
    return self._standardized

  @property
  def transformed(self):
    assert self._transformed is not None
    return self._transformed

  @property
  def transformed_and_standardized(self):
    assert self._transformed_and_standardized is not None
    return self._transformed_and_standardized

  @property
  def tfxio(self):
    assert self._tfxio is not None
    return self._tfxio

  @index.setter
  def index(self, val):
    self._index = val

  @standardized.setter
  def standardized(self, val):
    self._standardized = val

  @transformed.setter
  def transformed(self, val):
    self._transformed = val

  @transformed_and_standardized.setter
  def transformed_and_standardized(self, val):
    self._transformed_and_standardized = val

  @tfxio.setter
  def tfxio(self, val):
    self._tfxio = val


def _InvokeStatsOptionsUpdaterFn(
    stats_options_updater_fn: Callable[
        [stats_options_util.StatsType, tfdv.StatsOptions], tfdv.StatsOptions],
    stats_type: stats_options_util.StatsType,
    schema: Optional[schema_pb2.Schema] = None,
    asset_map: Optional[Dict[str, str]] = None,
    transform_output_path: Optional[str] = None) -> tfdv.StatsOptions:
  """Invokes the provided stats_options_updater_fn.

  Args:
    stats_options_updater_fn: The function to call.
    stats_type: The stats_type use in the function call.
    schema: The input schema to use in the function call.
    asset_map: A dictionary containing key to filename mappings.
    transform_output_path: The path to the transform output.

  Returns:
    The updated tfdv.StatsOptions.
  """
  options = {}
  if schema is not None:
    schema_copy = schema_pb2.Schema()
    schema_copy.CopyFrom(schema)
    options['schema'] = schema_copy
  if asset_map is not None:
    asset_path = os.path.join(transform_output_path, 'transform_fn',
                              tf.saved_model.ASSETS_DIRECTORY)
    vocab_paths = {k: os.path.join(asset_path, v) for k, v in asset_map.items()}
    options['vocab_paths'] = vocab_paths
  options['experimental_use_sketch_based_topk_uniques'] = True
  return stats_options_updater_fn(stats_type, tfdv.StatsOptions(**options))


def _FilterInternalColumn(
    record_batch: pa.RecordBatch,
    internal_column_index: Optional[int] = None) -> pa.RecordBatch:
  """Returns shallow copy of a RecordBatch with internal column removed."""
  if (internal_column_index is None and
      _TRANSFORM_INTERNAL_FEATURE_FOR_KEY not in record_batch.schema.names):
    return record_batch
  else:
    internal_column_index = (
        internal_column_index or
        record_batch.schema.names.index(_TRANSFORM_INTERNAL_FEATURE_FOR_KEY))
    # Making shallow copy since input modification is not allowed.
    filtered_columns = list(record_batch.columns)
    filtered_columns.pop(internal_column_index)
    filtered_schema = record_batch.schema.remove(internal_column_index)
    return pa.RecordBatch.from_arrays(filtered_columns, schema=filtered_schema)


class Executor(base_beam_executor.BaseBeamExecutor):
  """Transform executor."""

  def __init__(
      self, context: Optional[base_executor.BaseExecutor.Context] = None):
    super().__init__(context)
    self._pip_dependencies = []

  def _GetPreprocessingFn(
      self, inputs: Mapping[str, Any],
      unused_outputs: Mapping[str, Any]) -> Callable[..., Any]:
    """Returns a user defined preprocessing_fn.

    If a custom config is provided in inputs, and also needed in
    preprocessing_fn, bind it to preprocessing_fn.

    Args:
      inputs: A dictionary of labelled input values.
      unused_outputs: A dictionary of labelled output values.

    Returns:
      User defined function, optionally bound with a custom config.

    Raises:
      ValueError: When neither or both of MODULE_FILE and PREPROCESSING_FN
        are present in inputs.
    """
    executor_utils.ValidateOnlyOneSpecified(
        inputs,
        (labels.MODULE_FILE, labels.MODULE_PATH, labels.PREPROCESSING_FN))

    fn = udf_utils.get_fn(
        {
            standard_component_specs.MODULE_FILE_KEY:
                value_utils.GetSoleValue(
                    inputs, labels.MODULE_FILE, strict=False),
            standard_component_specs.MODULE_PATH_KEY:
                value_utils.GetSoleValue(
                    inputs, labels.MODULE_PATH, strict=False),
            standard_component_specs.PREPROCESSING_FN_KEY:
                value_utils.GetSoleValue(
                    inputs, labels.PREPROCESSING_FN, strict=False),
        }, standard_component_specs.PREPROCESSING_FN_KEY)

    return executor_utils.MaybeBindCustomConfig(inputs, fn)

  def _GetStatsOptionsUpdaterFn(
      self, inputs: Mapping[str, Any]
  ) -> Optional[Callable[[stats_options_util.StatsType, tfdv.StatsOptions],
                         tfdv.StatsOptions]]:
    """Returns the user-defined stats_options_updater_fn.

    If a custom config is provided in inputs, and also needed in
    stats_options_updater_fn, bind it to stats_options_updater_fn.

    Args:
      inputs: A dictionary of labelled input values.

    Returns:
      User defined function, optionally bound with a custom config.
    """
    has_fn = executor_utils.ValidateOnlyOneSpecified(
        inputs, (labels.MODULE_FILE, labels.MODULE_PATH,
                 labels.STATS_OPTIONS_UPDATER_FN),
        allow_missing=True)
    if not has_fn:
      return None

    fn = udf_utils.try_get_fn(
        {
            standard_component_specs.MODULE_FILE_KEY:
                value_utils.GetSoleValue(
                    inputs, labels.MODULE_FILE, strict=False),
            standard_component_specs.MODULE_PATH_KEY:
                value_utils.GetSoleValue(
                    inputs, labels.MODULE_PATH, strict=False),
            standard_component_specs.STATS_OPTIONS_UPDATER_FN_KEY:
                value_utils.GetSoleValue(
                    inputs, labels.STATS_OPTIONS_UPDATER_FN, strict=False)
        }, standard_component_specs.STATS_OPTIONS_UPDATER_FN_KEY)
    if fn is None:
      return fn
    return executor_utils.MaybeBindCustomConfig(inputs, fn)

  def Do(self, input_dict: Dict[str, List[types.Artifact]],
         output_dict: Dict[str, List[types.Artifact]],
         exec_properties: Dict[str, Any]) -> None:
    """TensorFlow Transform executor entrypoint.

    This implements BaseExecutor.Do() and is invoked by orchestration systems.
    This is not inteded for manual usage or further customization. Please use
    the Transform() function which takes an input format with no artifact
    dependency.

    Args:
      input_dict: Input dict from input key to a list of artifacts, including:
        - examples: A list of type `standard_artifacts.Examples` which should
          contain custom splits specified in splits_config. If custom split is
          not provided, this should contain two splits 'train' and 'eval'.
        - schema: A list of type `standard_artifacts.Schema` which should
          contain a single schema artifact.
        - analyzer_cache: Cache input of 'tf.Transform', where cached
          information for analyzed examples from previous runs will be read.
      output_dict: Output dict from key to a list of artifacts, including:
        - transform_graph: Output of 'tf.Transform', which includes an exported
          Tensorflow graph suitable for both training and serving;
        - transformed_examples: Materialized transformed examples, which
          includes transform splits as specified in splits_config. If custom
          split is not provided, this should include both 'train' and 'eval'
          splits.
        - updated_analyzer_cache: Cache output of 'tf.Transform', where
          cached information for analyzed examples will be written.
      exec_properties: A dict of execution properties, including:
        - module_file: The file path to a python module file, from which the
          'preprocessing_fn' function will be loaded. Exactly one of
          'module_file', 'module_path' and 'preprocessing_fn' should be set.
        - module_path: The python module path, from which the
          'preprocessing_fn' function will be loaded. Exactly one of
          'module_file', 'module_path' and 'preprocessing_fn' should be set.
        - preprocessing_fn: The module path to a python function that
          implements 'preprocessing_fn'. Exactly one of 'module_file',
          'module_path' and 'preprocessing_fn' should be set.
        - 'stats_options_updater_fn': The module path to a python function that
          implements 'stats_options_updater_fn'. This cannot be specified
          together with 'module_file'.
        - splits_config: A transform_pb2.SplitsConfig instance, providing splits
          that should be analyzed and splits that should be transformed. Note
          analyze and transform splits can have overlap. Default behavior (when
          splits_config is not set) is analyze the 'train' split and transform
          all splits. If splits_config is set, analyze cannot be empty.
        - force_tf_compat_v1: Whether to use TF in compat.v1 mode
          irrespective of installed/enabled TF behaviors.
        - disable_statistics: Whether to disable computation of pre-transform
          and post-transform statistics.

    Returns:
      None
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    executor_utils.MatchNumberOfTransformedExamplesArtifacts(
        input_dict, output_dict)

    splits_config = executor_utils.ResolveSplitsConfig(
        exec_properties.get(standard_component_specs.SPLITS_CONFIG_KEY),
        input_dict[standard_component_specs.EXAMPLES_KEY])

    payload_format, data_view_uri = (
        tfxio_utils.resolve_payload_format_and_data_view_uri(
            input_dict[standard_component_specs.EXAMPLES_KEY]))
    examples_file_formats = [
        examples_utils.get_file_format(artifact)
        for artifact in input_dict[standard_component_specs.EXAMPLES_KEY]
    ]
    schema_file = io_utils.get_only_uri_in_dir(
        artifact_utils.get_single_uri(
            input_dict[standard_component_specs.SCHEMA_KEY]))
    transform_output = artifact_utils.get_single_uri(
        output_dict[standard_component_specs.TRANSFORM_GRAPH_KEY])

    disable_statistics = bool(
        exec_properties.get(standard_component_specs.DISABLE_STATISTICS_KEY, 0))
    stats_output_paths = executor_utils.GetStatsOutputPathEntries(
        disable_statistics, output_dict)

    temp_path = os.path.join(transform_output, _TEMP_DIR_IN_TRANSFORM_OUTPUT)
    logging.debug('Using temp path %s for tft.beam', temp_path)

    analyze_data_paths = []
    analyze_file_formats = []
    for split in splits_config.analyze:
      data_uris = artifact_utils.get_split_uris(
          input_dict[standard_component_specs.EXAMPLES_KEY], split)
      assert len(data_uris) == len(
          examples_file_formats), 'Length of file formats is different'
      for data_uri, file_format in zip(data_uris, examples_file_formats):
        analyze_data_paths.append(io_utils.all_files_pattern(data_uri))
        analyze_file_formats.append(file_format)

    transform_data_paths = []
    transform_file_formats = []
    for split in splits_config.transform:
      data_uris = artifact_utils.get_split_uris(
          input_dict[standard_component_specs.EXAMPLES_KEY], split)
      assert len(data_uris) == len(
          examples_file_formats), 'Length of file formats is different'
      for data_uri, file_format in zip(data_uris, examples_file_formats):
        transform_data_paths.append(io_utils.all_files_pattern(data_uri))
        transform_file_formats.append(file_format)

    transformed_examples = output_dict.get(
        standard_component_specs.TRANSFORMED_EXAMPLES_KEY)
    executor_utils.SetSplitNames(splits_config.transform, transformed_examples)
    materialize_output_paths = executor_utils.GetSplitPaths(
        transformed_examples)

    force_tf_compat_v1 = bool(
        exec_properties.get(standard_component_specs.FORCE_TF_COMPAT_V1_KEY, 0))

    # Make sure user packages get propagated to the remote Beam worker.
    user_module_key = exec_properties.get(
        standard_component_specs.MODULE_PATH_KEY, None)
    _, extra_pip_packages = udf_utils.decode_user_module_key(user_module_key)
    for pip_package_path in extra_pip_packages:
      local_pip_package_path = io_utils.ensure_local(pip_package_path)
      self._beam_pipeline_args.append(_BEAM_EXTRA_PACKAGE_PREFIX +
                                      local_pip_package_path)
      self._pip_dependencies.append(local_pip_package_path)

    inputs_for_fn_resolution = {
        labels.MODULE_FILE:
            exec_properties.get(standard_component_specs.MODULE_FILE_KEY, None),
        labels.MODULE_PATH:
            user_module_key,
        labels.PREPROCESSING_FN:
            exec_properties.get(standard_component_specs.PREPROCESSING_FN_KEY,
                                None),
        labels.STATS_OPTIONS_UPDATER_FN:
            exec_properties.get(
                standard_component_specs.STATS_OPTIONS_UPDATER_FN_KEY, None),
        labels.CUSTOM_CONFIG:
            exec_properties.get(standard_component_specs.CUSTOM_CONFIG_KEY,
                                None),
        # Used in nitroml/automl/autodata/transform/executor.py
        labels.SCHEMA_PATH_LABEL:
            schema_file,
    }
    # Used in nitroml/automl/autodata/transform/executor.py
    outputs_for_fn_resolution = {
        labels.TRANSFORM_METADATA_OUTPUT_PATH_LABEL: transform_output,
    }
    # TODO(b/178065215): Refactor to pass exec_properties directly.
    #                    We need to change usages in nitroml, too.
    preprocessing_fn = self._GetPreprocessingFn(inputs_for_fn_resolution,
                                                outputs_for_fn_resolution)
    stats_options_updater_fn = self._GetStatsOptionsUpdaterFn(
        inputs_for_fn_resolution)

    label_inputs = {
        labels.DISABLE_STATISTICS_LABEL:
            disable_statistics,
        labels.SCHEMA_PATH_LABEL:
            schema_file,
        labels.EXAMPLES_DATA_FORMAT_LABEL:
            payload_format,
        labels.DATA_VIEW_LABEL:
            data_view_uri,
        labels.ANALYZE_DATA_PATHS_LABEL:
            analyze_data_paths,
        labels.ANALYZE_PATHS_FILE_FORMATS_LABEL:
            analyze_file_formats,
        labels.TRANSFORM_DATA_PATHS_LABEL:
            transform_data_paths,
        labels.TRANSFORM_PATHS_FILE_FORMATS_LABEL:
            transform_file_formats,
        labels.PREPROCESSING_FN:
            preprocessing_fn,
        labels.STATS_OPTIONS_UPDATER_FN:
            stats_options_updater_fn,
        labels.MAKE_BEAM_PIPELINE_FN:
            self._make_beam_pipeline,
        labels.FORCE_TF_COMPAT_V1_LABEL:
            force_tf_compat_v1,
        **executor_utils.GetCachePathEntry(
            standard_component_specs.ANALYZER_CACHE_KEY, input_dict)
    }

    label_outputs = {
        labels.TRANSFORM_METADATA_OUTPUT_PATH_LABEL:
            transform_output,
        labels.TRANSFORM_MATERIALIZE_OUTPUT_PATHS_LABEL:
            materialize_output_paths,
        labels.TEMP_OUTPUT_LABEL:
            str(temp_path),
        **stats_output_paths,
        **executor_utils.GetCachePathEntry(
            standard_component_specs.UPDATED_ANALYZER_CACHE_KEY, output_dict),
    }

    status_file = 'status_file'  # Unused

    # TempPipInstallContext is needed here so that subprocesses (which
    # may be created by the Beam multi-process DirectRunner) can find the
    # needed dependencies.
    # TODO(b/187122662): Move this to the ExecutorOperator or Launcher and
    # remove the `_pip_dependencies` attribute.
    with udf_utils.TempPipInstallContext(self._pip_dependencies):
      TransformProcessor().Transform(label_inputs, label_outputs, status_file)
    logging.debug('Cleaning up temp path %s on executor success', temp_path)
    io_utils.delete_dir(temp_path)


class TransformProcessor:
  """Transforms using Beam."""

  @staticmethod
  @beam.ptransform_fn
  @beam.typehints.with_input_types(beam.Pipeline)
  def _IncrementPipelineMetrics(
      pipeline: beam.Pipeline, total_columns_count: int,
      analyze_columns_count: int, transform_columns_count: int,
      analyze_paths_count: int, analyzer_cache_enabled: bool,
      disable_statistics: bool, materialize: bool,
      estimated_stage_count_with_cache: int):
    """A beam PTransform to increment counters of column usage."""

    def _MakeAndIncrementCounters(unused_element):
      """Increment column usage counters."""
      del unused_element
      beam.metrics.Metrics.counter(
          tft_beam.common.METRICS_NAMESPACE,
          'total_columns_count').inc(total_columns_count)
      beam.metrics.Metrics.counter(
          tft_beam.common.METRICS_NAMESPACE,
          'analyze_columns_count').inc(analyze_columns_count)
      beam.metrics.Metrics.counter(
          tft_beam.common.METRICS_NAMESPACE,
          'transform_columns_count').inc(transform_columns_count)
      beam.metrics.Metrics.counter(
          tft_beam.common.METRICS_NAMESPACE,
          'analyze_paths_count').inc(analyze_paths_count)
      beam.metrics.Metrics.counter(
          tft_beam.common.METRICS_NAMESPACE,
          'analyzer_cache_enabled').inc(int(analyzer_cache_enabled))
      beam.metrics.Metrics.counter(
          tft_beam.common.METRICS_NAMESPACE,
          'disable_statistics').inc(int(disable_statistics))
      beam.metrics.Metrics.counter(
          tft_beam.common.METRICS_NAMESPACE,
          'materialize').inc(int(materialize))
      beam.metrics.Metrics.distribution(
          tft_beam.common.METRICS_NAMESPACE,
          'estimated_stage_count_with_cache').update(
              estimated_stage_count_with_cache)
      return beam.pvalue.PDone(pipeline)

    return (
        pipeline
        | 'CreateSole' >> beam.Create([None])
        | 'Count' >> beam.Map(_MakeAndIncrementCounters))

  # TODO(b/139538871): Implement telemetry, on top of pa.Table once available.
  @staticmethod
  @beam.ptransform_fn
  @beam.typehints.with_input_types(Tuple[pa.RecordBatch, Dict[str, pa.Array]])
  def _EncodeAndWrite(pcoll: beam.PCollection, schema: schema_pb2.Schema,
                      file_format: str, output_path: str) -> beam.pvalue.PDone:
    """Encodes and writes transformed RecordBatches in the given file format.

    Args:
      pcoll: PCollection of transformed RecordBatches and unary pass-through
        features.
      schema: TFMD schema for the transformed data.
      file_format: The output file format.
      output_path: Path that will serve as a prefix for the produced files.

    Returns:
      beam.pvalue.PDone.
    """
    if file_format == labels.FORMAT_TFRECORD:
      return (pcoll
              | 'EncodeAndSerialize' >> beam.ParDo(
                  TransformProcessor._RecordBatchToExamplesFn(schema))
              | 'ExtractExamples' >> beam.Values()
              | 'WriteToTFRecord' >> beam.io.WriteToTFRecord(
                  output_path, file_name_suffix='.gz'))
    elif file_format == _FILE_FORMAT_PARQUET:
      arrow_schema = (
          impl_helper.make_tensor_to_arrow_converter(schema).arrow_schema())
      return (pcoll | 'ExtractRecordBatches' >> beam.Keys()
              | 'ToRecords' >>
              beam.FlatMap(lambda x: x.to_pandas().to_dict('records'))
              | 'WriteToParquet' >> beam.io.WriteToParquet(
                  output_path,
                  arrow_schema,
                  file_name_suffix='.parquet',
                  codec='snappy'))
    else:
      raise NotImplementedError(
          f'Unsupported output file format: {file_format}. Supported formats '
          f'are {labels.FORMAT_TFRECORD} and {_FILE_FORMAT_PARQUET}.')

  def _GetSchema(self, schema_path: str) -> schema_pb2.Schema:
    """Gets a tf.metadata schema.

    Args:
      schema_path: Path to schema file.

    Returns:
      A tf.metadata schema.
    """
    schema_reader = io_utils.SchemaReader()
    return schema_reader.read(schema_path)

  def _ReadMetadata(self, data_format: int,
                    schema_path: str) -> dataset_metadata.DatasetMetadata:
    """Returns a dataset_metadata.DatasetMetadata for the input data.

    Args:
      data_format: The data format of the dataset. One of the enums from
        example_gen_pb2.PayloadFormat.
      schema_path: path to schema file.

    Returns:
      A dataset_metadata.DatasetMetadata representing the provided set of
          columns.
    """
    if self._IsDataFormatProto(data_format):
      return dataset_metadata.DatasetMetadata(_RAW_EXAMPLE_SCHEMA)

    schema_proto = self._GetSchema(schema_path)
    input_dataset_metadata = dataset_metadata.DatasetMetadata(schema_proto)
    if self._DecodesSequenceExamplesAsRawRecords(data_format,
                                                 input_dataset_metadata.schema):
      return dataset_metadata.DatasetMetadata(_RAW_EXAMPLE_SCHEMA)
    return input_dataset_metadata

  @staticmethod
  @beam.ptransform_fn
  @beam.typehints.with_input_types(pa.RecordBatch)
  # Typehints are not supported for ptransform_fns/PTransfroms which return
  # multiple PCollections.
  def _GenerateAndMaybeValidateStats(
      pcoll: beam.pvalue.PCollection, stats_output_loc: Union[str, Dict[str,
                                                                        str]],
      stats_options: tfdv.StatsOptions, enable_validation: bool
  ) -> Tuple[beam.pvalue.PDone, Optional[beam.pvalue.PDone],
             Optional[beam.pvalue.PDone]]:
    """Generates statistics.

    Args:
      pcoll: PCollection of examples.
      stats_output_loc: path to the statistics folder to write all results to
        or a dictionary keyed with individual paths for 'schema', 'stats', and
        'anomalies'.
      stats_options: An instance of `tfdv.StatsOptions()` used when computing
        statistics.
      enable_validation: Whether to enable stats validation.

    Returns:
      A tuple containing the beam.pvalue.PDones for generating the stats,
      writing the schema, and writing the validation, in that order. If the
      schema is not present or validation is not enabled, the corresponding
      values are replaced with Nones.
    """
    if isinstance(stats_output_loc, dict):
      stats_output_path = stats_output_loc[_STATS_KEY]
      sharded_stats_output_prefix = stats_output_loc[_SHARDED_STATS_KEY]
      schema_output_path = stats_output_loc[_SCHEMA_KEY]
      anomalies_output_path = stats_output_loc.get(_ANOMALIES_KEY)
    else:
      stats_output_path = stats_output_loc
      stats_output_dir = os.path.dirname(stats_output_loc)
      schema_output_path = os.path.join(stats_output_dir, _SCHEMA_FILE)
      sharded_stats_output_prefix = os.path.join(stats_output_dir,
                                                 SHARDED_STATS_PREFIX)
      anomalies_output_path = os.path.join(stats_output_dir, _ANOMALIES_FILE)

    generated_stats = (
        pcoll
        | 'FilterInternalColumn' >> beam.Map(_FilterInternalColumn)
        | 'GenerateStatistics' >> tfdv.GenerateStatistics(stats_options))

    if (stats_options.experimental_result_partitions > 1 and
        tfdv.default_sharded_output_supported()):
      stats_result = (
          generated_stats
          | 'WriteStats' >> tfdv.WriteStatisticsToRecordsAndBinaryFile(
              binary_proto_path=stats_output_path,
              records_path_prefix=sharded_stats_output_prefix))
    else:
      stats_result = (
          generated_stats
          | 'WriteStats' >>
          tfdv.WriteStatisticsToBinaryFile(output_path=stats_output_path))

    if stats_options.schema is None:
      return (stats_result, None, None)

    # TODO(b/186867968): See if we should switch to common libraries.
    schema_result = (
        pcoll.pipeline
        | 'CreateSchema' >> beam.Create([
            text_format.MessageToString(stats_options.schema)])
        | 'WriteSchema' >> beam.io.WriteToText(
            schema_output_path,
            append_trailing_newlines=False,
            shard_name_template=''  # To force unsharded output.
            ))

    if not enable_validation:
      return (stats_result, schema_result, None)

    # TODO(b/186867968): See if we should switch to common libraries.
    validation_result = (
        generated_stats
        | 'ValidateStatistics' >> beam.Map(
            lambda stats: tfdv.validate_statistics(stats, stats_options.schema))
        | 'WriteValidation' >> beam.io.WriteToText(
            anomalies_output_path,
            append_trailing_newlines=False,
            shard_name_template='',  # To force unsharded output.
            coder=beam.coders.ProtoCoder(anomalies_pb2.Anomalies)))

    return (stats_result, schema_result, validation_result)

  # TODO(b/130807807): This is still used by pre-transform stats to decode raw
  # sequence examples as tf.example. Once only native sequence example path is
  # supported this can be removed.
  @beam.typehints.with_input_types(List[bytes])
  @beam.typehints.with_output_types(pa.RecordBatch)
  class _ToArrowRecordBatchesFn(beam.DoFn):
    """Converts a batch of serialized examples to an Arrow RecordBatch."""

    def __init__(self, schema: Optional[schema_pb2.Schema]):
      self._serialized_schema = schema.SerializeToString() if schema else None

    def setup(self):
      args = ([] if self._serialized_schema is None
              else [self._serialized_schema])
      self._decoder = (
          tfx_bsl.coders.example_coder.ExamplesToRecordBatchDecoder(*args))

    def process(self, element: List[bytes]) -> Iterable[pa.RecordBatch]:
      yield self._decoder.DecodeBatch(element)

  @beam.typehints.with_input_types(Tuple[pa.RecordBatch, Dict[str, pa.Array]])
  @beam.typehints.with_output_types(Tuple[Any, bytes])
  class _RecordBatchToExamplesFn(beam.DoFn):
    """Maps `pa.RecordBatch` to a generator of serialized `tf.Example`s."""

    def __init__(self, schema: schema_pb2.Schema):
      self._coder = tfx_bsl.coders.example_coder.RecordBatchToExamplesEncoder(
          schema)

    def process(
        self, data_batch: Tuple[pa.RecordBatch, Dict[str, pa.Array]]
    ) -> Iterable[Tuple[Any, bytes]]:
      record_batch, unary_passthrough_features = data_batch
      if _TRANSFORM_INTERNAL_FEATURE_FOR_KEY in record_batch.schema.names:
        keys_index = record_batch.schema.names.index(
            _TRANSFORM_INTERNAL_FEATURE_FOR_KEY)
        keys = record_batch.column(keys_index).to_pylist()
        # Filter the record batch to make sure that the internal column doesn't
        # get encoded.
        record_batch = _FilterInternalColumn(record_batch, keys_index)
        examples = self._coder.encode(record_batch)
        for key, example in zip(keys, examples):
          yield (None if key is None else key[0], example)
      else:
        # Internal feature key is not present in the record batch but may be
        # present in the unary pass-through features dict.
        key = unary_passthrough_features.get(
            _TRANSFORM_INTERNAL_FEATURE_FOR_KEY, None)
        if key is not None:
          # The key is `pa.large_list()` and is, therefore, doubly nested.
          key_list = key.to_pylist()[0]
          key = None if key_list is None else key_list[0]
        examples = self._coder.encode(record_batch)
        for example in examples:
          yield (key, example)

  @beam.typehints.with_input_types(beam.Pipeline)
  class _OptimizeRun(beam.PTransform):
    """Utilizes TFT cache if applicable and removes unused datasets."""

    # pyformat: disable
    def __init__(self,
                 input_cache_dir: str,
                 output_cache_dir: str,
                 analyze_data_list: List[_Dataset],
                 typespecs: Mapping[str, tf.TypeSpec],
                 preprocessing_fn: Any,
                 cache_source: beam.PTransform,
                 force_tf_compat_v1: bool):
      # pyformat: enable
      self._input_cache_dir = input_cache_dir
      self._output_cache_dir = output_cache_dir
      self._analyze_data_list = analyze_data_list
      self._feature_spec_or_typespec = typespecs
      self._preprocessing_fn = preprocessing_fn
      self._cache_source = cache_source
      self._force_tf_compat_v1 = force_tf_compat_v1

    # TODO(zoy): Remove this method once beam no longer pickles PTransforms,
    # once https://issues.apache.org/jira/browse/BEAM-3812 is resolved.
    def to_runner_api_pickled(self, context):
      # Overriding to_runner_api_pickled and calling to_runner_api_parameter
      # instead to make sure that beam doesn't try to pickle the
      # preprocessing_fn with the PTransform instance since it may not be
      # picklable.
      return self.to_runner_api_parameter(context)

    def expand(
        self, pipeline: beam.Pipeline
    ) -> Tuple[Dict[str, Optional[_Dataset]], Optional[Dict[str, Dict[
        str, beam.pvalue.PCollection]]], int]:
      # TODO(b/170304777): Remove this Create once the issue is fixed in beam.
      # Forcing beam to treat this PTransform as non-primitive.
      _ = pipeline | 'WorkaroundForBug170304777' >> beam.Create([None])

      dataset_keys_list = [
          dataset.dataset_key for dataset in self._analyze_data_list
      ]
      cache_entry_keys = (
          tft_beam.analysis_graph_builder.get_analysis_cache_entry_keys(
              self._preprocessing_fn, self._feature_spec_or_typespec,
              dataset_keys_list, self._force_tf_compat_v1))
      # We estimate the number of stages in the pipeline to be roughly:
      # analyzers * analysis_paths * 10.
      # TODO(b/37788560): Remove this restriction when a greater number of
      # stages can be handled efficiently.
      estimated_stage_count = (
          len(cache_entry_keys) * len(dataset_keys_list) * 10)
      if estimated_stage_count > _MAX_ESTIMATED_STAGES_COUNT:
        logging.warning(
            'Disabling cache because otherwise the number of stages might be '
            'too high (%d analyzers, %d analysis paths)', len(cache_entry_keys),
            len(dataset_keys_list))
        # Returning None as the input cache here disables both input and output
        # cache.
        return ({d.dataset_key: d for d in self._analyze_data_list}, None,
                estimated_stage_count)

      if self._input_cache_dir is not None:
        logging.info('Reading the following analysis cache entry keys: %s',
                     cache_entry_keys)
        input_cache = (
            pipeline
            | 'ReadCache' >> analyzer_cache.ReadAnalysisCacheFromFS(
                self._input_cache_dir,
                dataset_keys_list,
                source=self._cache_source,
                cache_entry_keys=cache_entry_keys))
      elif self._output_cache_dir is not None:
        input_cache = {}
      else:
        # Using None here to indicate that this pipeline will not read or write
        # cache.
        input_cache = None

      if input_cache is None:
        # Cache is disabled so we won't be filtering out any datasets, and will
        # always perform a flatten over all of them.
        filtered_analysis_dataset_keys = dataset_keys_list
      else:
        filtered_analysis_dataset_keys = (
            tft_beam.analysis_graph_builder.get_analysis_dataset_keys(
                self._preprocessing_fn, self._feature_spec_or_typespec,
                dataset_keys_list, input_cache, self._force_tf_compat_v1))

      new_analyze_data_dict = {}
      for dataset in self._analyze_data_list:
        if dataset.dataset_key in filtered_analysis_dataset_keys:
          new_analyze_data_dict[dataset.dataset_key] = dataset
        else:
          new_analyze_data_dict[dataset.dataset_key] = None

      return (new_analyze_data_dict, input_cache, estimated_stage_count)

  def Transform(self, inputs: Mapping[str, Any], outputs: Mapping[str, Any],
                status_file: Optional[str] = None) -> None:
    """Executes on request.

    This is the implementation part of transform executor. This is intended for
    using or extending the executor without artifact dependency.

    Args:
      inputs: A dictionary of labelled input values, including:
        - labels.DISABLE_STATISTICS_LABEL: Whether disable statistics
          compuatation.
        - labels.SCHEMA_PATH_LABEL: Path to schema file.
        - labels.EXAMPLES_DATA_FORMAT_LABEL: Example data format, one of the
            enums from example_gen_pb2.PayloadFormat.
        - labels.ANALYZE_DATA_PATHS_LABEL: Paths or path patterns to analyze
          data.
        - labels.ANALYZE_PATHS_FILE_FORMATS_LABEL: File formats of paths to
          analyze data.
        - labels.TRANSFORM_DATA_PATHS_LABEL: Paths or path patterns to transform
          data.
        - labels.TRANSFORM_PATHS_FILE_FORMATS_LABEL: File formats of paths to
          transform data.
        - labels.PREPROCESSING_FN: Python function that implements
          preprocessing_fn.
        - labels.STATS_OPTIONS_UPDATER_FN: Python function that implements
          stats_options_updater_fn, optional.
        - labels.MAKE_BEAM_PIPELINE_FN: Python function that makes a Beam
          pipeline object.
        - labels.DATA_VIEW_LABEL: DataView to be used to read the Example,
          optional
        - labels.FORCE_TF_COMPAT_V1_LABEL: Whether to use TF in compat.v1 mode
          irrespective of installed/enabled TF behaviors.
      outputs: A dictionary of labelled output values, including:
        - labels.PER_SET_STATS_OUTPUT_PATHS_LABEL: Paths to statistics output,
          optional.
        - labels.TRANSFORM_METADATA_OUTPUT_PATH_LABEL: A path to
          TFTransformOutput output.
        - labels.TRANSFORM_MATERIALIZE_OUTPUT_PATHS_LABEL: Paths to transform
          materialization.
        - labels.TEMP_OUTPUT_LABEL: A path to temporary directory.
        - labels.PRE_TRANSFORM_OUTPUT_SCHEMA_PATH_LABEL: A path to the output
          pre-transform schema, optional.
        - labels.PRE_TRANSFORM_OUTPUT_STATS_PATH_LABEL: A path to the output
          pre-transform statistics, optional.
        - labels.POST_TRANSFORM_OUTPUT_SCHEMA_PATH_LABEL: A path to the output
          post-transform schema, optional.
        - labels.POST_TRANSFORM_OUTPUT_STATS_PATH_LABEL: A path to the output
          post-transform statistics, optional.
        - labels.POST_TRANSFORM_OUTPUT_ANOMALIES_PATH_LABEL: A path to the
          output post-transform anomalies, optional.
      status_file: Where the status should be written (not yet implemented)
    """
    del status_file  # unused

    logging.debug('Inputs to executor.Transform function: %s', inputs)
    logging.debug('Outputs to executor.Transform function: %s', outputs)

    disable_statistics = value_utils.GetSoleValue(
        inputs, labels.DISABLE_STATISTICS_LABEL)
    transform_output_path = value_utils.GetSoleValue(
        outputs, labels.TRANSFORM_METADATA_OUTPUT_PATH_LABEL)
    raw_examples_data_format = value_utils.GetSoleValue(
        inputs, labels.EXAMPLES_DATA_FORMAT_LABEL)
    schema = value_utils.GetSoleValue(inputs, labels.SCHEMA_PATH_LABEL)
    input_dataset_metadata = self._ReadMetadata(raw_examples_data_format,
                                                schema)
    materialize_output_paths = value_utils.GetValues(
        outputs, labels.TRANSFORM_MATERIALIZE_OUTPUT_PATHS_LABEL)
    preprocessing_fn = inputs[labels.PREPROCESSING_FN]
    stats_options_updater_fn = inputs.get(labels.STATS_OPTIONS_UPDATER_FN)
    make_beam_pipeline_fn = inputs[labels.MAKE_BEAM_PIPELINE_FN]
    analyze_data_paths = value_utils.GetValues(inputs,
                                               labels.ANALYZE_DATA_PATHS_LABEL)
    analyze_paths_file_formats = value_utils.GetValues(
        inputs, labels.ANALYZE_PATHS_FILE_FORMATS_LABEL)
    transform_data_paths = value_utils.GetValues(
        inputs, labels.TRANSFORM_DATA_PATHS_LABEL)
    transform_paths_file_formats = value_utils.GetValues(
        inputs, labels.TRANSFORM_PATHS_FILE_FORMATS_LABEL)
    input_cache_dir = value_utils.GetSoleValue(
        inputs, labels.CACHE_INPUT_PATH_LABEL, strict=False)
    output_cache_dir = value_utils.GetSoleValue(
        outputs, labels.CACHE_OUTPUT_PATH_LABEL, strict=False)
    per_set_stats_output_paths = value_utils.GetValues(
        outputs, labels.PER_SET_STATS_OUTPUT_PATHS_LABEL)
    temp_path = value_utils.GetSoleValue(outputs, labels.TEMP_OUTPUT_LABEL)
    data_view_uri = value_utils.GetSoleValue(
        inputs, labels.DATA_VIEW_LABEL, strict=False)
    force_tf_compat_v1 = value_utils.GetSoleValue(
        inputs, labels.FORCE_TF_COMPAT_V1_LABEL)

    stats_labels_list = [
        labels.PRE_TRANSFORM_OUTPUT_STATS_PATH_LABEL,
        labels.PRE_TRANSFORM_OUTPUT_SCHEMA_PATH_LABEL,
        labels.POST_TRANSFORM_OUTPUT_ANOMALIES_PATH_LABEL,
        labels.POST_TRANSFORM_OUTPUT_STATS_PATH_LABEL,
        labels.POST_TRANSFORM_OUTPUT_SCHEMA_PATH_LABEL
    ]
    stats_output_paths = {}
    for label in stats_labels_list:
      value = value_utils.GetSoleValue(outputs, label, strict=False)
      if value:
        stats_output_paths[label] = value
    if stats_output_paths and len(stats_output_paths) != len(stats_labels_list):
      raise ValueError('Either all stats_output_paths should be'
                       ' specified or none.')

    logging.debug('Force tf.compat.v1: %s', force_tf_compat_v1)
    logging.debug('Analyze data patterns: %s',
                  list(enumerate(analyze_data_paths)))
    logging.debug('Transform data patterns: %s',
                  list(enumerate(transform_data_paths)))
    logging.debug('Transform materialization output paths: %s',
                  list(enumerate(materialize_output_paths)))
    logging.debug('Transform output path: %s', transform_output_path)

    if len(analyze_data_paths) != len(analyze_paths_file_formats):
      raise ValueError(
          'size of analyze_data_paths and '
          'analyze_paths_file_formats do not match: {} v.s {}'.format(
              len(analyze_data_paths), len(analyze_paths_file_formats)))
    if len(transform_data_paths) != len(transform_paths_file_formats):
      raise ValueError(
          'size of transform_data_paths and '
          'transform_paths_file_formats do not match: {} v.s {}'.format(
              len(transform_data_paths), len(transform_paths_file_formats)))

    can_process_analysis_jointly = not bool(output_cache_dir)
    analyze_data_list = self._MakeDatasetList(analyze_data_paths,
                                              analyze_paths_file_formats,
                                              raw_examples_data_format,
                                              data_view_uri,
                                              can_process_analysis_jointly)
    if not analyze_data_list:
      raise ValueError('Analyze data list must not be empty.')

    can_process_transform_jointly = not bool(per_set_stats_output_paths or
                                             materialize_output_paths)
    transform_data_list = self._MakeDatasetList(transform_data_paths,
                                                transform_paths_file_formats,
                                                raw_examples_data_format,
                                                data_view_uri,
                                                can_process_transform_jointly,
                                                per_set_stats_output_paths,
                                                materialize_output_paths)

    all_datasets = analyze_data_list + transform_data_list
    for d in all_datasets:
      d.tfxio = self._CreateTFXIO(d, input_dataset_metadata.schema)
    self._AssertSameTFXIOSchema(all_datasets)
    typespecs = all_datasets[0].tfxio.TensorAdapter().OriginalTypeSpecs()

    # Inspecting the preprocessing_fn even if we know we need a full pass in
    # order to fail faster if it fails.
    analyze_input_columns = tft.get_analyze_input_columns(
        preprocessing_fn, typespecs, force_tf_compat_v1=force_tf_compat_v1)

    if (disable_statistics and not materialize_output_paths and
        stats_options_updater_fn is None):
      if analyze_input_columns:
        logging.warning(
            'Not using the in-place Transform because the following features '
            'require analyzing: %s', tuple(c for c in analyze_input_columns))
      else:
        logging.warning(
            'Using the in-place Transform since disable_statistics=True, '
            'it does not materialize transformed data, and the configured '
            'preprocessing_fn appears to not require analyzing the data.')
        self._RunInPlaceImpl(preprocessing_fn, force_tf_compat_v1,
                             input_dataset_metadata, typespecs,
                             transform_output_path)
        # TODO(b/122478841): Writes status to status file.
        return

    stats_options_updater_fn = (stats_options_updater_fn
                                if stats_options_updater_fn else lambda _, x: x)

    materialization_format = (
        transform_paths_file_formats[-1] if materialize_output_paths else None)
    self._RunBeamImpl(analyze_data_list, transform_data_list, preprocessing_fn,
                      stats_options_updater_fn, force_tf_compat_v1,
                      input_dataset_metadata, transform_output_path,
                      raw_examples_data_format, temp_path, input_cache_dir,
                      output_cache_dir, disable_statistics,
                      per_set_stats_output_paths, materialization_format,
                      len(analyze_data_paths), stats_output_paths,
                      make_beam_pipeline_fn)
  # TODO(b/122478841): Writes status to status file.

  # pylint: disable=expression-not-assigned, no-value-for-parameter
  def _RunBeamImpl(
      self, analyze_data_list: List[_Dataset],
      transform_data_list: List[_Dataset], preprocessing_fn: Any,
      stats_options_updater_fn: Callable[
          [stats_options_util.StatsType, tfdv.StatsOptions],
          tfdv.StatsOptions], force_tf_compat_v1: bool,
      input_dataset_metadata: dataset_metadata.DatasetMetadata,
      transform_output_path: str, raw_examples_data_format: int, temp_path: str,
      input_cache_dir: Optional[str], output_cache_dir: Optional[str],
      disable_statistics: bool, per_set_stats_output_paths: Sequence[str],
      materialization_format: Optional[str], analyze_paths_count: int,
      stats_output_paths: Dict[str, str],
      make_beam_pipeline_fn: Callable[[], beam.Pipeline]) -> _Status:
    """Perform data preprocessing with TFT.

    Args:
      analyze_data_list: List of datasets for analysis.
      transform_data_list: List of datasets for transform.
      preprocessing_fn: The tf.Transform preprocessing_fn.
      stats_options_updater_fn: The user-specified function for updating stats
        options.
      force_tf_compat_v1: If True, call Transform's API to use Tensorflow in
        tf.compat.v1 mode.
      input_dataset_metadata: A DatasetMetadata object for the input data.
      transform_output_path: An absolute path to write the output to.
      raw_examples_data_format: The data format of the raw examples. One of the
        enums from example_gen_pb2.PayloadFormat.
      temp_path: A path to a temporary dir.
      input_cache_dir: A dir containing the input analysis cache. May be None.
      output_cache_dir: A dir to write the analysis cache to. May be None.
      disable_statistics: A bool indicating whether or to disable statistics.
      per_set_stats_output_paths: Paths to per-set statistics output. If empty,
        per-set statistics is not produced.
      materialization_format: A string describing the format of the materialized
        data or None if materialization is not enabled.
      analyze_paths_count: An integer, the number of paths that should be used
        for analysis.
      stats_output_paths: A dictionary specifying the output paths to use when
        computing statistics. If the dictionary is empty, the stats will be
        placed within the transform_output_path to preserve backwards
        compatibility.
      make_beam_pipeline_fn: A callable that can create a beam pipeline.

    Returns:
      Status of the execution.
    """
    self._AssertSameTFXIOSchema(analyze_data_list)
    unprojected_typespecs = (
        analyze_data_list[0].tfxio.TensorAdapter().OriginalTypeSpecs())

    analyze_input_columns = tft.get_analyze_input_columns(
        preprocessing_fn,
        unprojected_typespecs,
        force_tf_compat_v1=force_tf_compat_v1)
    analyze_columns_count = len(analyze_input_columns)

    transform_input_columns = tft.get_transform_input_columns(
        preprocessing_fn,
        unprojected_typespecs,
        force_tf_compat_v1=force_tf_compat_v1)
    # Use the same dataset (same columns) for AnalyzeDataset and computing
    # pre-transform stats so that the data will only be read once for these
    # two operations.
    if not disable_statistics:
      analyze_input_columns = list(
          set(list(analyze_input_columns) + list(transform_input_columns)))

    for d in analyze_data_list:
      d.tfxio = d.tfxio.Project(analyze_input_columns)

    self._AssertSameTFXIOSchema(analyze_data_list)
    analyze_data_tensor_adapter_config = (
        analyze_data_list[0].tfxio.TensorAdapterConfig())

    for d in transform_data_list:
      d.tfxio = d.tfxio.Project(transform_input_columns)

    desired_batch_size = self._GetDesiredBatchSize(
        raw_examples_data_format, input_dataset_metadata.schema)

    with make_beam_pipeline_fn() as pipeline:
      with tft_beam.Context(
          temp_dir=temp_path,
          desired_batch_size=desired_batch_size,
          passthrough_keys=self._GetTFXIOPassthroughKeys(),
          use_deep_copy_optimization=True,
          force_tf_compat_v1=force_tf_compat_v1):
        (new_analyze_data_dict, input_cache,
         estimated_stage_count_with_cache) = (
             pipeline
             | 'OptimizeRun' >> self._OptimizeRun(
                 input_cache_dir, output_cache_dir, analyze_data_list,
                 unprojected_typespecs, preprocessing_fn,
                 self._GetCacheSource(), force_tf_compat_v1))

        _ = (
            pipeline
            | 'IncrementPipelineMetrics' >> self._IncrementPipelineMetrics(
                total_columns_count=len(unprojected_typespecs),
                analyze_columns_count=analyze_columns_count,
                transform_columns_count=len(transform_input_columns),
                analyze_paths_count=analyze_paths_count,
                analyzer_cache_enabled=input_cache is not None,
                disable_statistics=disable_statistics,
                materialize=materialization_format is not None,
                estimated_stage_count_with_cache=(
                    estimated_stage_count_with_cache)))

        if input_cache:
          logging.debug('Analyzing data with cache.')

        full_analyze_dataset_keys_list = [
            dataset.dataset_key for dataset in analyze_data_list
        ]

        # Removing unneeded datasets if they won't be needed for statistics or
        # materialization.
        if materialization_format is None and disable_statistics:
          if None in new_analyze_data_dict.values():
            logging.debug(
                'Not reading the following datasets due to cache: %s', [
                    dataset.file_pattern
                    for dataset in analyze_data_list
                    if new_analyze_data_dict[dataset.dataset_key] is None
                ])
          analyze_data_list = [
              d for d in new_analyze_data_dict.values() if d is not None
          ]

        for dataset in analyze_data_list:
          infix = 'AnalysisIndex{}'.format(dataset.index)
          dataset.standardized = (
              pipeline
              | 'TFXIOReadAndDecode[{}]'.format(infix) >>
              dataset.tfxio.BeamSource(desired_batch_size))

        input_analysis_data = {}
        for key, dataset in new_analyze_data_dict.items():
          input_analysis_data[key] = (None if dataset is None else
                                      dataset.standardized)

        transform_fn, cache_output = (
            (input_analysis_data, input_cache,
             analyze_data_tensor_adapter_config)
            | 'Analyze' >> tft_beam.AnalyzeDatasetWithCache(
                preprocessing_fn, pipeline=pipeline))

        # Write the raw/input metadata.
        (input_dataset_metadata
         | 'WriteMetadata' >> tft_beam.WriteMetadata(
             os.path.join(transform_output_path,
                          tft.TFTransformOutput.RAW_METADATA_DIR), pipeline))

        # WriteTransformFn writes transform_fn and metadata to subdirectories
        # tensorflow_transform.SAVED_MODEL_DIR and
        # tensorflow_transform.TRANSFORMED_METADATA_DIR respectively.
        completed_transform = (
            transform_fn
            | 'WriteTransformFn' >>
            tft_beam.WriteTransformFn(transform_output_path))

        if output_cache_dir is not None and cache_output is not None:
          fileio.makedirs(output_cache_dir)
          logging.debug('Using existing cache in: %s', input_cache_dir)
          if input_cache_dir is not None:
            # Only copy cache that is relevant to this iteration. This is
            # assuming that this pipeline operates on rolling ranges, so those
            # cache entries may also be relevant for future iterations.
            for span_cache_dir in input_analysis_data:
              full_span_cache_dir = os.path.join(input_cache_dir,
                                                 span_cache_dir.key)
              if fileio.isdir(full_span_cache_dir):
                self._CopyCache(
                    full_span_cache_dir,
                    os.path.join(output_cache_dir, span_cache_dir.key))

          # TODO(b/157479287, b/171165988): Remove this condition when beam
          # 2.26 is used.
          if cache_output:
            (cache_output
             | 'WriteCache' >> analyzer_cache.WriteAnalysisCacheToFS(
                 pipeline=pipeline,
                 cache_base_dir=output_cache_dir,
                 sink=self._GetCacheSink(),
                 dataset_keys=full_analyze_dataset_keys_list))

        if not disable_statistics or materialization_format is not None:
          # Do not compute pre-transform stats if the input format is raw
          # proto, as StatsGen would treat any input as tf.Example. Note that
          # tf.SequenceExamples are wire-format compatible with tf.Examples.
          if (not disable_statistics and
              not self._IsDataFormatProto(raw_examples_data_format)):
            # Aggregated feature stats before transformation.
            if (self._DecodesSequenceExamplesAsRawRecords(
                raw_examples_data_format, input_dataset_metadata.schema)):
              schema_proto = None
            else:
              schema_proto = input_dataset_metadata.schema

            if (self._DecodesSequenceExamplesAsRawRecords(
                raw_examples_data_format, input_dataset_metadata.schema)):

              def _ExtractRawExampleBatches(record_batch):
                return record_batch.column(
                    record_batch.schema.get_field_index(
                        RAW_EXAMPLE_KEY)).flatten().to_pylist()

              # Make use of the fact that tf.SequenceExample is wire-format
              # compatible with tf.Example
              stats_input = []
              for dataset in analyze_data_list:
                infix = 'AnalysisIndex{}'.format(dataset.index)
                stats_input.append(
                    dataset.standardized
                    | 'ExtractRawExampleBatches[{}]'.format(
                        infix) >> beam.Map(_ExtractRawExampleBatches)
                    | 'DecodeSequenceExamplesAsExamplesIntoRecordBatches[{}]'
                    .format(infix) >> beam.ParDo(
                        self._ToArrowRecordBatchesFn(schema_proto)))
            else:
              stats_input = [
                  dataset.standardized for dataset in analyze_data_list
              ]

            pre_transform_stats_options = _InvokeStatsOptionsUpdaterFn(
                stats_options_updater_fn,
                stats_options_util.StatsType.PRE_TRANSFORM, schema_proto)
            if self._TFDVWriteShardedOutput():
              pre_transform_stats_options.experimental_result_partitions = (
                  _SHARDED_OUTPUT_PARTITIONS)
            else:
              if (pre_transform_stats_options.experimental_result_partitions !=
                  1):
                raise ValueError('Sharded output disabled requires '
                                 'experimental_result_partitions=1.')

            if stats_output_paths:
              pre_transform_feature_stats_loc = {
                  _STATS_KEY:
                      os.path.join(
                          stats_output_paths[
                              labels.PRE_TRANSFORM_OUTPUT_STATS_PATH_LABEL],
                          STATS_FILE),
                  _SHARDED_STATS_KEY:
                      os.path.join(
                          stats_output_paths[
                              labels.PRE_TRANSFORM_OUTPUT_STATS_PATH_LABEL],
                          SHARDED_STATS_PREFIX),
                  _SCHEMA_KEY:
                      os.path.join(
                          stats_output_paths[
                              labels.PRE_TRANSFORM_OUTPUT_SCHEMA_PATH_LABEL],
                          _SCHEMA_FILE)
              }
            else:
              pre_transform_feature_stats_loc = os.path.join(
                  transform_output_path,
                  tft.TFTransformOutput.PRE_TRANSFORM_FEATURE_STATS_PATH)

            (stats_input
             | 'FlattenAnalysisDatasets' >> beam.Flatten(pipeline=pipeline)
             | 'GenerateStats[FlattenedAnalysisDataset]' >>
             self._GenerateAndMaybeValidateStats(
                 pre_transform_feature_stats_loc,
                 stats_options=pre_transform_stats_options,
                 enable_validation=False))

          # transform_data_list is a superset of analyze_data_list, we pay the
          # cost to read the same dataset (analyze_data_list) again here to
          # prevent certain beam runner from doing large temp materialization.
          for dataset in transform_data_list:
            infix = 'TransformIndex{}'.format(dataset.index)
            dataset.standardized = (
                pipeline | 'TFXIOReadAndDecode[{}]'.format(infix) >>
                dataset.tfxio.BeamSource(desired_batch_size))
            (dataset.transformed, metadata) = (
                ((dataset.standardized, dataset.tfxio.TensorAdapterConfig()),
                 transform_fn)
                | 'Transform[{}]'.format(infix) >>
                tft_beam.TransformDataset(output_record_batches=True))

          _, metadata = transform_fn

          # TODO(b/70392441): Retain tf.Metadata (e.g., IntDomain) in
          # schema. Currently input dataset schema only contains dtypes,
          # and other metadata is dropped due to roundtrip to tensors.
          transformed_schema_proto = metadata.schema

          if not disable_statistics:
            # Aggregated feature stats after transformation.
            for dataset in transform_data_list:
              infix = 'TransformIndex{}'.format(dataset.index)
              dataset.transformed_and_standardized = (
                  dataset.transformed
                  | 'ExtractRecordBatches[{}]'.format(infix) >> beam.Keys())

            post_transform_stats_options = _InvokeStatsOptionsUpdaterFn(
                stats_options_updater_fn,
                stats_options_util.StatsType.POST_TRANSFORM,
                transformed_schema_proto, metadata.asset_map,
                transform_output_path)

            if self._TFDVWriteShardedOutput():
              post_transform_stats_options.experimental_result_partitions = (
                  _SHARDED_OUTPUT_PARTITIONS)
            else:
              if (post_transform_stats_options.experimental_result_partitions !=
                  1):
                raise ValueError('Sharded output disabled requires '
                                 'experimental_result_partitions=1.')

            if stats_output_paths:
              post_transform_feature_stats_loc = {
                  _STATS_KEY:
                      os.path.join(
                          stats_output_paths[
                              labels.POST_TRANSFORM_OUTPUT_STATS_PATH_LABEL],
                          STATS_FILE),
                  _SHARDED_STATS_KEY:
                      os.path.join(
                          stats_output_paths[
                              labels.POST_TRANSFORM_OUTPUT_STATS_PATH_LABEL],
                          SHARDED_STATS_PREFIX),
                  _SCHEMA_KEY:
                      os.path.join(
                          stats_output_paths[
                              labels.POST_TRANSFORM_OUTPUT_SCHEMA_PATH_LABEL],
                          _SCHEMA_FILE),
                  _ANOMALIES_KEY:
                      os.path.join(
                          stats_output_paths[
                              labels
                              .POST_TRANSFORM_OUTPUT_ANOMALIES_PATH_LABEL],
                          _ANOMALIES_FILE)
              }
            else:
              post_transform_feature_stats_loc = os.path.join(
                  transform_output_path,
                  tft.TFTransformOutput.POST_TRANSFORM_FEATURE_STATS_PATH)

            ([
                dataset.transformed_and_standardized
                for dataset in transform_data_list
            ]
             | 'FlattenTransformedDatasets' >> beam.Flatten(pipeline=pipeline)
             | 'WaitForTransformWrite' >> beam.Map(
                 lambda x, completion: x,
                 completion=beam.pvalue.AsSingleton(completed_transform))
             | 'GenerateAndValidateStats[FlattenedTransformedDatasets]' >>
             self._GenerateAndMaybeValidateStats(
                 post_transform_feature_stats_loc,
                 stats_options=post_transform_stats_options,
                 enable_validation=True))

            if per_set_stats_output_paths:
              # TODO(b/130885503): Remove duplicate stats gen compute that is
              # done both on a flattened view of the data, and on each span
              # below.
              for dataset in transform_data_list:
                infix = 'TransformIndex{}'.format(dataset.index)
                (dataset.transformed_and_standardized
                 | 'WaitForTransformWrite[{}]'.format(infix) >> beam.Map(
                     lambda x, completion: x,
                     completion=beam.pvalue.AsSingleton(completed_transform))
                 | 'GenerateAndValidateStats[{}]'.format(infix) >>
                 self._GenerateAndMaybeValidateStats(
                     dataset.stats_output_path,
                     stats_options=post_transform_stats_options,
                     enable_validation=True))

          if materialization_format is not None:
            for dataset in transform_data_list:
              infix = 'TransformIndex{}'.format(dataset.index)
              (dataset.transformed
               | 'EncodeAndWrite[{}]'.format(infix) >> self._EncodeAndWrite(
                   schema=transformed_schema_proto,
                   file_format=materialization_format,
                   output_path=dataset.materialize_output_path))

    return _Status.OK()
    # pylint: enable=expression-not-assigned, no-value-for-parameter

  def _RunInPlaceImpl(self, preprocessing_fn: Any, force_tf_compat_v1: bool,
                      metadata: dataset_metadata.DatasetMetadata,
                      typespecs: Dict[str, tf.TypeSpec],
                      transform_output_path: str) -> _Status:
    """Runs a transformation iteration in-place without looking at the data.

    Args:
      preprocessing_fn: The tf.Transform preprocessing_fn.
      force_tf_compat_v1: If True, call Transform's API to use Tensorflow in
        tf.compat.v1 mode.
      metadata: A DatasetMetadata object for the input data.
      typespecs: a Dict[Text, tf.TypeSpec]
      transform_output_path: An absolute path to write the output to.

    Returns:
      Status of the execution.
    """

    logging.debug('Processing an in-place transform')

    raw_metadata_dir = os.path.join(transform_output_path,
                                    tft.TFTransformOutput.RAW_METADATA_DIR)
    metadata_io.write_metadata(metadata, raw_metadata_dir)
    # TODO(b/149997088): Use typespecs for the tf.compat.v1 path as well.
    feature_specs = schema_utils.schema_as_feature_spec(
        metadata.schema).feature_spec
    impl_helper.analyze_in_place(preprocessing_fn, force_tf_compat_v1,
                                 feature_specs, typespecs,
                                 transform_output_path)

    return _Status.OK()

  # TODO(b/114444977): Remove the unused can_process_jointly argument.
  def _MakeDatasetList(
      self,
      file_patterns: Sequence[Union[str, int]],
      file_formats: Sequence[Union[str, int]],
      data_format: int,
      data_view_uri: Optional[str],
      can_process_jointly: bool,  # pylint: disable=unused-argument
      stats_output_paths: Optional[Sequence[str]] = None,
      materialize_output_paths: Optional[Sequence[str]] = None,
  ) -> List[_Dataset]:
    """Makes a list of Dataset from the given `file_patterns`.

    Args:
      file_patterns: A list of file patterns where each pattern corresponds to
        one `_Dataset`.
      file_formats: A list of file format where each format corresponds to one
        `_Dataset`. Must have the same size as `file_patterns`.
      data_format: The data format of the datasets. One of the enums from
        example_gen_pb2.PayloadFormat.
      data_view_uri: URI to the DataView to be used to parse the data.
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
        _Dataset(p, f, data_format, data_view_uri, s, m)
        for p, f, s, m in zip(file_patterns, file_formats, stats_output_paths,
                              materialize_output_paths)
    ]
    result = sorted(datasets, key=lambda dataset: dataset.dataset_key)
    for index, dataset in enumerate(result):
      dataset.index = index
    return result

  @classmethod
  def _ShouldDecodeAsRawExample(cls, data_format: int,
                                data_view_uri: Optional[str],
                                schema: schema_pb2.Schema) -> bool:
    """Returns true if data format should be decoded as raw example.

    Args:
      data_format: One of the enums from example_gen_pb2.PayloadFormat.
      data_view_uri: URI to the DataView to be used to parse the data.
      schema: A schema_pb2.Schema for the input data.

    Returns:
      True if data format should be decoded as raw example.
    """
    return (cls._DecodesSequenceExamplesAsRawRecords(data_format, schema) or
            (cls._IsDataFormatProto(data_format) and data_view_uri is None))

  @staticmethod
  def _IsDataFormatSequenceExample(data_format: int) -> bool:
    """Returns true if data format is sequence example.

    Args:
      data_format: One of the enums from example_gen_pb2.PayloadFormat.

    Returns:
      True if data format is sequence example.
    """
    return data_format == example_gen_pb2.FORMAT_TF_SEQUENCE_EXAMPLE

  @staticmethod
  def _IsDataFormatProto(data_format: int) -> bool:
    """Returns true if data format is protocol buffer.

    Args:
      data_format: One of the enums from example_gen_pb2.PayloadFormat.

    Returns:
      True if data format is protocol buffer.
    """
    return data_format == example_gen_pb2.FORMAT_PROTO

  def _GetDesiredBatchSize(self, data_format: int,
                           schema: schema_pb2.Schema) -> Optional[int]:
    """Returns batch size.

    Args:
      data_format: One of the enums from example_gen_pb2.PayloadFormat.
      schema: A schema for the input data.

    Returns:
      Batch size or None.
    """
    if self._DecodesSequenceExamplesAsRawRecords(data_format, schema):
      return 1
    return None

  @classmethod
  def _DecodesSequenceExamplesAsRawRecords(cls, data_format: int,
                                           schema: schema_pb2.Schema) -> bool:
    """Indicates whether data format is tf.SequenceExample and it should be decoded as raw records.

    Implemented to allow backward compatibility with users exercising hack
    implementation of SequenceExamples.

    Args:
      data_format: One of the enums from example_gen_pb2.PayloadFormat.
      schema: A schema_pb2.Schema for the input data.

    Returns:
      True if tensor_representation_group absent in Schema for SequenceExample
      indicating processing SequenceExample as raw records, else False,
      indicating native execution.
    """

    return (cls._IsDataFormatSequenceExample(data_format) and
            not bool(schema.tensor_representation_group))

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

  def _CreateTFXIO(self, dataset: _Dataset,
                   schema: schema_pb2.Schema) -> tfxio_module.TFXIO:
    """Creates a TFXIO instance for `dataset`."""
    read_as_raw_records = self._ShouldDecodeAsRawExample(
        dataset.data_format, dataset.data_view_uri, schema)
    return tfxio_utils.make_tfxio(
        file_pattern=dataset.file_pattern,
        file_format=dataset.file_format,
        telemetry_descriptors=_TELEMETRY_DESCRIPTORS,
        payload_format=dataset.data_format,
        data_view_uri=dataset.data_view_uri,
        schema=schema,
        read_as_raw_records=read_as_raw_records)

  def _AssertSameTFXIOSchema(self, datasets: Sequence[_Dataset]) -> None:
    if not datasets:
      return
    for dataset in datasets[1:]:
      assert (datasets[0].tfxio.ArrowSchema().equals(
          dataset.tfxio.ArrowSchema()))

  @staticmethod
  def _GetTFXIOPassthroughKeys() -> Optional[Set[str]]:
    """Always returns None."""
    return None

  # TODO(b/215448985): Remove this once sharded stats are written by default.
  @staticmethod
  def _TFDVWriteShardedOutput():
    return False
