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
"""Executor for TensorFlow Transform."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import hashlib
import os
from typing import Any, Callable, Dict, Generator, Iterable, List, Mapping, Optional, Sequence, Set, Text, Tuple, Union

import absl
import apache_beam as beam
import pyarrow as pa
import tensorflow as tf
import tensorflow_data_validation as tfdv
import tensorflow_transform as tft
from tensorflow_transform import impl_helper
from tensorflow_transform import tf2_utils
import tensorflow_transform.beam as tft_beam
from tensorflow_transform.beam import analyzer_cache
from tensorflow_transform.beam import common as tft_beam_common
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import metadata_io
from tensorflow_transform.tf_metadata import schema_utils
from tfx import types
from tfx.components.transform import labels
from tfx.components.transform import stats_options_util
from tfx.components.util import tfxio_utils
from tfx.components.util import value_utils
from tfx.dsl.components.base import base_executor
from tfx.dsl.io import fileio
from tfx.proto import example_gen_pb2
from tfx.proto import transform_pb2
from tfx.types import artifact_utils
from tfx.types import standard_component_specs
from tfx.utils import import_utils
from tfx.utils import io_utils
from tfx.utils import json_utils
from tfx.utils import proto_utils
import tfx_bsl
from tfx_bsl.tfxio import tfxio as tfxio_module

from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


# Key for temp path, for internal use only.
TEMP_PATH_KEY = 'temp_path'

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

_TRANSFORM_COMPONENT_DESCRIPTOR = 'Transform'
_TELEMETRY_DESCRIPTORS = [_TRANSFORM_COMPONENT_DESCRIPTOR]

# TODO(b/37788560): Increase this max, based on results of experimentation with
# many non-packable analyzers on our benchmarks.
_MAX_ESTIMATED_STAGES_COUNT = 20000


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
  _FILE_PATTERN_SUFFIX_LENGTH = 6

  def __init__(self, file_pattern: Text,
               file_format: Union[Text, int],
               data_format: int,
               data_view_uri: Optional[Text],
               stats_output_path: Optional[Text] = None,
               materialize_output_path: Optional[Text] = None):
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
    self._transformed_and_serialized = None
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
  def transformed_and_serialized(self):
    assert self._transformed_and_serialized is not None
    return self._transformed_and_serialized

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

  @transformed_and_serialized.setter
  def transformed_and_serialized(self, val):
    self._transformed_and_serialized = val

  @transformed_and_standardized.setter
  def transformed_and_standardized(self, val):
    self._transformed_and_standardized = val

  @tfxio.setter
  def tfxio(self, val):
    self._tfxio = val


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


def _InvokeStatsOptionsUpdaterFn(
    stats_options_updater_fn: Callable[
        [stats_options_util.StatsType, tfdv.StatsOptions], tfdv.StatsOptions],
    stats_type: stats_options_util.StatsType,
    schema: Optional[schema_pb2.Schema] = None,
    asset_map: Optional[Dict[Text, Text]] = None,
    transform_output_path: Optional[Text] = None) -> tfdv.StatsOptions:
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
  return stats_options_updater_fn(stats_type, tfdv.StatsOptions(**options))


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
          'preprocessing_fn' function will be loaded.
        - preprocessing_fn: The module path to a python function that
          implements 'preprocessing_fn'. Exactly one of 'module_file' and
          'preprocessing_fn' should be set.
        - splits_config: A transform_pb2.SplitsConfig instance, providing splits
          that should be analyzed and splits that should be transformed. Note
          analyze and transform splits can have overlap. Default behavior (when
          splits_config is not set) is analyze the 'train' split and transform
          all splits. If splits_config is set, analyze cannot be empty.
        - force_tf_compat_v1: Whether to use TF in compat.v1 mode
          irrespective of installed/enabled TF behaviors.

    Returns:
      None
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    splits_config = transform_pb2.SplitsConfig()
    if exec_properties.get(standard_component_specs.SPLITS_CONFIG_KEY, None):
      proto_utils.json_to_proto(
          exec_properties[standard_component_specs.SPLITS_CONFIG_KEY],
          splits_config)
      if not splits_config.analyze:
        raise ValueError('analyze cannot be empty when splits_config is set.')
    else:
      splits_config.analyze.append('train')

      # All input artifacts should have the same set of split names.
      split_names = artifact_utils.decode_split_names(
          input_dict[standard_component_specs.EXAMPLES_KEY][0].split_names)
      split_names_set = set(split_names)

      for artifact in input_dict[standard_component_specs.EXAMPLES_KEY]:
        artifact_split_names = artifact_utils.decode_split_names(
            artifact.split_names)
        if split_names_set != set(artifact_split_names):
          raise ValueError(
              'Not all input artifacts have the same split names: (%s, %s)' %
              (split_names, artifact_split_names))

      splits_config.transform.extend(split_names)
      absl.logging.info(
          "Analyze the 'train' split and transform all splits when "
          'splits_config is not set.')

    payload_format, data_view_uri = (
        tfxio_utils.resolve_payload_format_and_data_view_uri(
            input_dict[standard_component_specs.EXAMPLES_KEY]))
    schema_file = io_utils.get_only_uri_in_dir(
        artifact_utils.get_single_uri(
            input_dict[standard_component_specs.SCHEMA_KEY]))
    transform_output = artifact_utils.get_single_uri(
        output_dict[standard_component_specs.TRANSFORM_GRAPH_KEY])

    temp_path = os.path.join(transform_output, _TEMP_DIR_IN_TRANSFORM_OUTPUT)
    absl.logging.debug('Using temp path %s for tft.beam', temp_path)

    analyze_data_paths = []
    for split in splits_config.analyze:
      data_uris = artifact_utils.get_split_uris(
          input_dict[standard_component_specs.EXAMPLES_KEY], split)
      for data_uri in data_uris:
        analyze_data_paths.append(io_utils.all_files_pattern(data_uri))

    transform_data_paths = []
    materialize_output_paths = []
    if output_dict.get(
        standard_component_specs.TRANSFORMED_EXAMPLES_KEY) is not None:
      for transformed_example_artifact in output_dict[
          standard_component_specs.TRANSFORMED_EXAMPLES_KEY]:
        transformed_example_artifact.split_names = (
            artifact_utils.encode_split_names(list(splits_config.transform)))

      for split in splits_config.transform:
        data_uris = artifact_utils.get_split_uris(
            input_dict[standard_component_specs.EXAMPLES_KEY], split)
        for data_uri in data_uris:
          transform_data_paths.append(io_utils.all_files_pattern(data_uri))

        transformed_example_uris = artifact_utils.get_split_uris(
            output_dict[standard_component_specs.TRANSFORMED_EXAMPLES_KEY],
            split)
        for output_uri in transformed_example_uris:
          materialize_output_paths.append(
              os.path.join(output_uri, _DEFAULT_TRANSFORMED_EXAMPLES_PREFIX))

    def _GetCachePath(label, params_dict):
      if params_dict.get(label) is None:
        return None
      else:
        return artifact_utils.get_single_uri(params_dict[label])

    force_tf_compat_v1 = bool(
        exec_properties.get(standard_component_specs.FORCE_TF_COMPAT_V1_KEY, 1))
    if force_tf_compat_v1 and not tf2_utils.use_tf_compat_v1(False):
      absl.logging.warning(
          'The default value of `force_tf_compat_v1` will change in a future '
          'release from `True` to `False`. Since this pipeline has TF 2 '
          'behaviors enabled, Transform will use native TF 2 at that point. You'
          ' can test this behavior now by passing `force_tf_compat_v1=False` '
          'or disable it by explicitly setting `force_tf_compat_v1=True` in '
          'the Transform component.')

    label_inputs = {
        labels.COMPUTE_STATISTICS_LABEL:
            False,
        labels.SCHEMA_PATH_LABEL:
            schema_file,
        labels.EXAMPLES_DATA_FORMAT_LABEL:
            payload_format,
        labels.DATA_VIEW_LABEL:
            data_view_uri,
        labels.ANALYZE_DATA_PATHS_LABEL:
            analyze_data_paths,
        labels.ANALYZE_PATHS_FILE_FORMATS_LABEL: [labels.FORMAT_TFRECORD] *
                                                 len(analyze_data_paths),
        labels.TRANSFORM_DATA_PATHS_LABEL:
            transform_data_paths,
        labels.TRANSFORM_PATHS_FILE_FORMATS_LABEL: [labels.FORMAT_TFRECORD] *
                                                   len(transform_data_paths),
        labels.MODULE_FILE:
            exec_properties.get(standard_component_specs.MODULE_FILE_KEY, None),
        labels.PREPROCESSING_FN:
            exec_properties.get(standard_component_specs.PREPROCESSING_FN_KEY,
                                None),
        labels.CUSTOM_CONFIG:
            exec_properties.get(standard_component_specs.CUSTOM_CONFIG_KEY,
                                None),
        labels.FORCE_TF_COMPAT_V1_LABEL:
            force_tf_compat_v1,
    }
    cache_input = _GetCachePath(standard_component_specs.ANALYZER_CACHE_KEY,
                                input_dict)
    if cache_input is not None:
      label_inputs[labels.CACHE_INPUT_PATH_LABEL] = cache_input

    label_outputs = {
        labels.TRANSFORM_METADATA_OUTPUT_PATH_LABEL: transform_output,
        labels.TRANSFORM_MATERIALIZE_OUTPUT_PATHS_LABEL:
            materialize_output_paths,
        labels.TEMP_OUTPUT_LABEL: str(temp_path),
    }
    cache_output = _GetCachePath(
        standard_component_specs.UPDATED_ANALYZER_CACHE_KEY, output_dict)
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
  def _IncrementPipelineMetrics(pipeline: beam.Pipeline,
                                total_columns_count: int,
                                analyze_columns_count: int,
                                transform_columns_count: int,
                                analyze_paths_count: int):
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
      beam.metrics.Metrics.counter(
          tft_beam_common.METRICS_NAMESPACE,
          'analyze_paths_count').inc(analyze_paths_count)
      return beam.pvalue.PDone(pipeline)

    return (
        pipeline
        | 'CreateSole' >> beam.Create([None])
        | 'Count' >> beam.Map(_MakeAndIncrementCounters))

  @staticmethod
  @beam.ptransform_fn
  @beam.typehints.with_input_types(Tuple[Optional[bytes], bytes])
  @beam.typehints.with_output_types(beam.pvalue.PDone)
  def _WriteExamples(pcoll: beam.pvalue.PCollection, file_format: Text,
                     transformed_example_path: Text) -> beam.pvalue.PDone:
    """Writes transformed examples compressed in gzip format.

    Args:
      pcoll: PCollection of serialized transformed examples.
      file_format: The output file format.
      transformed_example_path: path to write to.

    Returns:
      beam.pvalue.PDone.
    """
    assert file_format == labels.FORMAT_TFRECORD, file_format

    # TODO(b/139538871): Implement telemetry, on top of pa.Table once available.
    return (
        pcoll
        | 'Values' >> beam.Values()
        | 'Write' >> beam.io.WriteToTFRecord(
            transformed_example_path, file_name_suffix='.gz'))

  def _GetSchema(self, schema_path: Text) -> schema_pb2.Schema:
    """Gets a tf.metadata schema.

    Args:
      schema_path: Path to schema file.

    Returns:
      A tf.metadata schema.
    """
    schema_reader = io_utils.SchemaReader()
    return schema_reader.read(schema_path)

  def _ReadMetadata(self, data_format: int,
                    schema_path: Text) -> dataset_metadata.DatasetMetadata:
    """Returns a dataset_metadata.DatasetMetadata for the input data.

    Args:
      data_format: The data format of the dataset. One of the enums from
        example_gen_pb2.PayloadFormat.
      schema_path: path to schema file.

    Returns:
      A dataset_metadata.DatasetMetadata representing the provided set of
          columns.
    """

    if (self._IsDataFormatSequenceExample(data_format) or
        self._IsDataFormatProto(data_format)):
      return dataset_metadata.DatasetMetadata(_RAW_EXAMPLE_SCHEMA)
    schema_proto = self._GetSchema(schema_path)
    # For compatibility with tensorflow_transform 0.13 and 0.14, we create and
    # then update a DatasetMetadata.
    result = dataset_metadata.DatasetMetadata(dataset_schema.Schema({}))
    _GetSchemaProto(result).CopyFrom(schema_proto)
    return result

  @staticmethod
  @beam.ptransform_fn
  @beam.typehints.with_input_types(pa.RecordBatch)
  @beam.typehints.with_output_types(beam.pvalue.PDone)
  def _GenerateStats(
      pcoll: beam.pvalue.PCollection,
      stats_output_path: Text,
      stats_options: tfdv.StatsOptions,
  ) -> beam.pvalue.PDone:
    """Generates statistics.

    Args:
      pcoll: PCollection of examples.
      stats_output_path: path where statistics is written to.
      stats_options: An instance of `tfdv.StatsOptions()` used when computing
        statistics.

    Returns:
      beam.pvalue.PDone.
    """
    def _FilterInternalColumn(record_batch):
      filtered_column_names = []
      filtered_columns = []
      for i, column_name in enumerate(record_batch.schema.names):
        if column_name != _TRANSFORM_INTERNAL_FEATURE_FOR_KEY:
          filtered_column_names.append(column_name)
          filtered_columns.append(record_batch.column(i))
      return pa.RecordBatch.from_arrays(filtered_columns, filtered_column_names)

    pcoll |= 'FilterInternalColumn' >> beam.Map(_FilterInternalColumn)
    # pylint: disable=no-value-for-parameter
    return (pcoll
            | 'GenerateStatistics' >> tfdv.GenerateStatistics(stats_options)
            | 'WriteStats' >> Executor._WriteStats(stats_output_path,
                                                   stats_options.schema))

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

  # TODO(b/160799442, b/130807807): Two code paths are still using this:
  # 1) post-transform stats (we convert from tf.example to recordbatch)
  # 2) sequence example pre-transform stats (we decode sequence example as
  # tf.example).
  # Once 1) and 2) are addressed this can be removed.
  @staticmethod
  @beam.ptransform_fn
  @beam.typehints.with_input_types(Tuple[Optional[bytes], bytes])
  @beam.typehints.with_output_types(pa.RecordBatch)
  def _ToArrowRecordBatches(
      pcoll: beam.pvalue.PCollection,
      schema: Optional[schema_pb2.Schema]) -> beam.pvalue.PCollection:
    """Converts serialized examples to Arrow RecordBatches.

    Args:
      pcoll: PCollection of Transformed data.
      schema: schema.

    Returns:
      PCollection of `DatasetFeatureStatisticsList`.
    """
    kwargs = tfx_bsl.coders.batch_util.GetBatchElementsKwargs(
        tft_beam.Context.get_desired_batch_size())
    return (
        pcoll
        | 'Values' >> beam.Values()
        | 'BatchElements' >> beam.BatchElements(**kwargs)
        | 'ToArrowRecordBatches' >> beam.ParDo(
            Executor._ToArrowRecordBatchesFn(schema)))

  @staticmethod
  @beam.ptransform_fn
  @beam.typehints.with_input_types(statistics_pb2.DatasetFeatureStatisticsList)
  @beam.typehints.with_output_types(beam.pvalue.PDone)
  def _WriteStats(
      pcollection_stats: beam.pvalue.PCollection,
      stats_output_path: Text,
      schema: Optional[schema_pb2.Schema] = None) -> beam.pvalue.PDone:
    """Writs Statistics outputs.

    Args:
      pcollection_stats: pcollection of statistics.
      stats_output_path: path to write statistics and the schema. The schema
        used to generate the statistics (if any) will be placed within the same
        subdirectory.
      schema: the schema used to generate the statistics.

    Returns:
      beam.pvalue.PDone.
    """
    stats_dir = os.path.dirname(stats_output_path)
    fileio.makedirs(stats_dir)
    if schema is not None:
      io_utils.write_pbtxt_file(os.path.join(stats_dir, 'schema.pbtxt'), schema)

    # TODO(b/117601471): Replace with utility method to write stats.
    return (pcollection_stats | 'Write' >> beam.io.WriteToText(
        stats_output_path,
        append_trailing_newlines=False,
        shard_name_template='',  # To force unsharded output.
        coder=beam.coders.ProtoCoder(
            statistics_pb2.DatasetFeatureStatisticsList)))

  @beam.typehints.with_input_types(Dict[Text, Any], schema=schema_pb2.Schema)
  @beam.typehints.with_output_types(Tuple[Optional[bytes], bytes])
  class _EncodeAsSerializedExamples(beam.DoFn):
    """Encodes data as serialized tf.Examples based on the given metadata."""

    def __init__(self):
      self._coder = None

    def process(self, element: Dict[Text, Any], schema: schema_pb2.Schema
               ) -> Generator[Tuple[Any, Any], None, None]:
      if self._coder is None:
        self._coder = tft.coders.ExampleProtoCoder(schema, serialized=True)

      # Make sure that the synthetic key feature doesn't get encoded.
      key = element.get(_TRANSFORM_INTERNAL_FEATURE_FOR_KEY, None)
      if key is not None:
        element = element.copy()
        del element[_TRANSFORM_INTERNAL_FEATURE_FOR_KEY]
      yield (key, self._coder.encode(element))

  @beam.typehints.with_input_types(beam.Pipeline)
  class _OptimizeRun(beam.PTransform):
    """Utilizes TFT cache if applicable and removes unused datasets."""

    # pyformat: disable
    def __init__(self,
                 input_cache_dir: Text,
                 output_cache_dir: Text,
                 analyze_data_list: List[_Dataset],
                 typespecs: Mapping[Text, tf.TypeSpec],
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
        self, pipeline
    ) -> Tuple[Dict[Text, Optional[_Dataset]], Optional[Dict[Text, Dict[
        Text, beam.pvalue.PCollection]]]]:
      # TODO(b/170304777): Remove this Create once the issue is fixed in beam.
      # Forcing beam to treat this PTransform as non-primitive.
      _ = pipeline | 'WorkaroundForBug170304777' >> beam.Create([None])

      dataset_keys_list = [
          dataset.dataset_key for dataset in self._analyze_data_list
      ]
      # TODO(b/37788560): Remove this restriction when a greater number of
      # stages can be handled efficiently.
      cache_entry_keys = (
          tft_beam.analysis_graph_builder.get_analysis_cache_entry_keys(
              self._preprocessing_fn, self._feature_spec_or_typespec,
              dataset_keys_list, self._force_tf_compat_v1))
      # We estimate the number of stages in the pipeline to be roughly:
      # analyzers * analysis_paths * 10.
      if (len(cache_entry_keys) * len(dataset_keys_list) * 10 >
          _MAX_ESTIMATED_STAGES_COUNT):
        absl.logging.warning(
            'Disabling cache because otherwise the number of stages might be '
            'too high ({} analyzers, {} analysis paths)'.format(
                len(cache_entry_keys), len(dataset_keys_list)))
        # Returning None as the input cache here disables both input and output
        # cache.
        return ({d.dataset_key: d for d in self._analyze_data_list}, None)

      if self._input_cache_dir is not None:
        absl.logging.info('Reading the following analysis cache entry keys: %s',
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

      return (new_analyze_data_dict, input_cache)

  def _MaybeBindCustomConfig(self, inputs: Mapping[Text, Any],
                             fn: Any) -> Callable[..., Any]:
    # For compatibility, only bind custom config if it's in the signature.
    if value_utils.FunctionHasArg(fn, labels.CUSTOM_CONFIG):
      custom_config_json = value_utils.GetSoleValue(inputs,
                                                    labels.CUSTOM_CONFIG)
      custom_config = (json_utils.loads(custom_config_json)
                       if custom_config_json else {}) or {}
      fn = functools.partial(fn, custom_config=custom_config)
    return fn

  def _GetPreprocessingFn(
      self, inputs: Mapping[Text, Any],
      unused_outputs: Mapping[Text, Any]) -> Callable[..., Any]:
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
    has_module_file = bool(
        value_utils.GetSoleValue(inputs, labels.MODULE_FILE, strict=False))
    has_preprocessing_fn = bool(
        value_utils.GetSoleValue(inputs, labels.PREPROCESSING_FN, strict=False))

    if has_module_file == has_preprocessing_fn:
      raise ValueError(
          'Neither or both of MODULE_FILE and PREPROCESSING_FN have been '
          'supplied in inputs.')

    if has_module_file:
      fn = import_utils.import_func_from_source(
          value_utils.GetSoleValue(inputs, labels.MODULE_FILE),
          standard_component_specs.PREPROCESSING_FN_KEY)
    else:
      preprocessing_fn_path_split = value_utils.GetSoleValue(
          inputs, labels.PREPROCESSING_FN).split('.')
      fn = import_utils.import_func_from_module(
          '.'.join(preprocessing_fn_path_split[0:-1]),
          preprocessing_fn_path_split[-1])

    return self._MaybeBindCustomConfig(inputs, fn)

  def _GetStatsOptionsUpdaterFn(
      self, inputs: Mapping[Text, Any]
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
    has_module_file = bool(
        value_utils.GetSoleValue(inputs, labels.MODULE_FILE, strict=False))

    fn = None
    if has_module_file:
      # Users do not have to define a stats_options_updater_fn. Return None if
      # they do not.
      try:
        fn = import_utils.import_func_from_source(
            value_utils.GetSoleValue(inputs, labels.MODULE_FILE),
            'stats_options_updater_fn')
        fn = self._MaybeBindCustomConfig(inputs, fn)
      except AttributeError:
        return None

    return fn

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
        - labels.MODULE_FILE: Path to a Python module that contains the
          preprocessing_fn, optional.
        - labels.PREPROCESSING_FN: Path to a Python function that implements
          preprocessing_fn, optional.
        - labels.CUSTOM_CONFIG: Dictionary of additional parameters for
          preprocessing_fn, optional.
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
      status_file: Where the status should be written (not yet implemented)
    """
    del status_file  # unused

    absl.logging.debug(
        'Inputs to executor.Transform function: {}'.format(inputs))
    absl.logging.debug(
        'Outputs to executor.Transform function: {}'.format(outputs))

    compute_statistics = value_utils.GetSoleValue(
        inputs, labels.COMPUTE_STATISTICS_LABEL)
    transform_output_path = value_utils.GetSoleValue(
        outputs, labels.TRANSFORM_METADATA_OUTPUT_PATH_LABEL)
    raw_examples_data_format = value_utils.GetSoleValue(
        inputs, labels.EXAMPLES_DATA_FORMAT_LABEL)
    schema = value_utils.GetSoleValue(inputs, labels.SCHEMA_PATH_LABEL)
    input_dataset_metadata = self._ReadMetadata(raw_examples_data_format,
                                                schema)
    materialize_output_paths = value_utils.GetValues(
        outputs, labels.TRANSFORM_MATERIALIZE_OUTPUT_PATHS_LABEL)
    preprocessing_fn = self._GetPreprocessingFn(inputs, outputs)
    stats_options_updater_fn = self._GetStatsOptionsUpdaterFn(inputs)
    per_set_stats_output_paths = value_utils.GetValues(
        outputs, labels.PER_SET_STATS_OUTPUT_PATHS_LABEL)
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

    absl.logging.debug('Force tf.compat.v1: %s', force_tf_compat_v1)
    absl.logging.debug('Analyze data patterns: %s',
                       list(enumerate(analyze_data_paths)))
    absl.logging.debug('Transform data patterns: %s',
                       list(enumerate(transform_data_paths)))
    absl.logging.debug('Transform materialization output paths: %s',
                       list(enumerate(materialize_output_paths)))
    absl.logging.debug('Transform output path: %s', transform_output_path)

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

    if (not compute_statistics and not materialize_output_paths and
        stats_options_updater_fn is None):
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
                      output_cache_dir, compute_statistics,
                      per_set_stats_output_paths, materialization_format,
                      len(analyze_data_paths))
  # TODO(b/122478841): Writes status to status file.

  def _RunBeamImpl(self, analyze_data_list: List[_Dataset],
                   transform_data_list: List[_Dataset], preprocessing_fn: Any,
                   stats_options_updater_fn: Callable[
                       [stats_options_util.StatsType, tfdv.StatsOptions],
                       tfdv.StatsOptions], force_tf_compat_v1: bool,
                   input_dataset_metadata: dataset_metadata.DatasetMetadata,
                   transform_output_path: Text, raw_examples_data_format: int,
                   temp_path: Text, input_cache_dir: Optional[Text],
                   output_cache_dir: Optional[Text], compute_statistics: bool,
                   per_set_stats_output_paths: Sequence[Text],
                   materialization_format: Optional[Text],
                   analyze_paths_count: int) -> _Status:
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
      compute_statistics: A bool indicating whether or not compute statistics.
      per_set_stats_output_paths: Paths to per-set statistics output. If empty,
        per-set statistics is not produced.
      materialization_format: A string describing the format of the materialized
        data or None if materialization is not enabled.
      analyze_paths_count: An integer, the number of paths that should be used
        for analysis.

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

    transform_input_columns = tft.get_transform_input_columns(
        preprocessing_fn,
        unprojected_typespecs,
        force_tf_compat_v1=force_tf_compat_v1)
    # Use the same dataset (same columns) for AnalyzeDataset and computing
    # pre-transform stats so that the data will only be read once for these
    # two operations.
    if compute_statistics:
      analyze_input_columns = list(
          set(list(analyze_input_columns) + list(transform_input_columns)))

    for d in analyze_data_list:
      d.tfxio = d.tfxio.Project(analyze_input_columns)

    self._AssertSameTFXIOSchema(analyze_data_list)
    analyze_data_tensor_adapter_config = (
        analyze_data_list[0].tfxio.TensorAdapterConfig())

    for d in transform_data_list:
      d.tfxio = d.tfxio.Project(transform_input_columns)

    desired_batch_size = self._GetDesiredBatchSize(raw_examples_data_format)

    with self._CreatePipeline(transform_output_path) as pipeline:
      with tft_beam.Context(
          temp_dir=temp_path,
          desired_batch_size=desired_batch_size,
          passthrough_keys=self._GetTFXIOPassthroughKeys(),
          use_deep_copy_optimization=True,
          force_tf_compat_v1=force_tf_compat_v1):
        # pylint: disable=expression-not-assigned
        # pylint: disable=no-value-for-parameter
        _ = (
            pipeline
            | 'IncrementPipelineMetrics' >> self._IncrementPipelineMetrics(
                len(unprojected_typespecs), len(analyze_input_columns),
                len(transform_input_columns), analyze_paths_count))

        (new_analyze_data_dict, input_cache) = (
            pipeline
            | 'OptimizeRun' >> self._OptimizeRun(
                input_cache_dir, output_cache_dir,
                analyze_data_list, unprojected_typespecs, preprocessing_fn,
                self._GetCacheSource(), force_tf_compat_v1))

        if input_cache:
          absl.logging.debug('Analyzing data with cache.')

        full_analyze_dataset_keys_list = [
            dataset.dataset_key for dataset in analyze_data_list
        ]

        # Removing unneeded datasets if they won't be needed for statistics or
        # materialization.
        if materialization_format is None and not compute_statistics:
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

        for dataset in analyze_data_list:
          infix = 'AnalysisIndex{}'.format(dataset.index)
          dataset.standardized = (
              pipeline
              | 'TFXIOReadAndDecode[{}]'.format(infix) >>
              dataset.tfxio.BeamSource(desired_batch_size))

        input_analysis_data = {}
        for key, dataset in new_analyze_data_dict.items():
          input_analysis_data[key] = (
              None if dataset is None else dataset.standardized)

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
          absl.logging.debug('Using existing cache in: %s', input_cache_dir)
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

          # TODO(b/157479287, b/171165988): Remove this condition when beam 2.26
          # is used.
          if cache_output:
            (cache_output
             | 'WriteCache' >> analyzer_cache.WriteAnalysisCacheToFS(
                 pipeline=pipeline,
                 cache_base_dir=output_cache_dir,
                 sink=self._GetCacheSink(),
                 dataset_keys=full_analyze_dataset_keys_list))

        if compute_statistics or materialization_format is not None:
          # Do not compute pre-transform stats if the input format is raw proto,
          # as StatsGen would treat any input as tf.Example. Note that
          # tf.SequenceExamples are wire-format compatible with tf.Examples.
          if (compute_statistics and
              not self._IsDataFormatProto(raw_examples_data_format)):
            # Aggregated feature stats before transformation.
            pre_transform_feature_stats_path = os.path.join(
                transform_output_path,
                tft.TFTransformOutput.PRE_TRANSFORM_FEATURE_STATS_PATH)

            if self._IsDataFormatSequenceExample(raw_examples_data_format):
              schema_proto = None
            else:
              schema_proto = _GetSchemaProto(input_dataset_metadata)

            if self._IsDataFormatSequenceExample(raw_examples_data_format):
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
                    | 'ExtractRawExampleBatches[{}]'.format(infix) >> beam.Map(
                        _ExtractRawExampleBatches)
                    | 'DecodeSequenceExamplesAsExamplesIntoRecordBatches[{}]'
                    .format(infix) >> beam.ParDo(
                        self._ToArrowRecordBatchesFn(schema_proto)))
            else:
              stats_input = [
                  dataset.standardized for dataset in analyze_data_list]

            pre_transform_stats_options = _InvokeStatsOptionsUpdaterFn(
                stats_options_updater_fn,
                stats_options_util.StatsType.PRE_TRANSFORM, schema_proto)

            (stats_input
             | 'FlattenAnalysisDatasets' >> beam.Flatten(pipeline=pipeline)
             | 'GenerateStats[FlattenedAnalysisDataset]' >> self._GenerateStats(
                 pre_transform_feature_stats_path,
                 stats_options=pre_transform_stats_options))

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
                | 'Transform[{}]'.format(infix) >> tft_beam.TransformDataset())

            dataset.transformed_and_serialized = (
                dataset.transformed
                | 'EncodeAndSerialize[{}]'.format(infix)
                >> beam.ParDo(self._EncodeAsSerializedExamples(),
                              _GetSchemaProto(metadata)))

          if compute_statistics:
            # Aggregated feature stats after transformation.
            _, metadata = transform_fn

            # TODO(b/70392441): Retain tf.Metadata (e.g., IntDomain) in
            # schema. Currently input dataset schema only contains dtypes,
            # and other metadata is dropped due to roundtrip to tensors.
            transformed_schema_proto = _GetSchemaProto(metadata)

            for dataset in transform_data_list:
              infix = 'TransformIndex{}'.format(dataset.index)
              dataset.transformed_and_standardized = (
                  dataset.transformed_and_serialized
                  | 'FromTransformedToArrowRecordBatches[{}]'
                  .format(infix)
                  >> self._ToArrowRecordBatches(
                      schema=transformed_schema_proto))

            post_transform_feature_stats_path = os.path.join(
                transform_output_path,
                tft.TFTransformOutput.POST_TRANSFORM_FEATURE_STATS_PATH)

            post_transform_stats_options = _InvokeStatsOptionsUpdaterFn(
                stats_options_updater_fn,
                stats_options_util.StatsType.POST_TRANSFORM,
                transformed_schema_proto, metadata.asset_map,
                transform_output_path)

            ([dataset.transformed_and_standardized
              for dataset in transform_data_list]
             | 'FlattenTransformedDatasets' >> beam.Flatten()
             | 'WaitForTransformWrite' >> beam.Map(
                 lambda x, completion: x,
                 completion=beam.pvalue.AsSingleton(completed_transform))
             | 'GenerateStats[FlattenedTransformedDatasets]' >>
             self._GenerateStats(
                 post_transform_feature_stats_path,
                 stats_options=post_transform_stats_options))

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
                 | 'GenerateStats[{}]'.format(infix) >> self._GenerateStats(
                     dataset.stats_output_path,
                     stats_options=post_transform_stats_options))

          if materialization_format is not None:
            for dataset in transform_data_list:
              infix = 'TransformIndex{}'.format(dataset.index)
              (dataset.transformed_and_serialized
               | 'Materialize[{}]'.format(infix) >> self._WriteExamples(
                   materialization_format,
                   dataset.materialize_output_path))

    return _Status.OK()

  def _RunInPlaceImpl(self, preprocessing_fn: Any, force_tf_compat_v1: bool,
                      metadata: dataset_metadata.DatasetMetadata,
                      typespecs: Dict[Text, tf.TypeSpec],
                      transform_output_path: Text) -> _Status:
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

    absl.logging.debug('Processing an in-place transform')

    raw_metadata_dir = os.path.join(transform_output_path,
                                    tft.TFTransformOutput.RAW_METADATA_DIR)
    metadata_io.write_metadata(metadata, raw_metadata_dir)
    # TODO(b/149997088): Use typespecs for the tf.compat.v1 path as well.
    feature_specs = schema_utils.schema_as_feature_spec(
        _GetSchemaProto(metadata)).feature_spec
    impl_helper.analyze_in_place(preprocessing_fn, force_tf_compat_v1,
                                 feature_specs, typespecs,
                                 transform_output_path)

    return _Status.OK()

  def _CreatePipeline(
      self, unused_transform_output_path: Text) -> beam.Pipeline:
    """Creates beam pipeline.

    Args:
      unused_transform_output_path: unused.

    Returns:
      Beam pipeline.
    """
    return self._make_beam_pipeline()

  # TODO(b/114444977): Remove the unused can_process_jointly argument.
  def _MakeDatasetList(
      self,
      file_patterns: Sequence[Union[Text, int]],
      file_formats: Sequence[Union[Text, int]],
      data_format: int,
      data_view_uri: Optional[Text],
      can_process_jointly: bool,
      stats_output_paths: Optional[Sequence[Text]] = None,
      materialize_output_paths: Optional[Sequence[Text]] = None,
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

  def _ShouldDecodeAsRawExample(self, data_format: int,
                                data_view_uri: Optional[Text]) -> bool:
    """Returns true if data format should be decoded as raw example.

    Args:
      data_format: One of the enums from example_gen_pb2.PayloadFormat.
      data_view_uri: URI to the DataView to be used to parse the data.

    Returns:
      True if data format should be decoded as raw example.
    """
    return (self._IsDataFormatSequenceExample(data_format) or
            (self._IsDataFormatProto(data_format) and data_view_uri is None))

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

  def _GetDesiredBatchSize(self, data_format: int) -> Optional[int]:
    """Returns batch size.

    Args:
      data_format: One of the enums from example_gen_pb2.PayloadFormat.

    Returns:
      Batch size or None.
    """
    if self._IsDataFormatSequenceExample(data_format):
      return 1
    return None

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
        dataset.data_format, dataset.data_view_uri)
    return tfxio_utils.make_tfxio(
        file_pattern=dataset.file_pattern,
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
  def _GetTFXIOPassthroughKeys() -> Optional[Set[Text]]:
    """Always returns None."""
    return None
