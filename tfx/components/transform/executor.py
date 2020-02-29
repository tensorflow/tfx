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

import os
from typing import Any, Dict, Generator, Iterable, List, Mapping, Optional, Sequence, Text, Tuple, Union

import absl
import apache_beam as beam
import pyarrow as pa
import tensorflow as tf
import tensorflow_data_validation as tfdv
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
import tfx_bsl
from tfx_bsl.tfxio import raw_tf_record
from tfx_bsl.tfxio import tf_example_record
from tfx_bsl.tfxio import tfxio
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2
from tfx import types
from tfx.components.base import base_executor
from tfx.components.transform import labels
from tfx.components.transform import stats_options as transform_stats_options
from tfx.components.transform import messages
from tfx.components.util import value_utils
from tfx.types import artifact_utils
from tfx.utils import import_utils
from tfx.utils import io_utils

# Key for examples in executor input_dict.
EXAMPLES_KEY = 'examples'
# Key for schema in executor input_dict.
SCHEMA_KEY = 'schema'

# Key for temp path, for internal use only.
TEMP_PATH_KEY = 'temp_path'

# Key for transform graph in executor output_dict.
TRANSFORM_GRAPH_KEY = 'transform_graph'
# Key for output model in executor output_dict.
TRANSFORMED_EXAMPLES_KEY = 'transformed_examples'

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

# TODO(b/150159972): Remove this branch after 0.22.
if tft.__version__ < '0.22':
  _create_batched_placeholders = (
      impl_helper.feature_spec_as_batched_placeholders)
else:
  _create_batched_placeholders = impl_helper.batched_placeholders_from_specs


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

  def __init__(self, file_pattern: Text,
               file_format: Union[Text, int],
               data_format: Union[Text, int],
               stats_output_path: Optional[Text] = None,
               materialize_output_path: Optional[Text] = None):
    """Initialize a Dataset.

    Args:
      file_pattern: The file pattern of the dataset.
      file_format: The file format of the dataset.
      data_format: The data format of the dataset.
      stats_output_path: The file path where to write stats for the dataset.
      materialize_output_path: The file path where to write the dataset.
    """
    self._file_pattern = file_pattern
    file_pattern_suffix = os.path.join(
        *file_pattern.split(os.sep)[-self._FILE_PATTERN_SUFFIX_LENGTH:])
    self._dataset_key = analyzer_cache.make_dataset_key(
        # TODO(b/143087691): Remove this replace once TFT 0.16 is released.
        file_pattern_suffix).replace('\\', '-')
    self._file_format = file_format
    self._data_format = data_format
    self._stats_output_path = stats_output_path
    self._materialize_output_path = materialize_output_path
    self._index = None
    self._serialized = None
    self._decoded = None
    self._standardized = None
    self._transformed = None
    self._transformed_and_encoded = None
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
  def file_format(self):
    assert self._file_format
    return self._file_format

  @property
  def serialized(self):
    assert self._serialized is not None
    return self._serialized

  @property
  def decoded(self):
    assert self._decoded is not None
    return self._decoded

  @property
  def standardized(self):
    assert self._standardized is not None
    return self._standardized

  @property
  def transformed(self):
    assert self._transformed is not None
    return self._transformed

  @property
  def transformed_and_encoded(self):
    assert self._transformed_and_encoded is not None
    return self._transformed_and_encoded

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

  @serialized.setter
  def serialized(self, val):
    self._serialized = val

  @decoded.setter
  def decoded(self, val):
    self._decoded = val

  @standardized.setter
  def standardized(self, val):
    self._standardized = val

  @transformed.setter
  def transformed(self, val):
    self._transformed = val

  @transformed_and_encoded.setter
  def transformed_and_encoded(self, val):
    self._transformed_and_encoded = val

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
        - input_data: A list of type `standard_artifacts.Examples` which
          should contain two splits 'train' and 'eval'.
        - schema: A list of type `standard_artifacts.Schema` which should
          contain a single schema artifact.
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
    train_data_uri = artifact_utils.get_split_uri(input_dict[EXAMPLES_KEY],
                                                  'train')
    eval_data_uri = artifact_utils.get_split_uri(input_dict[EXAMPLES_KEY],
                                                 'eval')
    schema_file = io_utils.get_only_uri_in_dir(
        artifact_utils.get_single_uri(input_dict[SCHEMA_KEY]))
    transform_output = artifact_utils.get_single_uri(
        output_dict[TRANSFORM_GRAPH_KEY])
    transformed_train_output = artifact_utils.get_split_uri(
        output_dict[TRANSFORMED_EXAMPLES_KEY], 'train')
    transformed_eval_output = artifact_utils.get_split_uri(
        output_dict[TRANSFORMED_EXAMPLES_KEY], 'eval')
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
        # TODO(b/149754658): switch to True once the TFXIO integration is
        # complete.
        labels.USE_TFXIO_LABEL: False,
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

    return (
        pipeline
        | 'CreateSole' >> beam.Create([None])
        | 'Count' >> beam.Map(_MakeAndIncrementCounters))

  @staticmethod
  @beam.ptransform_fn
  @beam.typehints.with_input_types(beam.Pipeline)
  # TODO(b/38376110): Obviate the first bytes (ie the key part).
  @beam.typehints.with_output_types(Tuple[bytes, bytes])
  def _ReadExamples(
      pipeline: beam.Pipeline, dataset: _Dataset,
      input_dataset_metadata: dataset_metadata.DatasetMetadata
  ) -> beam.pvalue.PCollection:
    """Reads examples from the given `dataset`.

    Args:
      pipeline: beam pipeline.
      dataset: A `_Dataset` object that represents the data to read.
      input_dataset_metadata: A `dataset_metadata.DatasetMetadata`. Not used.

    Returns:
      A PCollection containing KV pairs of bytes.
    """
    del input_dataset_metadata
    assert dataset.file_format == labels.FORMAT_TFRECORD, dataset.file_format

    return (
        pipeline
        | 'Read' >> beam.io.ReadFromTFRecord(
            dataset.file_pattern,
            coder=beam.coders.BytesCoder(),
            # TODO(b/114938612): Eventually remove this override.
            validate=False)
        | 'AddKey' >> beam.Map(lambda x: (None, x)))

  @staticmethod
  @beam.ptransform_fn
  @beam.typehints.with_input_types(Tuple[bytes, tf.train.Example])
  @beam.typehints.with_output_types(beam.pvalue.PDone)
  def _WriteExamples(pcoll: beam.pvalue.PCollection, file_format: Text,
                     transformed_example_path: Text) -> beam.pvalue.PDone:
    """Writes transformed examples compressed in gzip format.

    Args:
      pcoll: PCollection of transformed examples.
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
            transformed_example_path,
            file_name_suffix='.gz',
            coder=beam.coders.ProtoCoder(tf.train.Example)))

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
      Union[Tuple[bytes, Union[bytes, tf.train.Example]],  # Legacy format.
            Union[pa.Table, pa.RecordBatch]])  # TFDV format.
  @beam.typehints.with_output_types(beam.pvalue.PDone)
  def _GenerateStats(
      pcoll: beam.pvalue.PCollection,
      stats_output_path: Text,
      schema: schema_pb2.Schema,
      stats_options: tfdv.StatsOptions,
      # TODO(b/115684207): Remove this and all related code.
      use_tfdv=True,
      # TODO(b/115684207): Remove this and all related code.
      examples_are_serialized=False,
      # TODO(b/149308973): Remove this and all related code.
      input_from_tfxio=False,
  ) -> beam.pvalue.PDone:
    """Generates statistics.

    Args:
      pcoll: PCollection of examples.
      stats_output_path: path where statistics is written to.
      schema: schema.
      stats_options: An instance of `tfdv.StatsOptions()` used when computing
        statistics.
      use_tfdv: whether use TFDV for computing statistics.
      examples_are_serialized: Unused.
      input_from_tfxio: whether the input data is produced from TFXIO.

    Returns:
      beam.pvalue.PDone.
    """
    assert use_tfdv
    del examples_are_serialized  # Unused

    stats_options.schema = schema
    # pylint: disable=no-value-for-parameter
    # TODO(b/149308973): Remove once TFDV starts accepting RecordBatches.
    if input_from_tfxio:
      pcoll |= 'RecordBatchToTable' >> beam.Map(
          lambda rb: pa.Table.from_batches([rb])).with_input_types(
              pa.RecordBatch)
    return (
        pcoll
        | 'GenerateStatistics' >> tfdv.GenerateStatistics(stats_options)
        | 'WriteStats' >> Executor._WriteStats(stats_output_path))

  # TODO(zhuo): Obviate this once TFXIO is used.
  @beam.typehints.with_input_types(List[bytes])
  @beam.typehints.with_output_types(pa.Table)
  class _ToArrowTablesFn(beam.DoFn):
    """Converts a batch of serialized examples to an Arrow Table."""

    __slots__ = ['_serialized_schema', '_decoder']

    def __init__(self, schema: schema_pb2.Schema):
      self._serialized_schema = schema.SerializeToString()  # pylint: disable=assigning-non-slot

    def setup(self):
      self._decoder = (  # pylint: disable=assigning-non-slot
          tfx_bsl.coders.example_coder.ExamplesToRecordBatchDecoder(
              self._serialized_schema))

    def process(self, element: List[bytes]) -> Iterable[pa.Table]:
      yield pa.Table.from_batches([self._decoder.DecodeBatch(element)])

  # TODO(zhuo): Obviate this once TFXIO is used.
  @staticmethod
  @beam.ptransform_fn
  @beam.typehints.with_input_types(Tuple[bytes, bytes])
  @beam.typehints.with_output_types(pa.Table)
  def _FromSerializedToArrowTables(
      pcoll: beam.pvalue.PCollection,
      schema: schema_pb2.Schema) -> beam.pvalue.PCollection:
    """Converts serialized examples to Arrow Tables.

    Args:
      pcoll: PCollection of Transformed data.
      schema: schema.

    Returns:
      PCollection of `DatasetFeatureStatisticsList`.
    """
    kwargs = tfdv.utils.batch_util.GetBeamBatchKwargs(
        tft_beam.Context.get_desired_batch_size())
    return (
        pcoll
        | 'Values' >> beam.Values()
        | 'BatchElements' >> beam.BatchElements(**kwargs)
        | 'ToArrowTables' >> beam.ParDo(Executor._ToArrowTablesFn(schema)))

  @staticmethod
  @beam.ptransform_fn
  @beam.typehints.with_input_types(Dict[Text, Any])
  @beam.typehints.with_output_types(pa.Table)
  def _FromDictsToArrowTables(
      pcoll: beam.pvalue.PCollection,
      schema: schema_pb2.Schema) -> beam.pvalue.PCollection:
    """Converts Dicts to Arrow Tables."""

    # TODO(pachristopher): Remove encoding and batching steps once TFT
    # supports Arrow tables.
    return (
        pcoll
        | 'ToSerializedTFExamples'
        >> beam.ParDo(Executor._EncodeAsExamples(serialized=True), schema
                     ).with_output_types(Tuple[Optional[bytes], bytes])
        | 'FromSerializedToArrowTables'
        >> Executor._FromSerializedToArrowTables(schema=schema))  # pylint: disable=no-value-for-parameter

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
  @beam.typehints.with_input_types(Tuple[bytes, bytes])
  @beam.typehints.with_output_types(Dict[Text, Any])
  def _DecodeInputs(pcoll: beam.pvalue.PCollection,
                    decode_fn: Any) -> beam.pvalue.PCollection:
    """Decodes the given PCollection while handling KV data.

    Args:
      pcoll: PCollection of data.
      decode_fn: Function used to decode data.

    Returns:
      PCollection of decoded data.
    """

    def decode_example(kv: Tuple[Optional[bytes], bytes]) -> Dict[Text, Any]:  # pylint: disable=invalid-name
      """Decodes a single example."""
      (key, value) = kv
      result = decode_fn(value)
      if _TRANSFORM_INTERNAL_FEATURE_FOR_KEY in result:
        raise ValueError('"{}" is a reserved feature name, '
                         'it should not be present in the dataset.'.format(
                             _TRANSFORM_INTERNAL_FEATURE_FOR_KEY))
      result[_TRANSFORM_INTERNAL_FEATURE_FOR_KEY] = key
      return result

    return pcoll | 'ApplyDecodeFn' >> beam.Map(decode_example)

  # TODO(katsiapis): Understand why 'Optional' is needed for the key of the
  # output type.
  @beam.typehints.with_input_types(Dict[Text, Any], schema=schema_pb2.Schema)
  @beam.typehints.with_output_types(Tuple[Optional[bytes],
                                          Union[bytes, tf.train.Example]])
  class _EncodeAsExamples(beam.DoFn):
    """Encodes data as tf.Examples based on the given metadata."""

    __slots__ = ['_serialized', '_coder']

    def __init__(self, serialized):
      self._serialized = serialized  # pylint: disable=assigning-non-slot
      self._coder = None  # pylint: disable=assigning-non-slot

    def process(self, element: Dict[Text, Any], schema: schema_pb2.Schema
               ) -> Generator[Tuple[Any, Any], None, None]:
      if self._coder is None:
        self._coder = tft.coders.ExampleProtoCoder(  # pylint: disable=assigning-non-slot
            schema, serialized=self._serialized)

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
                 feature_spec_or_typespec: Mapping[Text, Any],
                 preprocessing_fn: Any,
                 cache_source: beam.PTransform):
      # pyformat: enable
      self._input_cache_dir = input_cache_dir
      self._output_cache_dir = output_cache_dir
      self._analyze_data_list = analyze_data_list
      self._feature_spec_or_typespec = feature_spec_or_typespec
      self._preprocessing_fn = preprocessing_fn
      self._cache_source = cache_source

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
    ) -> Tuple[Dict[Text, Optional[_Dataset]],
               Optional[Dict[Text, Dict[Text, beam.pvalue.PCollection]]],
               bool]:
      dataset_keys_list = [
          dataset.dataset_key for dataset in self._analyze_data_list
      ]
      if self._input_cache_dir is not None:
        input_cache = (
            pipeline
            | 'ReadCache' >> analyzer_cache.ReadAnalysisCacheFromFS(
                self._input_cache_dir,
                dataset_keys_list,
                source=self._cache_source))
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
        flat_data_required = True
      else:
        filtered_analysis_dataset_keys, flat_data_required = (
            tft_beam.analysis_graph_builder.get_analysis_dataset_keys(
                self._preprocessing_fn, self._feature_spec_or_typespec,
                dataset_keys_list, input_cache))

      new_analyze_data_dict = {}
      for dataset in self._analyze_data_list:
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
        - labels.USE_TFXIO_LABEL: Whether use the TFXIO-based TFT APIs.
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
    use_tfxio = value_utils.GetSoleValue(inputs, labels.USE_TFXIO_LABEL)
    materialize_output_paths = value_utils.GetValues(
        outputs, labels.TRANSFORM_MATERIALIZE_OUTPUT_PATHS_LABEL)
    preprocessing_fn = self._GetPreprocessingFn(inputs, outputs)
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
                                              can_process_analysis_jointly)
    if not analyze_data_list:
      raise ValueError('Analyze data list must not be empty.')

    can_process_transform_jointly = not bool(per_set_stats_output_paths or
                                             materialize_output_paths)
    transform_data_list = self._MakeDatasetList(transform_data_paths,
                                                transform_paths_file_formats,
                                                raw_examples_data_format,
                                                can_process_transform_jointly,
                                                per_set_stats_output_paths,
                                                materialize_output_paths)

    if use_tfxio:
      all_datasets = analyze_data_list + transform_data_list
      for d in all_datasets:
        d.tfxio = self._CreateTFXIO(d, input_dataset_metadata.schema)
      self._AssertSameTFXIOSchema(all_datasets)
      feature_spec_or_typespecs = (
          all_datasets[0].tfxio.TensorAdapter().OriginalTypeSpecs())
    else:
      feature_spec_or_typespecs = schema_utils.schema_as_feature_spec(
          _GetSchemaProto(input_dataset_metadata)).feature_spec

      # NOTE: We disallow an empty schema, which we detect by testing the
      # number of columns.  While in principal an empty schema is valid, in
      # practice this is a sign of a user error, and this is a convenient
      # place to catch that error.
      if (not feature_spec_or_typespecs and
          not self._ShouldDecodeAsRawExample(raw_examples_data_format)):
        raise ValueError(messages.SCHEMA_EMPTY)

    # Inspecting the preprocessing_fn even if we know we need a full pass in
    # order to fail faster if it fails.
    analyze_input_columns = tft.get_analyze_input_columns(
        preprocessing_fn, feature_spec_or_typespecs)

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
                             feature_spec_or_typespecs, transform_output_path)
        # TODO(b/122478841): Writes status to status file.
        return

    stats_use_tfdv = value_utils.GetSoleValue(
        inputs, labels.TFT_STATISTICS_USE_TFDV_LABEL)
    materialization_format = (
        transform_paths_file_formats[-1] if materialize_output_paths else None)
    self._RunBeamImpl(use_tfxio, analyze_data_list, transform_data_list,
                      preprocessing_fn, input_dataset_metadata,
                      transform_output_path, raw_examples_data_format,
                      temp_path, input_cache_dir, output_cache_dir,
                      compute_statistics, stats_use_tfdv,
                      per_set_stats_output_paths,
                      materialization_format)
  # TODO(b/122478841): Writes status to status file.

  def _RunBeamImpl(self,
                   use_tfxio: bool,
                   analyze_data_list: List[_Dataset],
                   transform_data_list: List[_Dataset],
                   preprocessing_fn: Any,
                   input_dataset_metadata: dataset_metadata.DatasetMetadata,
                   transform_output_path: Text,
                   raw_examples_data_format: Text,
                   temp_path: Text,
                   input_cache_dir: Optional[Text],
                   output_cache_dir: Optional[Text],
                   compute_statistics: bool,
                   stats_use_tfdv: bool,
                   per_set_stats_output_paths: Sequence[Text],
                   materialization_format: Optional[Text]) -> _Status:
    """Perform data preprocessing with TFT.

    Args:
      use_tfxio: if True, use the TFXIO-based TFT APIs.
      analyze_data_list: List of datasets for analysis.
      transform_data_list: List of datasets for transform.
      preprocessing_fn: The tf.Transform preprocessing_fn.
      input_dataset_metadata: A DatasetMetadata object for the input data.
      transform_output_path: An absolute path to write the output to.
      raw_examples_data_format: A string describing the raw data format.
      temp_path: A path to a temporary dir.
      input_cache_dir: A dir containing the input analysis cache. May be None.
      output_cache_dir: A dir to write the analysis cache to. May be None.
      compute_statistics: A bool indicating whether or not compute statistics.
      stats_use_tfdv: Always True.
      per_set_stats_output_paths: Paths to per-set statistics output. If empty,
        per-set statistics is not produced.
      materialization_format: A string describing the format of the materialized
        data or None if materialization is not enabled.

    Returns:
      Status of the execution.
    """
    if use_tfxio:
      assert stats_use_tfdv
      # TODO(zhuo): add support for sequence example on par with the non-TFXIO
      # path. Currently what's missing is to compute pre-transform stats as if
      # they are tf.Examples
      assert not self._IsDataFormatSequenceExample(raw_examples_data_format)
      self._AssertSameTFXIOSchema(analyze_data_list)
      feature_spec_or_typespec = (
          analyze_data_list[0].tfxio.TensorAdapter().OriginalTypeSpecs())
    else:
      feature_spec_or_typespec = schema_utils.schema_as_feature_spec(
          _GetSchemaProto(input_dataset_metadata)).feature_spec

    analyze_input_columns = tft.get_analyze_input_columns(
        preprocessing_fn, feature_spec_or_typespec)
    transform_input_columns = tft.get_transform_input_columns(
        preprocessing_fn, feature_spec_or_typespec)
    # Use the same dataset (same columns) for AnalyzeDataset and computing
    # pre-transform stats so that the data will only be read once for these
    # two operations.
    if compute_statistics:
      analyze_input_columns = list(
          set(list(analyze_input_columns) + list(transform_input_columns)))

    if use_tfxio:
      for d in analyze_data_list:
        d.tfxio = d.tfxio.Project(analyze_input_columns)
      for d in transform_data_list:
        d.tfxio = d.tfxio.Project(transform_input_columns)
      analyze_data_tensor_adapter_config = (
          analyze_data_list[0].tfxio.TensorAdapterConfig())
    else:
      if input_dataset_metadata.schema is _RAW_EXAMPLE_SCHEMA:
        analyze_input_dataset_metadata = input_dataset_metadata
        transform_input_dataset_metadata = input_dataset_metadata
      else:
        analyze_input_dataset_metadata = dataset_metadata.DatasetMetadata(
            schema_utils.schema_from_feature_spec({
                feature: feature_spec_or_typespec[feature]
                for feature in analyze_input_columns
            }))
        transform_input_dataset_metadata = dataset_metadata.DatasetMetadata(
            schema_utils.schema_from_feature_spec({
                feature: feature_spec_or_typespec[feature]
                for feature in transform_input_columns
            }))

    desired_batch_size = self._GetDesiredBatchSize(raw_examples_data_format)

    # Build a kwargs dict instead of passing the keyword arguments directly
    # to tft_beam.Context() because older TFT version doesn't not have the
    # argument `use_tfxio`.
    beam_context_kwargs = {
        'temp_dir': temp_path,
        'desired_batch_size': desired_batch_size,
        'passthrough_keys': {_TRANSFORM_INTERNAL_FEATURE_FOR_KEY},
        'use_deep_copy_optimization': True
    }
    if use_tfxio:
      # TODO(zhuo): add support for formats that have passthrough_keys (only KV
      # formats do).
      beam_context_kwargs['passthrough_keys'] = None
      beam_context_kwargs['use_tfxio'] = True

    with self._CreatePipeline(transform_output_path) as pipeline:
      with tft_beam.Context(**beam_context_kwargs):
        # pylint: disable=expression-not-assigned
        # pylint: disable=no-value-for-parameter
        _ = (
            pipeline
            | 'IncrementColumnUsageCounter'
            >> self._IncrementColumnUsageCounter(
                len(feature_spec_or_typespec), len(analyze_input_columns),
                len(transform_input_columns)))

        (new_analyze_data_dict, input_cache, flat_data_required) = (
            pipeline
            | 'OptimizeRun' >> self._OptimizeRun(
                input_cache_dir, output_cache_dir, analyze_data_list,
                feature_spec_or_typespec, preprocessing_fn,
                self._GetCacheSource()))

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
          if use_tfxio:
            dataset.standardized = (
                pipeline
                | 'TFXIOReadAndDecode[{}]'.format(infix) >>
                dataset.tfxio.BeamSource(desired_batch_size))
          else:
            dataset.serialized = (
                pipeline
                | 'ReadDataset[{}]'.format(infix) >> self._ReadExamples(
                    dataset, analyze_input_dataset_metadata))

        if not use_tfxio:
          analyze_decode_fn = (
              self._GetDecodeFunction(raw_examples_data_format,
                                      analyze_input_dataset_metadata.schema))

        input_analysis_data = {}
        for key, dataset in new_analyze_data_dict.items():
          if dataset is None:
            input_analysis_data[key] = None
          else:
            infix = 'AnalysisIndex{}'.format(dataset.index)
            if use_tfxio:
              input_analysis_data[key] = dataset.standardized
            else:
              dataset.decoded = (
                  dataset.serialized
                  | 'Decode[{}]'.format(infix) >>
                  self._DecodeInputs(analyze_decode_fn))
              input_analysis_data[key] = dataset.decoded

        flat_input_analysis_data = None
        if flat_data_required:
          flat_input_analysis_data = (
              [
                  dataset for dataset in input_analysis_data.values()
                  if dataset is not None
              ]
              | 'FlattenAnalysisDatasetsBecauseItIsRequired' >>
              beam.Flatten(pipeline=pipeline))

        analyze_input_metadata = (
            analyze_data_tensor_adapter_config
            if use_tfxio else input_dataset_metadata)
        transform_fn, cache_output = (
            (flat_input_analysis_data, input_analysis_data, input_cache,
             analyze_input_metadata)
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
        (transform_fn
         | 'WriteTransformFn'
         >> tft_beam.WriteTransformFn(transform_output_path))

        if output_cache_dir is not None and cache_output is not None:
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

            schema_proto = _GetSchemaProto(
                input_dataset_metadata
                if use_tfxio else analyze_input_dataset_metadata)

            if stats_use_tfdv:
              if not use_tfxio:
                for dataset in analyze_data_list:
                  infix = 'AnalysisIndex{}'.format(dataset.index)
                  dataset.standardized = (
                      dataset.serialized
                      | 'FromSerializedToArrowTables[{}]'.format(infix)
                      >> self._FromSerializedToArrowTables(schema_proto))

            pre_transform_stats_options = (
                transform_stats_options.get_pre_transform_stats_options())
            ([
                dataset.standardized if stats_use_tfdv else dataset.serialized
                for dataset in analyze_data_list
            ]
             | 'FlattenAnalysisDatasets' >> beam.Flatten(pipeline=pipeline)
             | 'GenerateStats[FlattenedAnalysisDatasets]' >>
             self._GenerateStats(
                 pre_transform_feature_stats_path,
                 schema_proto,
                 stats_options=pre_transform_stats_options,
                 use_tfdv=stats_use_tfdv,
                 examples_are_serialized=True,
                 input_from_tfxio=use_tfxio))

          # transform_data_list is a superset of analyze_data_list, we pay the
          # cost to read the same dataset (analyze_data_list) again here to
          # prevent certain beam runner from doing large temp materialization.
          for dataset in transform_data_list:
            infix = 'TransformIndex{}'.format(dataset.index)
            if use_tfxio:
              dataset.standardized = (
                  pipeline | 'TFXIOReadAndDecode[{}]'.format(infix) >>
                  dataset.tfxio.BeamSource(desired_batch_size))
            else:
              transform_decode_fn = (
                  self._GetDecodeFunction(
                      raw_examples_data_format,
                      transform_input_dataset_metadata.schema))
              dataset.serialized = (
                  pipeline
                  | 'ReadDataset[{}]'.format(infix) >> self._ReadExamples(
                      dataset, transform_input_dataset_metadata))
              dataset.decoded = (
                  dataset.serialized
                  | 'Decode[{}]'.format(infix)
                  >> self._DecodeInputs(transform_decode_fn))
            tft_transform_input_metadata = (
                dataset.tfxio.TensorAdapterConfig() if use_tfxio else
                transform_input_dataset_metadata)
            data = dataset.standardized if use_tfxio else dataset.decoded
            (dataset.transformed, metadata) = (
                ((data, tft_transform_input_metadata), transform_fn)
                | 'Transform[{}]'.format(infix) >> tft_beam.TransformDataset())

            if materialization_format is not None or not stats_use_tfdv:
              dataset.transformed_and_encoded = (
                  dataset.transformed
                  | 'Encode[{}]'.format(infix)
                  >> beam.ParDo(self._EncodeAsExamples(serialized=False),
                                _GetSchemaProto(metadata)))

          if compute_statistics:
            # Aggregated feature stats after transformation.
            _, metadata = transform_fn

            # TODO(b/70392441): Retain tf.Metadata (e.g., IntDomain) in
            # schema. Currently input dataset schema only contains dtypes,
            # and other metadata is dropped due to roundtrip to tensors.
            transformed_schema_proto = _GetSchemaProto(metadata)

            if stats_use_tfdv:
              for dataset in transform_data_list:
                infix = 'TransformIndex{}'.format(dataset.index)
                dataset.transformed_and_standardized = (
                    dataset.transformed
                    | 'FromDictsToArrowTables[{}]'.format(infix)
                    >> self._FromDictsToArrowTables(transformed_schema_proto))

            post_transform_feature_stats_path = os.path.join(
                transform_output_path,
                tft.TFTransformOutput.POST_TRANSFORM_FEATURE_STATS_PATH)

            post_transform_stats_options = (
                transform_stats_options.get_post_transform_stats_options())
            ([(dataset.transformed_and_standardized
               if stats_use_tfdv else dataset.transformed_and_encoded)
              for dataset in transform_data_list]
             | 'FlattenTransformedDatasets' >> beam.Flatten()
             | 'GenerateStats[FlattenedTransformedDatasets]' >>
             self._GenerateStats(
                 post_transform_feature_stats_path,
                 transformed_schema_proto,
                 stats_options=post_transform_stats_options,
                 use_tfdv=stats_use_tfdv))

            if per_set_stats_output_paths:
              # TODO(b/130885503): Remove duplicate stats gen compute that is
              # done both on a flattened view of the data, and on each span
              # below.
              for dataset in transform_data_list:
                infix = 'TransformIndex{}'.format(dataset.index)
                if stats_use_tfdv:
                  data = dataset.transformed_and_standardized
                else:
                  data = dataset.transformed_and_encoded
                data | 'GenerateStats[{}]'.format(infix) >> self._GenerateStats(
                    dataset.stats_output_path,
                    transformed_schema_proto,
                    stats_options=post_transform_stats_options,
                    use_tfdv=stats_use_tfdv)

          if materialization_format is not None:
            for dataset in transform_data_list:
              infix = 'TransformIndex{}'.format(dataset.index)
              (dataset.transformed_and_encoded
               | 'Materialize[{}]'.format(infix) >> self._WriteExamples(
                   materialization_format,
                   dataset.materialize_output_path))

    return _Status.OK()

  def _RunInPlaceImpl(
      self, preprocessing_fn: Any,
      metadata: dataset_metadata.DatasetMetadata,
      feature_spec_or_typespecs: Dict[Text, Any],
      transform_output_path: Text) -> _Status:
    """Runs a transformation iteration in-place without looking at the data.

    Args:
      preprocessing_fn: The tf.Transform preprocessing_fn.
      metadata: A DatasetMetadata object for the input data.
      feature_spec_or_typespecs: a Dict[Text, Union[FeatureSpec, tf.TypeSpec]]
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

        input_signature = _create_batched_placeholders(
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
      data_format: Text,
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
        _Dataset(p, f, data_format, s, m)
        for p, f, s, m in zip(file_patterns, file_formats, stats_output_paths,
                              materialize_output_paths)
    ]
    result = sorted(datasets, key=lambda dataset: dataset.dataset_key)
    for index, dataset in enumerate(result):
      dataset.index = index
    return result

  def _ShouldDecodeAsRawExample(self, data_format: Union[Text, int]) -> bool:
    """Returns true if data format should be decoded as raw example.

    Args:
      data_format: name of data format.

    Returns:
      True if data format should be decoded as raw example.
    """
    return (self._IsDataFormatSequenceExample(data_format) or
            self._IsDataFormatProto(data_format))

  @staticmethod
  def _IsDataFormatSequenceExample(data_format: Union[Text, int]) -> bool:
    """Returns true if data format is sequence example.

    Args:
      data_format: name of data format.

    Returns:
      True if data format is sequence example.
    """
    assert not isinstance(data_format, int), data_format
    return data_format == labels.FORMAT_TF_SEQUENCE_EXAMPLE

  @staticmethod
  def _IsDataFormatProto(data_format: Union[Text, int]) -> bool:
    """Returns true if data format is protocol buffer.

    Args:
      data_format: name of data format.

    Returns:
      True if data format is protocol buffer.
    """
    assert not isinstance(data_format, int), data_format
    return data_format == labels.FORMAT_PROTO

  def _GetDesiredBatchSize(
      self, data_format: Union[Text, int]) -> Optional[int]:
    """Returns batch size.

    Args:
      data_format: name of data format.

    Returns:
      Batch size or None.
    """
    if self._IsDataFormatSequenceExample(data_format):
      return 1
    return None

  def _GetDecodeFunction(self, data_format: Union[Text, int],
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
      return lambda x: {RAW_EXAMPLE_KEY: x}
    else:
      return tft.coders.ExampleProtoCoder(schema, serialized=True).decode

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
                   schema: schema_pb2.Schema) -> tfxio.TFXIO:
    """Creates a TFXIO instance for `dataset`."""
    if self._ShouldDecodeAsRawExample(dataset.data_format):
      return raw_tf_record.RawTfRecordTFXIO(dataset.file_pattern,
                                            RAW_EXAMPLE_KEY)
    else:
      return tf_example_record.TFExampleRecord(
          dataset.file_pattern,
          # TODO(b/114938612): Eventually remove this override.
          validate=False,
          schema=schema)

  def _AssertSameTFXIOSchema(self, datasets: Sequence[_Dataset]) -> None:
    if not datasets:
      return
    for dataset in datasets[1:]:
      assert (datasets[0].tfxio.ArrowSchema().equals(
          dataset.tfxio.ArrowSchema()))
