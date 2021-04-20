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
"""Generic TFX example gen base executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import bisect
import hashlib
import os
from typing import Any, Dict, List, Text, Union

from absl import logging
import apache_beam as beam
from six import with_metaclass
import tensorflow as tf
from tfx import types
from tfx.components.example_gen import utils
from tfx.components.util import examples_utils
from tfx.dsl.components.base import base_beam_executor
from tfx.proto import example_gen_pb2
from tfx.types import artifact_utils
from tfx.types import standard_component_specs
from tfx.utils import proto_utils
from tfx_bsl.telemetry import util

# Default file name for TFRecord output file prefix.
DEFAULT_FILE_NAME = 'data_tfrecord'

# Metrics namespace for ExampleGen.
_METRICS_NAMESPACE = util.MakeTfxNamespace(['ExampleGen'])


def _GeneratePartitionKey(record: Union[tf.train.Example,
                                        tf.train.SequenceExample, bytes],
                          split_config: example_gen_pb2.SplitConfig) -> bytes:
  """Generates key for partition."""

  if not split_config.HasField('partition_feature_name'):
    if isinstance(record, bytes):
      return record
    return record.SerializeToString(deterministic=True)

  if isinstance(record, tf.train.Example):
    features = record.features.feature  # pytype: disable=attribute-error
  elif isinstance(record, tf.train.SequenceExample):
    features = record.context.feature  # pytype: disable=attribute-error
  else:
    raise RuntimeError('Split by `partition_feature_name` is only supported '
                       'for FORMAT_TF_EXAMPLE and FORMAT_TF_SEQUENCE_EXAMPLE '
                       'payload format.')

  # Use a feature for partitioning the examples.
  feature_name = split_config.partition_feature_name
  if feature_name not in features:
    raise RuntimeError('Feature name `{}` does not exist.'.format(feature_name))
  feature = features[feature_name]
  if not feature.HasField('kind'):
    raise RuntimeError('Partition feature does not contain any value.')
  if (not feature.HasField('bytes_list') and
      not feature.HasField('int64_list')):
    raise RuntimeError('Only `bytes_list` and `int64_list` features are '
                       'supported for partition.')
  return feature.SerializeToString(deterministic=True)


def _PartitionFn(
    record: Union[tf.train.Example, tf.train.SequenceExample, bytes],
    num_partitions: int,
    buckets: List[int],
    split_config: example_gen_pb2.SplitConfig,
) -> int:
  """Partition function for the ExampleGen's output splits."""
  assert num_partitions == len(
      buckets), 'Partitions do not match bucket number.'
  partition_str = _GeneratePartitionKey(record, split_config)
  bucket = int(hashlib.sha256(partition_str).hexdigest(), 16) % buckets[-1]
  # For example, if buckets is [10,50,80], there will be 3 splits:
  #   bucket >=0 && < 10, returns 0
  #   bucket >=10 && < 50, returns 1
  #   bucket >=50 && < 80, returns 2
  return bisect.bisect(buckets, bucket)


@beam.ptransform_fn
@beam.typehints.with_input_types(Union[tf.train.Example,
                                       tf.train.SequenceExample, bytes])
@beam.typehints.with_output_types(beam.pvalue.PDone)
def _WriteSplit(
    example_split: beam.pvalue.PCollection,
    output_split_path: Text,
) -> beam.pvalue.PDone:
  """Shuffles and writes output split as serialized records in TFRecord."""

  class _MaybeSerialize(beam.DoFn):
    """Serializes the proto if needed."""

    def __init__(self):
      self._num_instances = beam.metrics.Metrics.counter(
          _METRICS_NAMESPACE, 'num_instances')

    def process(self, e):
      self._num_instances.inc(1)
      if isinstance(e, (tf.train.Example, tf.train.SequenceExample)):
        yield e.SerializeToString()
      else:
        yield e

  return (example_split
          # TODO(jyzhao): make shuffle optional.
          | 'MaybeSerialize' >> beam.ParDo(_MaybeSerialize())
          | 'Shuffle' >> beam.transforms.Reshuffle()
          # TODO(jyzhao): multiple output format.
          | 'Write' >> beam.io.WriteToTFRecord(
              os.path.join(output_split_path, DEFAULT_FILE_NAME),
              file_name_suffix='.gz'))


class BaseExampleGenExecutor(
    with_metaclass(abc.ABCMeta, base_beam_executor.BaseBeamExecutor)):
  """Generic TFX example gen base executor.

  The base ExampleGen executor takes a configuration and converts external data
  sources to TensorFlow Examples (tf.train.Example, tf.train.SequenceExample),
  or any other protocol buffer as subclass defines.

  The common configuration (defined in
  https://github.com/tensorflow/tfx/blob/master/tfx/proto/example_gen.proto#L44.)
  describes the general properties of input data and shared instructions when
  producing output data.

  The conversion is done in `GenerateExamplesByBeam` as a Beam pipeline, which
  validates the configuration, reads the external data sources, converts the
  record in the input source to any supported output payload formats
  (e.g., tf.Example or tf.SequenceExample) if needed, and splits the examples
  if the output split config is given. Then the executor's `Do` writes the
  results in splits to the output path.

  For simple custom ExampleGens, the details of transforming input data
  record(s) to a specific output payload format (e.g., tf.Example or
  tf.SequenceExample) is expected to be given in
  `GetInputSourceToExamplePTransform`, which returns a Beam PTransform with the
  actual implementation. For complex use cases, such as joining multiple data
  sources and different interpretations of the configurations, the custom
  ExampleGen can override `GenerateExamplesByBeam`.
  """

  @abc.abstractmethod
  def GetInputSourceToExamplePTransform(self) -> beam.PTransform:
    """Returns PTransform for converting input source to records.

    The record is by default assumed to be tf.train.Example protos, subclassses
    can serialize any protocol buffer into bytes as output PCollection,
    so long as the downstream component can consume it.

    Note that each input split will be transformed by this function separately.
    For complex use case, consider override 'GenerateExamplesByBeam' instead.

    Here is an example PTransform:
      @beam.ptransform_fn
      @beam.typehints.with_input_types(beam.Pipeline)
      @beam.typehints.with_output_types(Union[tf.train.Example,
                                              tf.train.SequenceExample,
                                              bytes])
      def ExamplePTransform(
          pipeline: beam.Pipeline,
          exec_properties: Dict[Text, Any],
          split_pattern: Text) -> beam.pvalue.PCollection
    """
    pass

  def GenerateExamplesByBeam(
      self,
      pipeline: beam.Pipeline,
      exec_properties: Dict[Text, Any],
  ) -> Dict[Text, beam.pvalue.PCollection]:
    """Converts input source to serialized record splits based on configs.

    Custom ExampleGen executor should provide GetInputSourceToExamplePTransform
    for converting input split to serialized records. Overriding this
    'GenerateExamplesByBeam' method instead if complex logic is need, e.g.,
    custom spliting logic.

    Args:
      pipeline: Beam pipeline.
      exec_properties: A dict of execution properties. Depends on detailed
        example gen implementation.
        - input_base: an external directory containing the data files.
        - input_config: JSON string of example_gen_pb2.Input instance, providing
          input configuration.
        - output_config: JSON string of example_gen_pb2.Output instance,
          providing output configuration.
        - output_data_format: Payload format of generated data in output
          artifact, one of example_gen_pb2.PayloadFormat enum.

    Returns:
      Dict of beam PCollection with split name as key, each PCollection is a
      single output split that contains serialized records.
    """
    # Get input split information.
    input_config = example_gen_pb2.Input()
    proto_utils.json_to_proto(
        exec_properties[standard_component_specs.INPUT_CONFIG_KEY],
        input_config)
    # Get output split information.
    output_config = example_gen_pb2.Output()
    proto_utils.json_to_proto(
        exec_properties[standard_component_specs.OUTPUT_CONFIG_KEY],
        output_config)
    # Get output split names.
    split_names = utils.generate_output_split_names(input_config, output_config)
    # Make beam_pipeline_args available in exec_properties since certain
    # example_gen executors need this information.
    exec_properties['_beam_pipeline_args'] = self._beam_pipeline_args or []

    example_splits = []
    input_to_record = self.GetInputSourceToExamplePTransform()
    if output_config.split_config.splits:
      # Use output splits, input must have only one split.
      assert len(
          input_config.splits
      ) == 1, 'input must have only one split when output split is specified.'
      # Calculate split buckets.
      buckets = []
      total_buckets = 0
      for split in output_config.split_config.splits:
        total_buckets += split.hash_buckets
        buckets.append(total_buckets)
      example_splits = (
          pipeline
          | 'InputToRecord' >>
          # pylint: disable=no-value-for-parameter
          input_to_record(exec_properties, input_config.splits[0].pattern)
          | 'SplitData' >> beam.Partition(_PartitionFn, len(buckets), buckets,
                                          output_config.split_config))
    else:
      # Use input splits.
      for split in input_config.splits:
        examples = (
            pipeline
            | 'InputToRecord[{}]'.format(split.name) >>
            # pylint: disable=no-value-for-parameter
            input_to_record(exec_properties, split.pattern))
        example_splits.append(examples)

    result = {}
    for index, example_split in enumerate(example_splits):
      result[split_names[index]] = example_split
    return result

  def Do(
      self,
      input_dict: Dict[Text, List[types.Artifact]],
      output_dict: Dict[Text, List[types.Artifact]],
      exec_properties: Dict[Text, Any],
  ) -> None:
    """Take input data source and generates serialized data splits.

    The output is intended to be serialized tf.train.Examples or
    tf.train.SequenceExamples protocol buffer in gzipped TFRecord format,
    but subclasses can choose to override to write to any serialized records
    payload into gzipped TFRecord as specified, so long as downstream
    component can consume it. The format of payload is added to
    `payload_format` custom property of the output Example artifact.

    Args:
      input_dict: Input dict from input key to a list of Artifacts. Depends on
        detailed example gen implementation.
      output_dict: Output dict from output key to a list of Artifacts.
        - examples: splits of serialized records.
      exec_properties: A dict of execution properties. Depends on detailed
        example gen implementation.
        - input_base: an external directory containing the data files.
        - input_config: JSON string of example_gen_pb2.Input instance,
          providing input configuration.
        - output_config: JSON string of example_gen_pb2.Output instance,
          providing output configuration.
        - output_data_format: Payload format of generated data in output
          artifact, one of example_gen_pb2.PayloadFormat enum.

    Returns:
      None
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    input_config = example_gen_pb2.Input()
    proto_utils.json_to_proto(
        exec_properties[standard_component_specs.INPUT_CONFIG_KEY],
        input_config)
    output_config = example_gen_pb2.Output()
    proto_utils.json_to_proto(
        exec_properties[standard_component_specs.OUTPUT_CONFIG_KEY],
        output_config)

    examples_artifact = artifact_utils.get_single_instance(
        output_dict[standard_component_specs.EXAMPLES_KEY])
    examples_artifact.split_names = artifact_utils.encode_split_names(
        utils.generate_output_split_names(input_config, output_config))

    logging.info('Generating examples.')
    with self._make_beam_pipeline() as pipeline:
      example_splits = self.GenerateExamplesByBeam(pipeline, exec_properties)

      # pylint: disable=expression-not-assigned, no-value-for-parameter
      for split_name, example_split in example_splits.items():
        (example_split
         | 'WriteSplit[{}]'.format(split_name) >> _WriteSplit(
             artifact_utils.get_split_uri(
                 output_dict[standard_component_specs.EXAMPLES_KEY],
                 split_name)))
      # pylint: enable=expression-not-assigned, no-value-for-parameter

    output_payload_format = exec_properties.get(
        standard_component_specs.OUTPUT_DATA_FORMAT_KEY)
    if output_payload_format:
      for output_examples_artifact in output_dict[
          standard_component_specs.EXAMPLES_KEY]:
        examples_utils.set_payload_format(output_examples_artifact,
                                          output_payload_format)
    logging.info('Examples generated.')
