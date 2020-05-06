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
from typing import Any, Dict, List, Text

import absl
import apache_beam as beam
from six import with_metaclass

from google.protobuf import json_format
from tfx import types
from tfx.components.base import base_executor
from tfx.components.example_gen import utils
from tfx.proto import example_gen_pb2
from tfx.types import artifact_utils

# Default file name for TFRecord output file prefix.
DEFAULT_FILE_NAME = 'data_tfrecord'
# Key for input in executor input_dict.
INPUT_KEY = 'input'

# Key for output examples in executor output_dict.
EXAMPLES_KEY = 'examples'


def _PartitionFn(record: bytes, num_partitions: int, buckets: List[int]) -> int:
  assert num_partitions == len(
      buckets), 'Partitions do not match bucket number.'
  bucket = int(hashlib.sha256(record).hexdigest(), 16) % buckets[-1]
  # For example, if buckets is [10,50,80], there will be 3 splits:
  #   bucket >=0 && < 10, returns 0
  #   bucket >=10 && < 50, returns 1
  #   bucket >=50 && < 80, returns 2
  return bisect.bisect(buckets, bucket)


@beam.ptransform_fn
@beam.typehints.with_input_types(bytes)
@beam.typehints.with_output_types(beam.pvalue.PDone)
def _WriteSplit(example_split: beam.pvalue.PCollection,
                output_split_path: Text) -> beam.pvalue.PDone:
  """Shuffles and writes output split."""
  return (example_split
          # TODO(jyzhao): make shuffle optional.
          | 'Shuffle' >> beam.transforms.Reshuffle()
          # TODO(jyzhao): multiple output format.
          | 'Write' >> beam.io.WriteToTFRecord(
              os.path.join(output_split_path, DEFAULT_FILE_NAME),
              file_name_suffix='.gz'))


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(bytes)
def _InputToSerializedExample(pipeline: beam.Pipeline,
                              input_to_example: beam.PTransform,
                              input_dict: Dict[Text, List[types.Artifact]],
                              exec_properties: Dict[Text, Any],
                              split_pattern: Text) -> beam.pvalue.PCollection:
  """Converts input to serialized TF examples."""
  return (pipeline
          | 'InputSourceToExample' >> input_to_example(
              input_dict, exec_properties, split_pattern)
          # Returns deterministic string as partition is based on it.
          | 'SerializeDeterministically' >>
          beam.Map(lambda x: x.SerializeToString(deterministic=True)))


class BaseExampleGenExecutor(
    with_metaclass(abc.ABCMeta, base_executor.BaseExecutor)):
  """Generic TFX example gen base executor.

  The base ExampleGen executor takes a configuration and converts external data
  sources to TensorFlow Examples (tf.Example).

  The common configuration (defined in
  https://github.com/tensorflow/tfx/blob/master/tfx/proto/example_gen.proto#L44.)
  describes the general properties of input data and shared instructions when
  producing output data.

  The conversion is done in `GenerateExamplesByBeam` as a Beam pipeline, which
  validates the configuration, reads the external data sources, converts the
  record in the input source to tf.Example if needed, and splits the examples if
  the output split config is given. Then the executor's `Do` writes the results
  in splits to the output path.

  For simple custom ExampleGens, the details of transforming input data
  record(s) to a tf.Example is expected to be given in
  `GetInputSourceToExamplePTransform`, which returns a Beam PTransform with the
  actual implementation. For complex use cases, such as joining multiple data
  sources and different interpretations of the configurations, the custom
  ExampleGen can override `GenerateExamplesByBeam`.
  """

  @abc.abstractmethod
  def GetInputSourceToExamplePTransform(self) -> beam.PTransform:
    """Returns PTransform for converting input source to TF examples.

    Note that each input split will be transformed by this function separately.
    For complex use case, consider override 'GenerateExamplesByBeam' instead.

    Here is an example PTransform:
      @beam.ptransform_fn
      @beam.typehints.with_input_types(beam.Pipeline)
      @beam.typehints.with_output_types(tf.train.Example)
      def ExamplePTransform(
          pipeline: beam.Pipeline,
          input_dict: Dict[Text, List[types.Artifact]],
          exec_properties: Dict[Text, Any],
          split_pattern: Text) -> beam.pvalue.PCollection
    """
    pass

  def GenerateExamplesByBeam(
      self, pipeline: beam.Pipeline, input_dict: Dict[Text,
                                                      List[types.Artifact]],
      exec_properties: Dict[Text, Any]) -> Dict[Text, beam.pvalue.PCollection]:
    """Converts input source to TF example splits based on configs.

    Custom ExampleGen executor should provide GetInputSourceToExamplePTransform
    for converting input split to TF Examples. Overriding this
    'GenerateExamplesByBeam' method instead if complex logic is need, e.g.,
    custom spliting logic.

    Args:
      pipeline: beam pipeline.
      input_dict: Input dict from input key to a list of Artifacts. Depends on
        detailed example gen implementation.
      exec_properties: A dict of execution properties. Depends on detailed
        example gen implementation.
        - input: JSON string of example_gen_pb2.Input instance, providing input
          configuration.
        - output: JSON string of example_gen_pb2.Output instance, providing
          output configuration.

    Returns:
      Dict of beam PCollection with split name as key, each PCollection is a
      single output split that contains serialized TF Examples.
    """
    # Get input split information.
    input_config = example_gen_pb2.Input()
    json_format.Parse(exec_properties['input_config'], input_config)
    # Get output split information.
    output_config = example_gen_pb2.Output()
    json_format.Parse(exec_properties['output_config'], output_config)
    # Get output split names.
    split_names = utils.generate_output_split_names(input_config, output_config)
    # Make beam_pipeline_args available in exec_properties since certain
    # example_gen executors need this information.
    # TODO(b/155441037): Revisit necessity of this when BigQueryExampleGen
    # does not branch on project or runner anymore.
    exec_properties['_beam_pipeline_args'] = self._beam_pipeline_args or []

    example_splits = []
    input_to_example = self.GetInputSourceToExamplePTransform()
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
          | 'InputToSerializedExample' >> _InputToSerializedExample(  # pylint: disable=no-value-for-parameter
              input_to_example, input_dict, exec_properties,
              input_config.splits[0].pattern)
          | 'SplitData' >> beam.Partition(_PartitionFn, len(buckets), buckets))
    else:
      # Use input splits.
      for split in input_config.splits:
        examples = (
            pipeline
            | 'InputToSerializedExample[{}]'.format(split.name) >>
            _InputToSerializedExample(  # pylint: disable=no-value-for-parameter
                input_to_example, input_dict, exec_properties, split.pattern))
        example_splits.append(examples)

    result = {}
    for index, example_split in enumerate(example_splits):
      result[split_names[index]] = example_split
    return result

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """Take input data source and generates TF Example splits.

    Args:
      input_dict: Input dict from input key to a list of Artifacts. Depends on
        detailed example gen implementation.
      output_dict: Output dict from output key to a list of Artifacts.
        - examples: splits of tf examples.
      exec_properties: A dict of execution properties. Depends on detailed
        example gen implementation.
        - input: JSON string of example_gen_pb2.Input instance, providing input
          configuration.
        - output: JSON string of example_gen_pb2.Output instance, providing
          output configuration.

    Returns:
      None
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    absl.logging.info('Generating examples.')
    with self._make_beam_pipeline() as pipeline:
      example_splits = self.GenerateExamplesByBeam(pipeline, input_dict,
                                                   exec_properties)

      # pylint: disable=expression-not-assigned, no-value-for-parameter
      for split_name, example_split in example_splits.items():
        (example_split
         | 'WriteSplit[{}]'.format(split_name) >> _WriteSplit(
             artifact_utils.get_split_uri(output_dict['examples'], split_name)))
      # pylint: enable=expression-not-assigned, no-value-for-parameter

    absl.logging.info('Examples generated.')
