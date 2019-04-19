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
import apache_beam as beam
from six import with_metaclass
import tensorflow as tf
from typing import Any, Dict, List, Text
from tfx.components.base import base_executor
from tfx.proto import example_gen_pb2
from tfx.utils import types
from google.protobuf import json_format

# Default file name for TFRecord output file prefix.
DEFAULT_FILE_NAME = 'data_tfrecord'


def _partition_fn(
    record,
    num_partitions,  # pylint: disable=unused-argument
    buckets):
  bucket = int(hashlib.sha256(record).hexdigest(), 16) % buckets[-1]
  # For example, if buckets is [10,50,80], there will be 3 splits:
  #   bucket >=0 && < 10, returns 0
  #   bucket >=10 && < 50, returns 1
  #   bucket >=50 && < 80, returns 2
  return bisect.bisect(buckets, bucket)


class BaseExampleGenExecutor(
    with_metaclass(abc.ABCMeta, base_executor.BaseExecutor)):
  """Generic TFX example gen base executor."""

  # TODO(b/67107830): align with GenerateExamplesForVersion.
  @abc.abstractmethod
  def GetInputSourceToExamplePTransform(self):
    """Returns PTransform for converting input source to TF examples.

    Here is an example PTransform:
      @beam.ptransform_fn
      @beam.typehints.with_input_types(beam.Pipeline)
      @beam.typehints.with_output_types(tf.train.Example)
      def ExamplePTransform(
          pipeline: beam.Pipeline,
          input_dict: Dict[Text, List[types.TfxType]],
          exec_properties: Dict[Text, Any]) -> beam.pvalue.PCollection
    """
    pass

  def _check_split_config(self,
                          split_config):
    split_names = set()
    for split in split_config.splits:
      if not split.name or split.hash_buckets <= 0:
        raise RuntimeError('Split name and hash_buckets are required.')
      if split.name in split_names:
        raise RuntimeError('Duplicated split name {}.'.format(split.name))
      else:
        split_names.add(split.name)
    # TODO(jyzhao): use input splits if output splits are not specified.
    if not split_names:
      raise RuntimeError('ExampleGen output split is missing.')
    # TODO(jyzhao): support custom split for downstream components.
    if not {'train', 'eval'}.issubset(split_names):
      raise RuntimeError(
          'ExampleGen output splits must contain train and eval.')

  def Do(self, input_dict,
         output_dict,
         exec_properties):
    """Take input data source and generates train and eval tf examples.

    Args:
      input_dict: Input dict from input key to a list of Artifacts. Depends on
        detailed example gen implementation.
      output_dict: Output dict from output key to a list of Artifacts.
        - examples: train and eval split of tf examples.
      exec_properties: A dict of execution properties. Depends on detailed
        example gen implementation.
        - output: JSON string of example_gen_pb2.Output instance, providing
          output configuration.

    Returns:
      None

    Raises:
      RuntimeError: if output split config is not specified.
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    # Get output split information.
    output_config = example_gen_pb2.Output()
    json_format.Parse(exec_properties['output'], output_config)
    self._check_split_config(output_config.split_config)
    splits = output_config.split_config.splits
    # Calculate split buckets.
    buckets = []
    total_buckets = 0
    for split in splits:
      total_buckets += split.hash_buckets
      buckets.append(total_buckets)

    tf.logging.info('Generating examples.')
    with beam.Pipeline(argv=self._get_beam_pipeline_args()) as pipeline:
      input_to_example = self.GetInputSourceToExamplePTransform()
      example_splits = (
          pipeline
          | 'InputSourceToExample' >> input_to_example(input_dict,
                                                       exec_properties)
          # Returns deterministic string as partition is based on it.
          | 'SerializeDeterministically' >>
          beam.Map(lambda x: x.SerializeToString(deterministic=True))
          | 'SplitData' >> beam.Partition(_partition_fn, len(buckets), buckets))
      # TODO(jyzhao): make shuffle optional.
      # pylint: disable=expression-not-assigned
      for index, example_split in enumerate(example_splits):
        (example_split
         | 'ShuffleSplit' + splits[index].name >> beam.transforms.Reshuffle()
         | 'OutputSplit' + splits[index].name >> beam.io.WriteToTFRecord(
             os.path.join(
                 types.get_split_uri(output_dict['examples'],
                                     splits[index].name), DEFAULT_FILE_NAME),
             file_name_suffix='.gz'))
      # pylint: enable=expression-not-assigned

    tf.logging.info('Examples generated.')
