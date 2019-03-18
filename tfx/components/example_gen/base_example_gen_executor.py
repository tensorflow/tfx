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
import hashlib
import os
import apache_beam as beam
from six import with_metaclass
import tensorflow as tf
from typing import Any, Dict, List, Text
from tfx.components.base import base_executor
from tfx.utils import types

# Default file name for TFRecord output file prefix.
DEFAULT_FILE_NAME = 'data_tfrecord'


def _partition_fn(record, num_partitions):  # pylint: disable=unused-argument
  # TODO(jyzhao): support custom split.
  # Splits data, train(partition=0) : eval(partition=1) = 2 : 1
  return 1 if int(hashlib.sha256(record).hexdigest(), 16) % 3 == 0 else 0


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

    Returns:
      None
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    training_tfrecord = types.get_split_uri(output_dict['examples'], 'train')
    eval_tfrecord = types.get_split_uri(output_dict['examples'], 'eval')

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
          | 'SplitData' >> beam.Partition(_partition_fn, 2))
      # TODO(jyzhao): make shuffle optional.
      # pylint: disable=expression-not-assigned
      (example_splits[0]
       | 'ShuffleTrainSplit' >> beam.transforms.Reshuffle()
       | 'OutputTrainSplit' >> beam.io.WriteToTFRecord(
           os.path.join(training_tfrecord, DEFAULT_FILE_NAME),
           file_name_suffix='.gz'))
      (example_splits[1]
       | 'ShuffleEvalSplit' >> beam.transforms.Reshuffle()
       | 'OutputEvalSplit' >> beam.io.WriteToTFRecord(
           os.path.join(eval_tfrecord, DEFAULT_FILE_NAME),
           file_name_suffix='.gz'))
      # pylint: enable=expression-not-assigned

    tf.logging.info('Examples generated.')
