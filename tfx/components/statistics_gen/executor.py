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
"""Generic TFX statistics_gen executor."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import apache_beam as beam
import tensorflow as tf
from tensorflow_data_validation.api import stats_api
from tensorflow_data_validation.coders import tf_example_decoder
from tensorflow_data_validation.statistics import stats_options as options
from typing import Any, Dict, List, Text
from tensorflow_metadata.proto.v0 import statistics_pb2
from tfx.components.base import base_executor
from tfx.utils import io_utils
from tfx.utils import types

# Default file name for stats generated.
_DEFAULT_FILE_NAME = 'stats_tfrecord'


class Executor(base_executor.BaseExecutor):
  """Generic TFX statsgen executor."""

  def Do(self, input_dict,
         output_dict,
         exec_properties):
    """Computes stats for each split of input using tensorflow_data_validation.

    Args:
      input_dict: Input dict from input key to a list of Artifacts.
        - input_data: A list of 'ExamplesPath' type. This should contain both
          'train' and 'eval' split.
      output_dict: Output dict from output key to a list of Artifacts.
        - output: A list of 'ExampleStatisticsPath' type. This should contain
          both 'train' and 'eval' split.
      exec_properties: A dict of execution properties. Not used yet.

    Returns:
      None
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    split_to_instance = {x.split: x for x in input_dict['input_data']}
    with beam.Pipeline(argv=self._get_beam_pipeline_args()) as p:
      # TODO(b/126263006): Support more stats_options through config.
      stats_options = options.StatsOptions()
      for split, instance in split_to_instance.items():
        tf.logging.info('Generating statistics for split {}'.format(split))
        input_uri = io_utils.all_files_pattern(instance.uri)
        output_uri = types.get_split_uri(output_dict['output'], split)
        output_path = os.path.join(output_uri, _DEFAULT_FILE_NAME)
        _ = (
            p
            | 'ReadData.' + split >>
            beam.io.ReadFromTFRecord(file_pattern=input_uri)
            | 'DecodeData.' + split >> tf_example_decoder.DecodeTFExample()
            | 'GenerateStatistics.' + split >>
            stats_api.GenerateStatistics(stats_options)
            | 'WriteStatsOutput.' + split >> beam.io.WriteToTFRecord(
                output_path,
                shard_name_template='',
                coder=beam.coders.ProtoCoder(
                    statistics_pb2.DatasetFeatureStatisticsList)))
      tf.logging.info('Statistics written to {}.'.format(output_uri))
