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
"""TFX statistics_gen executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Any, Dict, List, Text

import absl
import apache_beam as beam
from tensorflow_data_validation.api import stats_api
from tensorflow_data_validation.coders import tf_example_decoder
from tensorflow_data_validation.statistics import stats_options as options

from tensorflow_metadata.proto.v0 import statistics_pb2
from tfx import types
from tfx.components.base import base_executor
from tfx.types import artifact_utils
from tfx.utils import io_utils


# Key for examples in executor input_dict.
EXAMPLES_KEY = 'examples'

# Key for output statistics in executor output_dict.
STATISTICS_KEY = 'statistics'

# Default file name for stats generated.
_DEFAULT_FILE_NAME = 'stats_tfrecord'


class Executor(base_executor.BaseExecutor):
  """Computes statistics over input training data for example validation.

  The StatisticsGen component generates features statistics and random samples
  over training data, which can be used for visualization and validation.
  StatisticsGen uses Beam and appropriate algorithms to scale to large datasets.

  To include StatisticsGen in a TFX pipeline, configure your pipeline similar to
  https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_pipeline_simple.py#L75.
  """

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """Computes stats for each split of input using tensorflow_data_validation.

    Args:
      input_dict: Input dict from input key to a list of Artifacts.
        - input_data: A list of type `standard_artifacts.Examples`. This should
          contain both 'train' and 'eval' split.
      output_dict: Output dict from output key to a list of Artifacts.
        - output: A list of type `standard_artifacts.ExampleStatistics`. This
          should contain both the 'train' and 'eval' splits.
      exec_properties: A dict of execution properties. Not used yet.

    Returns:
      None
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    split_uris = []
    for artifact in input_dict[EXAMPLES_KEY]:
      for split in artifact_utils.decode_split_names(artifact.split_names):
        uri = os.path.join(artifact.uri, split)
        split_uris.append((split, uri))
    with self._make_beam_pipeline() as p:
      # TODO(b/126263006): Support more stats_options through config.
      stats_options = options.StatsOptions()
      for split, uri in split_uris:
        absl.logging.info('Generating statistics for split {}'.format(split))
        input_uri = io_utils.all_files_pattern(uri)
        output_uri = artifact_utils.get_split_uri(output_dict[STATISTICS_KEY],
                                                  split)
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
        absl.logging.info('Statistics for split {} written to {}.'.format(
            split, output_uri))
