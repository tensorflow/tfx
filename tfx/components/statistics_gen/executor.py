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
import os
from typing import Any, Dict, List

from absl import logging
import tensorflow_data_validation as tfdv
from tensorflow_data_validation.statistics import stats_options as options
from tfx import types
from tfx.components.statistics_gen import stats_artifact_utils
from tfx.components.util import tfxio_utils
from tfx.dsl.components.base import base_beam_executor
from tfx.types import artifact_utils
from tfx.types import standard_component_specs
from tfx.utils import io_utils
from tfx.utils import json_utils


# Default file name for stats generated.
DEFAULT_FILE_NAME = 'FeatureStats.pb'

_TELEMETRY_DESCRIPTORS = ['StatisticsGen']


class Executor(base_beam_executor.BaseBeamExecutor):
  """Computes statistics over input training data for example validation.

  The StatisticsGen component generates features statistics and random samples
  over training data, which can be used for visualization and validation.
  StatisticsGen uses Beam and appropriate algorithms to scale to large datasets.

  To include StatisticsGen in a TFX pipeline, configure your pipeline similar to
  https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_pipeline_simple.py#L75.
  """

  def Do(self, input_dict: Dict[str, List[types.Artifact]],
         output_dict: Dict[str, List[types.Artifact]],
         exec_properties: Dict[str, Any]) -> None:
    """Computes stats for each split of input using tensorflow_data_validation.

    Args:
      input_dict: Input dict from input key to a list of Artifacts.
        - examples: A list of type `standard_artifacts.Examples`. This should
          contain both 'train' and 'eval' split.
        - schema: Optionally, a list of type `standard_artifacts.Schema`. When
          the stats_options exec_property also contains a schema, this input
          should not be provided.
      output_dict: Output dict from output key to a list of Artifacts.
        - statistics: A list of type `standard_artifacts.ExampleStatistics`.
          This should contain both the 'train' and 'eval' splits.
      exec_properties: A dict of execution properties.
        - stats_options_json: Optionally, a JSON representation of StatsOptions.
          When a schema is provided as an input, the StatsOptions value should
          not also contain a schema.
        - exclude_splits: JSON-serialized list of names of splits where
          statistics and sample should not be generated.

    Raises:
      ValueError when a schema is provided both as an input and as part of the
      StatsOptions exec_property, or if execution properties specify
      write_sharded_output when unsupported.

    Returns:
      None
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    # Load and deserialize exclude splits from execution properties.
    exclude_splits = json_utils.loads(
        exec_properties.get(standard_component_specs.EXCLUDE_SPLITS_KEY,
                            'null')) or []
    if not isinstance(exclude_splits, list):
      raise ValueError('exclude_splits in execution properties needs to be a '
                       'list. Got %s instead.' % type(exclude_splits))
    # Setup output splits.
    examples = artifact_utils.get_single_instance(
        input_dict[standard_component_specs.EXAMPLES_KEY])
    examples_split_names = artifact_utils.decode_split_names(
        examples.split_names)
    split_names = [
        split for split in examples_split_names if split not in exclude_splits
    ]
    statistics_artifact = artifact_utils.get_single_instance(
        output_dict[standard_component_specs.STATISTICS_KEY])
    statistics_artifact.split_names = artifact_utils.encode_split_names(
        split_names)

    stats_options = options.StatsOptions()
    stats_options_json = exec_properties.get(
        standard_component_specs.STATS_OPTIONS_JSON_KEY)
    if stats_options_json:
      # TODO(b/150802589): Move jsonable interface to tfx_bsl and use
      # json_utils
      stats_options = options.StatsOptions.from_json(stats_options_json)

    write_sharded_output = exec_properties.get(
        standard_component_specs.SHARDED_STATS_OUTPUT_KEY, False)
    if write_sharded_output and not tfdv.default_sharded_output_supported():
      raise ValueError('Sharded output requested but not supported.')

    if input_dict.get(standard_component_specs.SCHEMA_KEY):
      if stats_options.schema:
        raise ValueError('A schema was provided as an input and the '
                         'stats_options exec_property also contains a schema '
                         'value. At most one of these may be set.')
      else:
        schema = io_utils.SchemaReader().read(
            io_utils.get_only_uri_in_dir(
                artifact_utils.get_single_uri(
                    input_dict[standard_component_specs.SCHEMA_KEY])))
        stats_options.schema = schema

    split_and_tfxio = []
    tfxio_factory = tfxio_utils.get_tfxio_factory_from_artifact(
        examples=[examples],
        telemetry_descriptors=_TELEMETRY_DESCRIPTORS)
    for split in artifact_utils.decode_split_names(examples.split_names):
      if split in exclude_splits:
        continue

      uri = artifact_utils.get_split_uri([examples], split)
      split_and_tfxio.append(
          (split, tfxio_factory(io_utils.all_files_pattern(uri))))
    if not split_and_tfxio:
      raise ValueError('No splits for examples artifact: %s' % examples)
    with self._make_beam_pipeline() as p:
      for split, tfxio in split_and_tfxio:
        logging.info('Generating statistics for split %s.', split)
        output_uri = artifact_utils.get_split_uri(
            output_dict[standard_component_specs.STATISTICS_KEY], split)
        binary_stats_output_path = os.path.join(output_uri, DEFAULT_FILE_NAME)

        data = p | 'TFXIORead[%s]' % split >> tfxio.BeamSource()
        if write_sharded_output:
          sharded_stats_output_prefix = os.path.join(
              output_uri, stats_artifact_utils.SHARDED_STATS_PREFIX +
              tfdv.default_sharded_output_suffix())
          write_transform = tfdv.WriteStatisticsToRecordsAndBinaryFile(
              binary_proto_path=binary_stats_output_path,
              records_path_prefix=sharded_stats_output_prefix)
        else:
          write_transform = tfdv.WriteStatisticsToBinaryFile(
              binary_stats_output_path)
        _ = (
            data
            | 'GenerateStatistics[%s]' % split >>
            tfdv.GenerateStatistics(stats_options)
            | 'WriteStatsOutput[%s]' % split >> write_transform)
        logging.info('Statistics for split %s written to %s.', split,
                     output_uri)
