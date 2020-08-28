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
from typing import Any, Dict, List, Optional, Text, Tuple, Type, Union

from absl import logging
import apache_beam as beam
from tensorflow_data_validation.api import stats_api
from tensorflow_data_validation.statistics import stats_options as options
from tfx import types
from tfx.components.base import base_executor
from tfx.components.util import tfxio_utils
from tfx.types import artifact_utils
from tfx.utils import io_utils
from tfx.utils import json_utils
from tfx_bsl.tfxio.raw_tf_record import RawBeamRecordTFXIO
from tfx_bsl.tfxio.tfxio import TFXIO

from tensorflow_metadata.proto.v0 import statistics_pb2

# TODO(b/162532479): switch to support List[Text] exclusively, once tfx-bsl
# post-0.22 is released.
OneOrMorePatterns = Union[Text, List[Text]]

# Key for examples in executor input_dict.
EXAMPLES_KEY = 'examples'
# Key for statistics in executor input_dict.
SCHEMA_KEY = 'schema'

# Key for stats options json in executor exec_properties dict.
STATS_OPTIONS_JSON_KEY = 'stats_options_json'
# Key for exclude splits in executor exec_properties dict.
EXCLUDE_SPLITS_KEY = 'exclude_splits'

# Key for statistics in executor output_dict.
STATISTICS_KEY = 'statistics'

# Default file name for stats generated.
_DEFAULT_FILE_NAME = 'stats_tfrecord'

_TELEMETRY_DESCRIPTORS = ['StatisticsGen']


class Executor(base_executor.FuseableBeamExecutor):
  """Computes statistics over input training data for example validation.

  The StatisticsGen component generates features statistics and random samples
  over training data, which can be used for visualization and validation.
  StatisticsGen uses Beam and appropriate algorithms to scale to large datasets.

  To include StatisticsGen in a TFX pipeline, configure your pipeline similar to
  https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_pipeline_simple.py#L75.
  """

  def __init__(self,
               context: Optional[base_executor.BaseExecutor.Context] = None):
    self._split_and_tfxio = []
    super(Executor, self).__init__(context=context)

  def _get_split_and_tfxio(
      self, input_dict: Dict[Text, List[types.Artifact]],
      output_dict: Dict[Text, List[types.Artifact]],
      exec_properties: Dict[Text, Union[int, float, Text]]
  ) -> List[Tuple[Text, TFXIO]]:
    if self._split_and_tfxio:
      return self._split_and_tfxio

    # Load and deserialize exclude splits from execution properties.
    exclude_splits = json_utils.loads(
        exec_properties.get(EXCLUDE_SPLITS_KEY, 'null')) or []
    if not isinstance(exclude_splits, list):
      raise ValueError('exclude_splits in execution properties needs to be a '
                       'list. Got %s instead.' % type(exclude_splits))
    # Setup output splits.
    examples = artifact_utils.get_single_instance(input_dict[EXAMPLES_KEY])
    examples_split_names = artifact_utils.decode_split_names(
        examples.split_names)
    split_names = [
        split for split in examples_split_names if split not in exclude_splits
    ]
    statistics_artifact = artifact_utils.get_single_instance(
        output_dict[STATISTICS_KEY])
    statistics_artifact.split_names = artifact_utils.encode_split_names(
        split_names)

    self._split_and_tfxio = []
    tfxio_factory = tfxio_utils.get_tfxio_factory_from_artifact(
        examples=[examples], telemetry_descriptors=_TELEMETRY_DESCRIPTORS)
    for split in artifact_utils.decode_split_names(examples.split_names):
      if split in exclude_splits:
        continue

      uri = os.path.join(examples.uri, split)
      self._split_and_tfxio.append(
          (split, tfxio_factory(io_utils.all_files_pattern(uri))))

    return self._split_and_tfxio

  def beam_io_signature(
      self,
      input_dict: Dict[Text, List[types.Artifact]],
      output_dict: Dict[Text, List[types.Artifact]],
      exec_properties: Dict[Text, Union[int, float, Text]]
  ) -> Tuple[Dict[Tuple[Text, Text], Type], Dict[Tuple[Text, Text], Type]]:  # pylint: disable=g-bare-generic
    input_signature = {}
    output_signature = {}
    split_and_tfxio = self._get_split_and_tfxio(input_dict, output_dict,
                                                exec_properties)
    for split, _ in split_and_tfxio:
      input_signature[(EXAMPLES_KEY, split)] = bytes
      output_signature[(STATISTICS_KEY,
                        split)] = statistics_pb2.DatasetFeatureStatisticsList

    return input_signature, output_signature

  def read_inputs(
      self, pipeline: beam.Pipeline,
      input_dict: Dict[Text, List[types.Artifact]],
      output_dict: Dict[Text, List[types.Artifact]],
      exec_properties: Dict[Text, Union[int, float, Text]]
  ) -> Dict[Tuple[Text, Text], beam.pvalue.PCollection]:
    beam_inputs = {}
    split_and_tfxio = self._get_split_and_tfxio(input_dict, output_dict,
                                                exec_properties)
    for split, tfxio in split_and_tfxio:
      logging.info('Generating statistics for split %s.', split)
      beam_inputs[(EXAMPLES_KEY, split)] = None
      beam_inputs[(EXAMPLES_KEY, 'pyarrow_records:%s' % split)] = (
          pipeline
          | 'TFXIORead[{}]'.format(split) >> tfxio.BeamSource())

    return beam_inputs

  def run_component(
      self, pipeline: beam.Pipeline,
      beam_inputs: Dict[Tuple[Text, Text], beam.pvalue.PCollection],
      input_dict: Dict[Text, List[types.Artifact]],
      output_dict: Dict[Text, List[types.Artifact]],
      exec_properties: Dict[Text, Union[int, float, Text]]
  ) -> Dict[Tuple[Text, Text], beam.pvalue.PCollection]:
    self._log_startup(input_dict, output_dict, exec_properties)

    stats_options = options.StatsOptions()
    stats_options_json = exec_properties.get(STATS_OPTIONS_JSON_KEY)
    if stats_options_json:
      # TODO(b/150802589): Move jsonable interface to tfx_bsl and use
      # json_utils
      stats_options = options.StatsOptions.from_json(stats_options_json)
    if input_dict.get(SCHEMA_KEY):
      if stats_options.schema:
        raise ValueError('A schema was provided as an input and the '
                         'stats_options exec_property also contains a schema '
                         'value. At most one of these may be set.')
      else:
        schema = io_utils.SchemaReader().read(
            io_utils.get_only_uri_in_dir(
                artifact_utils.get_single_uri(input_dict[SCHEMA_KEY])))
        stats_options.schema = schema

    beam_outputs = {}
    split_and_tfxio = self._get_split_and_tfxio(input_dict, output_dict,
                                                exec_properties)
    for split, _ in split_and_tfxio:
      # TODO(ccy): Currently, a workaround is needed here to allow for
      # PCollection fusion with an upstream ExampleGen instance. The upstream
      # PCollection is a PCollection of byte-encoded TFRecords, but our
      # input here provided by tfxio's BeamSource does not produce an
      # intermediate PCollection of bytes that can substitute directly. The
      # workaround right now is to special-case the input here, so that the
      # PCollection of PyArrow batches of TFRecords directly when this
      # component is not fused.
      passthrough_key = 'pyarrow_records:%s' % split
      if (EXAMPLES_KEY, passthrough_key) in beam_inputs:
        input_pcoll = beam_inputs[(EXAMPLES_KEY, passthrough_key)]
      else:
        input_pcoll = (
            beam_inputs[(EXAMPLES_KEY, split)]
            | 'TFXIOConvert_%s' % split >> (RawBeamRecordTFXIO(
                physical_format='tfrecord',
                raw_record_column_name='raw_example',
                telemetry_descriptors=_TELEMETRY_DESCRIPTORS)
                                            .RawRecordToRecordBatch()))
      beam_outputs[(STATISTICS_KEY, split)] = (
          input_pcoll
          | 'GenerateStatistics[%s]' % split >>
          stats_api.GenerateStatistics(stats_options))
    return beam_outputs

  def write_outputs(
      self,
      pipeline: beam.Pipeline,
      beam_outputs: Dict[Tuple[Text, Text], beam.pvalue.PCollection],
      input_dict: Dict[Text, List[types.Artifact]],
      output_dict: Dict[Text, List[types.Artifact]],
      exec_properties: Dict[Text, Union[int, float, Text]]) -> None:
    split_and_tfxio = self._get_split_and_tfxio(input_dict, output_dict,
                                                exec_properties)
    for split, _ in split_and_tfxio:
      output_uri = artifact_utils.get_split_uri(output_dict[STATISTICS_KEY],
                                                split)
      output_path = os.path.join(output_uri, _DEFAULT_FILE_NAME)
      _ = (
          beam_outputs[(STATISTICS_KEY, split)]
          | 'WriteStatsOutput[%s]' % split >>
          stats_api.WriteStatisticsToTFRecord(output_path))
      logging.info('Statistics for split %s written to %s.', split, output_uri)

  def Do(self,
         input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """Computes stats for each split of input using tensorflow_data_validation.

    Args:
      input_dict: Input dict from input key to a list of Artifacts.
        - input_data: A list of type `standard_artifacts.Examples`. This should
          contain both 'train' and 'eval' split.
        - schema: Optionally, a list of type `standard_artifacts.Schema`. When
          the stats_options exec_property also contains a schema, this input
          should not be provided.
      output_dict: Output dict from output key to a list of Artifacts.
        - output: A list of type `standard_artifacts.ExampleStatistics`. This
          should contain both the 'train' and 'eval' splits.
      exec_properties: A dict of execution properties.
        - stats_options_json: Optionally, a JSON representation of StatsOptions.
          When a schema is provided as an input, the StatsOptions value should
          not also contain a schema.
        - exclude_splits: JSON-serialized list of names of splits where
          statistics and sample should not be generated.

    Raises:
      ValueError when a schema is provided both as an input and as part of the
      StatsOptions exec_property.

    Returns:
      None
    """
    with self._make_beam_pipeline() as pipeline:
      beam_inputs = self.read_inputs(pipeline, input_dict, output_dict,
                                     exec_properties)
      beam_outputs = self.run_component(pipeline, beam_inputs, input_dict,
                                        output_dict, exec_properties)
      self.write_outputs(pipeline, beam_outputs, input_dict, output_dict,
                         exec_properties)
