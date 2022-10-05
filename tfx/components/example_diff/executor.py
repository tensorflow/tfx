# Copyright 2022 Google LLC. All Rights Reserved.
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
"""TFX example_diff executor."""
import os
from typing import Any, Dict, List

from absl import logging
import apache_beam as beam
import tensorflow as tf
from tensorflow_data_validation.skew import feature_skew_detector
from tfx import types
from tfx.components.util import tfxio_utils
from tfx.dsl.components.base import base_beam_executor
from tfx.proto import example_diff_pb2
from tfx.types import artifact_utils
from tfx.types import standard_component_specs
from tfx.utils import io_utils
from tfx.utils import json_utils
from tfx_bsl.tfxio import record_based_tfxio

STATS_FILE_NAME = 'skew_stats'
MATCH_STATS_FILE_NAME = 'match_stats'
CONFUSION_FILE_NAME = 'confusion'
_SAMPLE_FILE_NAME = 'sample_pairs'

_TELEMETRY_DESCRIPTORS = ['ExampleDiff']


class _IncludedSplitPairs(object):
  """Checks includedness of split pairs."""

  def __init__(self, include_split_pairs: List[List[str]]):
    if include_split_pairs is None:
      self._include_split_pairs = None
    else:
      self._include_split_pairs = set(
          (test, base) for test, base in include_split_pairs)

  def included(self, test_split: str, base_split: str) -> bool:
    if self._include_split_pairs is None:
      return True
    return (test_split, base_split) in self._include_split_pairs

  def get_included_split_pairs(self):
    return self._include_split_pairs


def _parse_example(serialized: bytes):
  # TODO(b/227361696): Validate that data are examples.
  ex = tf.train.Example()
  ex.ParseFromString(serialized)
  return ex


def _get_confusion_configs(
    config: example_diff_pb2.ExampleDiffConfig
) -> List[feature_skew_detector.ConfusionConfig]:
  result = []
  for confusion in config.paired_example_skew.confusion_config:
    result.append(feature_skew_detector.ConfusionConfig(confusion.feature_name))
  return result


def _config_to_kwargs(config: example_diff_pb2.ExampleDiffConfig):
  """Convert ExampleDiffConfig to DetectFeatureSkewImpl kwargs."""
  kwargs = {}
  if not config.HasField('paired_example_skew'):
    raise ValueError('ExampleDiffConfig missing required paired_example_skew.')
  kwargs['identifier_features'] = list(
      config.paired_example_skew.identifier_features)
  kwargs['features_to_ignore'] = list(
      config.paired_example_skew.ignore_features)
  kwargs['sample_size'] = config.paired_example_skew.skew_sample_size
  kwargs['float_round_ndigits'] = config.paired_example_skew.float_round_ndigits
  kwargs[
      'allow_duplicate_identifiers'] = config.paired_example_skew.allow_duplicate_identifiers
  # TODO(b/227361696): Add better unit tests here. This will require generating
  # a new dataset for test purposes.
  kwargs['confusion_configs'] = _get_confusion_configs(config)
  return kwargs


class Executor(base_beam_executor.BaseBeamExecutor):
  """Computes example level diffs. See TFDV feature_skew_detector.py."""

  def Do(self, input_dict: Dict[str, List[types.Artifact]],
         output_dict: Dict[str, List[types.Artifact]],
         exec_properties: Dict[str, Any]) -> None:
    """Computes example diffs for each split pair.

    Args:
      input_dict: Input dict from input key to a list of Artifacts.
      output_dict: Output dict from output key to a list of Artifacts.
      exec_properties: A dict of execution properties.

    Raises:
      ValueError: If examples are in a non- record-based format.

    Returns:
      None
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    # Load and deserialize included split pairs from execution properties.
    included_split_pairs = _IncludedSplitPairs(
        json_utils.loads(
            exec_properties.get(
                standard_component_specs.INCLUDE_SPLIT_PAIRS_KEY, 'null')) or
        None)

    test_examples = artifact_utils.get_single_instance(
        input_dict[standard_component_specs.EXAMPLES_KEY])
    base_examples = artifact_utils.get_single_instance(
        input_dict[standard_component_specs.BASELINE_EXAMPLES_KEY])

    example_diff_artifact = artifact_utils.get_single_instance(
        output_dict[standard_component_specs.EXAMPLE_DIFF_RESULT_KEY])
    diff_config = exec_properties.get(
        standard_component_specs.EXAMPLE_DIFF_CONFIG_KEY)

    logging.info('Running examplediff with config %s', diff_config)

    test_tfxio_factory = tfxio_utils.get_tfxio_factory_from_artifact(
        examples=[test_examples], telemetry_descriptors=_TELEMETRY_DESCRIPTORS)
    base_tfxio_factory = tfxio_utils.get_tfxio_factory_from_artifact(
        examples=[base_examples], telemetry_descriptors=_TELEMETRY_DESCRIPTORS)

    # First set up the pairs we'll operate on.
    split_pairs = []
    for test_split in artifact_utils.decode_split_names(
        test_examples.split_names):
      for base_split in artifact_utils.decode_split_names(
          base_examples.split_names):
        if included_split_pairs.included(test_split, base_split):
          split_pairs.append((test_split, base_split))
    if not split_pairs:
      raise ValueError(
          'No split pairs from test and baseline examples: %s, %s' %
          (test_examples, base_examples))
    if included_split_pairs.get_included_split_pairs():
      missing_split_pairs = included_split_pairs.get_included_split_pairs(
      ) - set(split_pairs)
      if missing_split_pairs:
        raise ValueError(
            'Missing split pairs identified in include_split_pairs: %s' %
            ', '.join([
                '%s_%s' % (test, baseline)
                for test, baseline in missing_split_pairs
            ]))
    with self._make_beam_pipeline() as p:
      for test_split, base_split in split_pairs:
        test_uri = artifact_utils.get_split_uri([test_examples], test_split)
        base_uri = artifact_utils.get_split_uri([base_examples], base_split)
        test_tfxio = test_tfxio_factory(io_utils.all_files_pattern(test_uri))
        base_tfxio = base_tfxio_factory(io_utils.all_files_pattern(base_uri))
        if not isinstance(
            test_tfxio, record_based_tfxio.RecordBasedTFXIO) or not isinstance(
                base_tfxio, record_based_tfxio.RecordBasedTFXIO):
          # TODO(b/227361696): Support more general sources.
          raise ValueError('Only RecordBasedTFXIO supported, got %s, %s' %
                           (test_tfxio, base_tfxio))

        split_pair = '%s_%s' % (test_split, base_split)
        logging.info('Processing split pair %s', split_pair)
        # pylint: disable=cell-var-from-loop
        @beam.ptransform_fn
        def _iteration(p):
          base_examples = (
              p | 'TFXIORead[base]' >> test_tfxio.RawRecordBeamSource()
              | 'Parse[base]' >> beam.Map(_parse_example))
          test_examples = (
              p | 'TFXIORead[test]' >> base_tfxio.RawRecordBeamSource()
              | 'Parse[test]' >> beam.Map(_parse_example))
          results = ((base_examples, test_examples)
                     | feature_skew_detector.DetectFeatureSkewImpl(
                         **_config_to_kwargs(diff_config)))

          output_uri = os.path.join(example_diff_artifact.uri,
                                    'SplitPair-%s' % split_pair)
          _ = (
              results[feature_skew_detector.SKEW_RESULTS_KEY]
              | 'WriteStats' >> feature_skew_detector.skew_results_sink(
                  os.path.join(output_uri, STATS_FILE_NAME)))
          _ = (
              results[feature_skew_detector.SKEW_PAIRS_KEY]
              | 'WriteSample' >> feature_skew_detector.skew_pair_sink(
                  os.path.join(output_uri, _SAMPLE_FILE_NAME)))
          _ = (
              results[feature_skew_detector.MATCH_STATS_KEY]
              | 'WriteMatchStats' >> feature_skew_detector.match_stats_sink(
                  os.path.join(output_uri, MATCH_STATS_FILE_NAME)))
          if feature_skew_detector.CONFUSION_KEY in results:
            _ = (
                results[feature_skew_detector.CONFUSION_KEY]
                |
                'WriteConfusion' >> feature_skew_detector.confusion_count_sink(
                    os.path.join(output_uri, CONFUSION_FILE_NAME)))

          # pylint: enable=cell-var-from-loop

        _ = p | 'ProcessSplits[%s]' % split_pair >> _iteration()  # pylint: disable=no-value-for-parameter
