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
"""TFX DistributionValidator executor."""

import os
from typing import Any, Dict, List

from absl import logging
import tensorflow_data_validation as tfdv
from tensorflow_data_validation.utils import path
from tensorflow_data_validation.utils import schema_util
from tfx import types
from tfx.components.statistics_gen import stats_artifact_utils
from tfx.dsl.components.base import base_executor
from tfx.proto import distribution_validator_pb2
from tfx.types import artifact_utils
from tfx.types import standard_component_specs
from tfx.utils import io_utils
from tfx.utils import json_utils

from tensorflow_metadata.proto.v0 import anomalies_pb2
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2

# Default file name for anomalies output.
DEFAULT_FILE_NAME = 'SchemaDiff.pb'

_COMPARISON_ANOMALY_TYPES = frozenset([
    anomalies_pb2.AnomalyInfo.Type.COMPARATOR_CONTROL_DATA_MISSING,
    anomalies_pb2.AnomalyInfo.Type.COMPARATOR_TREATMENT_DATA_MISSING,
    anomalies_pb2.AnomalyInfo.Type.COMPARATOR_L_INFTY_HIGH,
    anomalies_pb2.AnomalyInfo.Type.COMPARATOR_JENSEN_SHANNON_DIVERGENCE_HIGH,
    anomalies_pb2.AnomalyInfo.Type.COMPARATOR_LOW_NUM_EXAMPLES,
    anomalies_pb2.AnomalyInfo.Type.COMPARATOR_HIGH_NUM_EXAMPLES
])


def _get_comparison_only_anomalies(
    anomalies: anomalies_pb2.Anomalies) -> anomalies_pb2.Anomalies:
  """Returns new Anomalies proto with only info from statistics comparison."""
  new_anomalies = anomalies_pb2.Anomalies()
  new_anomalies.CopyFrom(anomalies)
  for feature in list(new_anomalies.anomaly_info):
    # Remove top-level anomaly info description, short_description, and
    # diff_regions entries, since we don't have a good way of separating out the
    # comparison-related portion from the rest.
    new_anomalies.anomaly_info[feature].ClearField('description')
    new_anomalies.anomaly_info[feature].ClearField('short_description')
    new_anomalies.anomaly_info[feature].ClearField('diff_regions')
    reasons_to_keep = [
        r for r in new_anomalies.anomaly_info[feature].reason
        if r.type in _COMPARISON_ANOMALY_TYPES
    ]
    del new_anomalies.anomaly_info[feature].reason[:]
    new_anomalies.anomaly_info[feature].reason.extend(reasons_to_keep)
    # If all of the reasons have been removed, remove the feature's entry from
    # anomaly_info altogether.
    if not new_anomalies.anomaly_info[feature].reason:
      del new_anomalies.anomaly_info[feature]
  new_anomalies.dataset_anomaly_info.ClearField('description')
  new_anomalies.dataset_anomaly_info.ClearField('short_description')
  new_anomalies.dataset_anomaly_info.ClearField('diff_regions')
  dataset_reasons_to_keep = [
      r for r in new_anomalies.dataset_anomaly_info.reason
      if r.type in _COMPARISON_ANOMALY_TYPES
  ]
  del new_anomalies.dataset_anomaly_info.reason[:]
  if dataset_reasons_to_keep:
    new_anomalies.dataset_anomaly_info.reason.extend(dataset_reasons_to_keep)
  else:
    new_anomalies.ClearField('dataset_anomaly_info')
  new_anomalies.ClearField('data_missing')
  return new_anomalies


def _make_schema_from_config(
    config: distribution_validator_pb2.DistributionValidatorConfig,
    statistics_list: statistics_pb2.DatasetFeatureStatisticsList
) -> schema_pb2.Schema:
  """Converts a config to a schema that can be used for data validation."""
  schema = tfdv.infer_schema(statistics_list)
  for feature in config.default_slice_config.feature:
    try:
      schema_feature = schema_util.get_feature(
          schema, path.FeaturePath.from_proto(feature.path))
    except ValueError:
      # Statistics could be missing for features in the config in which case
      # they will not be present in the schema. Just continue; an anomaly will
      # be raised for such features in the step that looks for missing
      # comparisons in the result.
      pass
    else:
      schema_feature.drift_comparator.CopyFrom(feature.distribution_comparator)
  if config.default_slice_config.HasField('num_examples_comparator'):
    schema.dataset_constraints.num_examples_drift_comparator.CopyFrom(
        config.default_slice_config.num_examples_comparator)
  return schema


def _add_anomalies_for_missing_comparisons(
    raw_anomalies: anomalies_pb2.Anomalies,
    config: distribution_validator_pb2.DistributionValidatorConfig
) -> anomalies_pb2.Anomalies:
  """Identifies whether comparison could be done on the configured features.

  If comparison was not done for a configured feature, adds an anomaly flagging
  that.

  Args:
    raw_anomalies: The Anomalies proto to be checked for comparison.
    config: The config that identifies the features for which distribution
      validation will be done.

  Returns:
    An Anomalies proto with anomalies added for features for which comparisons
    could not be done.
  """
  compared_features = set(
      ['.'.join(info.path.step) for info in raw_anomalies.drift_skew_info])
  anomalies = anomalies_pb2.Anomalies()
  anomalies.CopyFrom(raw_anomalies)
  anomalies.anomaly_name_format = (
      anomalies_pb2.Anomalies.AnomalyNameFormat.SERIALIZED_PATH)
  for feature in config.default_slice_config.feature:
    if '.'.join(feature.path.step) in compared_features:
      continue
    feature_key = '.'.join(feature.path.step)
    reason = anomalies.anomaly_info[feature_key].reason.add()
    # TODO(b/239734255): Update with new anomaly type.
    reason.type = anomalies_pb2.AnomalyInfo.Type.STATS_NOT_AVAILABLE
    reason.short_description = 'Comparison could not be done.'
    reason.description = (
        'Validation could not be done, which could be '
        'due to missing data, use of a comparator that is not suitable for the '
        'feature type, or some other reason.')
    anomalies.anomaly_info[feature_key].path.CopyFrom(feature.path)
    anomalies.anomaly_info[
        feature_key].severity = anomalies_pb2.AnomalyInfo.Severity.ERROR
  return anomalies


class Executor(base_executor.BaseExecutor):
  """DistributionValidator component executor."""

  def Do(self, input_dict: Dict[str, List[types.Artifact]],
         output_dict: Dict[str, List[types.Artifact]],
         exec_properties: Dict[str, Any]) -> None:
    """DistributionValidator executor entrypoint.

    This checks for changes in data distributions from one dataset to another,
    based on the summary statitics for those datasets.

    Args:
      input_dict: Input dict from input key to a list of artifacts.
      output_dict: Output dict from key to a list of artifacts.
      exec_properties: A dict of execution properties.

    Returns:
      None
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    # Load and deserialize include splits from execution properties.
    include_splits_list = json_utils.loads(
        exec_properties.get(standard_component_specs.INCLUDE_SPLIT_PAIRS_KEY,
                            'null')) or []
    include_splits = set((test, base) for test, base in include_splits_list)

    test_statistics = artifact_utils.get_single_instance(
        input_dict[standard_component_specs.STATISTICS_KEY])
    baseline_statistics = artifact_utils.get_single_instance(
        input_dict[standard_component_specs.BASELINE_STATISTICS_KEY])

    config = exec_properties.get(
        standard_component_specs.DISTRIBUTION_VALIDATOR_CONFIG_KEY)

    logging.info('Running distribution_validator with config %s', config)

    # Set up pairs of splits to validate.
    split_pairs = []
    for test_split in artifact_utils.decode_split_names(
        test_statistics.split_names):
      for baseline_split in artifact_utils.decode_split_names(
          baseline_statistics.split_names):
        if (test_split, baseline_split) in include_splits:
          split_pairs.append((test_split, baseline_split))
        elif not include_splits and test_split == baseline_split:
          split_pairs.append((test_split, baseline_split))
    if not split_pairs:
      raise ValueError(
          'No split pairs from test and baseline statistics: %s, %s' %
          (test_statistics, baseline_statistics))
    if include_splits:
      missing_split_pairs = include_splits - set(split_pairs)
      if missing_split_pairs:
        raise ValueError(
            'Missing split pairs identified in include_split_pairs: %s' %
            ', '.join([
                '%s_%s' % (test, baseline)
                for test, baseline in missing_split_pairs
            ]))

    anomalies_artifact = artifact_utils.get_single_instance(
        output_dict[standard_component_specs.ANOMALIES_KEY])
    anomalies_artifact.split_names = artifact_utils.encode_split_names(
        ['%s_%s' % (test, baseline) for test, baseline in split_pairs])

    for test_split, baseline_split in split_pairs:
      split_pair = '%s_%s' % (test_split, baseline_split)
      logging.info('Processing split pair %s', split_pair)
      test_stats_split = stats_artifact_utils.load_statistics(
          test_statistics, test_split).proto()
      baseline_stats_split = stats_artifact_utils.load_statistics(
          baseline_statistics, baseline_split).proto()

      schema = _make_schema_from_config(config, baseline_stats_split)
      full_anomalies = tfdv.validate_statistics(
          test_stats_split, schema, previous_statistics=baseline_stats_split)
      anomalies = _get_comparison_only_anomalies(full_anomalies)
      anomalies = _add_anomalies_for_missing_comparisons(anomalies, config)
      io_utils.write_bytes_file(
          os.path.join(anomalies_artifact.uri, 'SplitPair-%s' % split_pair,
                       DEFAULT_FILE_NAME), anomalies.SerializeToString())
