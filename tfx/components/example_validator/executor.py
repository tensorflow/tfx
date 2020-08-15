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
"""Generic TFX example_validator executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Any, Dict, List, Text

from absl import logging
import tensorflow_data_validation as tfdv

from tfx import types
from tfx.components.base import base_executor
from tfx.components.example_validator import labels
from tfx.components.util import value_utils
from tfx.types import artifact_utils
from tfx.utils import io_utils
from tfx.utils import json_utils


# Key for statistics in executor input_dict.
STATISTICS_KEY = 'statistics'
# Key for schema in executor input_dict.
SCHEMA_KEY = 'schema'

# Key for exclude splits in executor exec_properties dict.
EXCLUDE_SPLITS_KEY = 'exclude_splits'

# Key for anomalies in executor output_dict.
ANOMALIES_KEY = 'anomalies'

# Default file name for anomalies output.
DEFAULT_FILE_NAME = 'anomalies.pbtxt'


class Executor(base_executor.BaseExecutor):
  """TensorFlow ExampleValidator component executor."""

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """TensorFlow ExampleValidator executor entrypoint.

    This validates the statistics on the 'eval' split against the schema.

    Args:
      input_dict: Input dict from input key to a list of artifacts, including:
        - stats: A list of type `standard_artifacts.ExampleStatistics` generated
          by StatisticsGen.
        - schema: A list of type `standard_artifacts.Schema` which should
          contain a single schema artifact.
      output_dict: Output dict from key to a list of artifacts, including:
        - output: A list of 'ExampleValidationPath' artifact of size one. It
          will include a single pbtxt file which contains all anomalies found.
      exec_properties: A dict of execution properties.
        - exclude_splits: JSON-serialized list of names of splits that the
          example validator should not validate.

    Returns:
      None
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    # Load and deserialize exclude splits from execution properties.
    exclude_splits = json_utils.loads(
        exec_properties.get(EXCLUDE_SPLITS_KEY, 'null')) or []
    if not isinstance(exclude_splits, list):
      raise ValueError('exclude_splits in execution properties needs to be a '
                       'list. Got %s instead.' % type(exclude_splits))
    # Setup output splits.
    stats_artifact = artifact_utils.get_single_instance(
        input_dict[STATISTICS_KEY])
    stats_split_names = artifact_utils.decode_split_names(
        stats_artifact.split_names)
    split_names = [
        split for split in stats_split_names if split not in exclude_splits
    ]
    anomalies_artifact = artifact_utils.get_single_instance(
        output_dict[ANOMALIES_KEY])
    anomalies_artifact.split_names = artifact_utils.encode_split_names(
        split_names)

    schema = io_utils.SchemaReader().read(
        io_utils.get_only_uri_in_dir(
            artifact_utils.get_single_uri(input_dict[SCHEMA_KEY])))

    for split in artifact_utils.decode_split_names(stats_artifact.split_names):
      if split in exclude_splits:
        continue

      logging.info(
          'Validating schema against the computed statistics for '
          'split %s.', split)
      label_inputs = {
          labels.STATS:
              tfdv.load_statistics(
                  io_utils.get_only_uri_in_dir(
                      os.path.join(stats_artifact.uri, split))),
          labels.SCHEMA:
              schema
      }
      output_uri = artifact_utils.get_split_uri(output_dict[ANOMALIES_KEY],
                                                split)
      label_outputs = {labels.SCHEMA_DIFF_PATH: output_uri}
      self._Validate(label_inputs, label_outputs)
      logging.info(
          'Validation complete for split %s. Anomalies written to '
          '%s.', split, output_uri)

  def _Validate(self, inputs: Dict[Text, Any], outputs: Dict[Text,
                                                             Any]) -> None:
    """Validate the inputs and put validate result into outputs.

      This is the implementation part of example validator executor. This is
      intended for using or extending the executor without artifact dependecy.

    Args:
      inputs: A dictionary of labeled input values, including:
        - labels.STATS: the feature statistics to validate
        - labels.SCHEMA: the schema to respect
        - (Optional) labels.ENVIRONMENT: if an environment is specified, only
          validate the feature statistics of the fields in that environment.
          Otherwise, validate all fields.
        - (Optional) labels.PREV_SPAN_FEATURE_STATISTICS: the feature
          statistics of a previous span.
        - (Optional) labels.PREV_VERSION_FEATURE_STATISTICS: the feature
          statistics of a previous version.
        - (Optional) labels.FEATURES_NEEDED: the feature needed to be
          validated on.
        - (Optional) labels.VALIDATION_CONFIG: the configuration of this
          validation.
        - (Optional) labels.EXTERNAL_CONFIG_VERSION: the version number of
          external config file.
      outputs: A dictionary of labeled output values, including:
          - labels.SCHEMA_DIFF_PATH: the path to write the schema diff to
    """
    schema = value_utils.GetSoleValue(inputs, labels.SCHEMA)
    stats = value_utils.GetSoleValue(inputs, labels.STATS)
    schema_diff_path = value_utils.GetSoleValue(
        outputs, labels.SCHEMA_DIFF_PATH)
    anomalies = tfdv.validate_statistics(stats, schema)
    io_utils.write_pbtxt_file(
        os.path.join(schema_diff_path, DEFAULT_FILE_NAME), anomalies)
