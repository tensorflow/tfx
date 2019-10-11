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
import absl
import tensorflow_data_validation as tfdv
from typing import Any, Dict, List, Text
from tfx import types
from tfx.components.base import base_executor
from tfx.components.example_validator import labels
from tfx.components.util import value_utils
from tfx.types import artifact_utils
from tfx.utils import io_utils

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
        - stats: A list of 'ExampleStatisticsPath' type which should contain
          split 'eval'. Stats on other splits are ignored.
        - schema: A list of 'SchemaPath' type which should contain a single
          schema artifact.
      output_dict: Output dict from key to a list of artifacts, including:
        - output: A list of 'ExampleValidationPath' artifact of size one. It
          will include a single pbtxt file which contains all anomalies found.
      exec_properties: A dict of execution properties. Not used yet.

    Returns:
      None
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    absl.logging.info('Validating schema against the computed statistics.')
    label_inputs = {
        labels.STATS:
            tfdv.load_statistics(
                io_utils.get_only_uri_in_dir(
                    artifact_utils.get_split_uri(input_dict['stats'], 'eval'))),
        labels.SCHEMA:
            io_utils.SchemaReader().read(
                io_utils.get_only_uri_in_dir(
                    artifact_utils.get_single_uri(input_dict['schema'])))
    }
    output_uri = artifact_utils.get_single_uri(output_dict['output'])
    label_outputs = {labels.SCHEMA_DIFF_PATH: output_uri}
    self._Validate(label_inputs, label_outputs)
    absl.logging.info(
        'Validation complete. Anomalies written to {}.'.format(output_uri))

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
