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
"""Tests for tfx.components.example_validator.executor."""

import os
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
from tensorflow_data_validation.anomalies.proto import custom_validation_config_pb2
from tfx.components.example_validator import executor
from tfx.dsl.io import fileio
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs
from tfx.utils import io_utils
from tfx.utils import json_utils
from tensorflow_metadata.proto.v0 import anomalies_pb2

from google.protobuf import text_format


class ExecutorTest(parameterized.TestCase):

  def _get_temp_dir(self):
    return tempfile.mkdtemp()

  def _assert_equal_anomalies(self, actual_anomalies, expected_anomalies):
    # Check if the actual anomalies matches with the expected anomalies.
    for feature_name in expected_anomalies:
      self.assertIn(feature_name, actual_anomalies.anomaly_info)
      # Do not compare diff_regions.
      actual_anomalies.anomaly_info[feature_name].ClearField('diff_regions')

      self.assertEqual(actual_anomalies.anomaly_info[feature_name],
                       expected_anomalies[feature_name])
    self.assertEqual(
        len(actual_anomalies.anomaly_info), len(expected_anomalies))

  @parameterized.named_parameters(
      {
          'testcase_name': 'No_anomalies',
          'custom_validation_config': None,
          'expected_anomalies': {}
      }, {
          'testcase_name':
              'Custom_validation',
          'custom_validation_config':
              """
              feature_validations {
              feature_path { step: 'company' }
              validations {
                sql_expression: 'feature.string_stats.common_stats.min_num_values > 5'
                severity: ERROR
                description: 'Feature does not have enough values.'
                }
              }
              """,
          'expected_anomalies': {
              'company': text_format.Parse(
                  """
                  path {
                    step: 'company'
                  }
                  severity: ERROR
                  short_description: 'Feature does not have enough values.'
                  description: 'Custom validation triggered anomaly. Query: feature.string_stats.common_stats.min_num_values > 5 Test dataset: default slice'
                  reason {
                    description: 'Custom validation triggered anomaly. Query: feature.string_stats.common_stats.min_num_values > 5 Test dataset: default slice'
                    type: CUSTOM_VALIDATION
                    short_description: 'Feature does not have enough values.'
                  }
                  """, anomalies_pb2.AnomalyInfo())
          }
      })
  def testDo(self, custom_validation_config, expected_anomalies):
    source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')

    eval_stats_artifact = standard_artifacts.ExampleStatistics()
    eval_stats_artifact.uri = os.path.join(source_data_dir, 'statistics_gen')
    eval_stats_artifact.split_names = artifact_utils.encode_split_names(
        ['train', 'eval', 'test'])

    schema_artifact = standard_artifacts.Schema()
    schema_artifact.uri = os.path.join(source_data_dir, 'schema_gen')

    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self._get_temp_dir()),
        self._testMethodName)

    validation_output = standard_artifacts.ExampleAnomalies()
    validation_output.uri = os.path.join(output_data_dir, 'output')

    input_dict = {
        standard_component_specs.STATISTICS_KEY: [eval_stats_artifact],
        standard_component_specs.SCHEMA_KEY: [schema_artifact],
    }

    if custom_validation_config is not None:
      custom_validation_config = text_format.Parse(
          custom_validation_config,
          custom_validation_config_pb2.CustomValidationConfig()
      )
    exec_properties = {
        # List needs to be serialized before being passed into Do function.
        standard_component_specs.EXCLUDE_SPLITS_KEY:
            json_utils.dumps(['test']),
        standard_component_specs.CUSTOM_VALIDATION_CONFIG_KEY:
            custom_validation_config,
    }

    output_dict = {
        standard_component_specs.ANOMALIES_KEY: [validation_output],
    }

    example_validator_executor = executor.Executor()
    example_validator_executor.Do(input_dict, output_dict, exec_properties)

    self.assertEqual(
        artifact_utils.encode_split_names(['train', 'eval']),
        validation_output.split_names)

    # Check example_validator outputs.
    train_anomalies_path = os.path.join(validation_output.uri, 'Split-train',
                                        'SchemaDiff.pb')
    eval_anomalies_path = os.path.join(validation_output.uri, 'Split-eval',
                                       'SchemaDiff.pb')
    self.assertTrue(fileio.exists(train_anomalies_path))
    self.assertTrue(fileio.exists(eval_anomalies_path))
    train_anomalies_bytes = io_utils.read_bytes_file(train_anomalies_path)
    train_anomalies = anomalies_pb2.Anomalies()
    train_anomalies.ParseFromString(train_anomalies_bytes)
    eval_anomalies_bytes = io_utils.read_bytes_file(eval_anomalies_path)
    eval_anomalies = anomalies_pb2.Anomalies()
    eval_anomalies.ParseFromString(eval_anomalies_bytes)

    self._assert_equal_anomalies(train_anomalies, expected_anomalies)
    self._assert_equal_anomalies(eval_anomalies, expected_anomalies)

    # Assert 'test' split is excluded.
    train_file_path = os.path.join(validation_output.uri, 'Split-test',
                                   'SchemaDiff.pb')
    self.assertFalse(fileio.exists(train_file_path))
    # TODO(zhitaoli): Add comparison to expected anomolies.


if __name__ == '__main__':
  absltest.main()
