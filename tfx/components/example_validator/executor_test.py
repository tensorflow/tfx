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
import tensorflow as tf

from tfx.components.example_validator import executor
from tfx.dsl.io import fileio
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs
from tfx.utils import io_utils
from tfx.utils import json_utils
from tensorflow_metadata.proto.v0 import anomalies_pb2


class ExecutorTest(tf.test.TestCase):

  def testDo(self):
    source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')

    eval_stats_artifact = standard_artifacts.ExampleStatistics()
    eval_stats_artifact.uri = os.path.join(source_data_dir, 'statistics_gen')
    eval_stats_artifact.split_names = artifact_utils.encode_split_names(
        ['train', 'eval', 'test'])

    schema_artifact = standard_artifacts.Schema()
    schema_artifact.uri = os.path.join(source_data_dir, 'schema_gen')

    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    validation_output = standard_artifacts.ExampleAnomalies()
    validation_output.uri = os.path.join(output_data_dir, 'output')

    input_dict = {
        standard_component_specs.STATISTICS_KEY: [eval_stats_artifact],
        standard_component_specs.SCHEMA_KEY: [schema_artifact],
    }

    exec_properties = {
        # List needs to be serialized before being passed into Do function.
        standard_component_specs.EXCLUDE_SPLITS_KEY: json_utils.dumps(['test'])
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
    self.assertEqual(0, len(train_anomalies.anomaly_info))
    self.assertEqual(0, len(eval_anomalies.anomaly_info))

    # Assert 'test' split is excluded.
    train_file_path = os.path.join(validation_output.uri, 'Split-test',
                                   'SchemaDiff.pb')
    self.assertFalse(fileio.exists(train_file_path))
    # TODO(zhitaoli): Add comparison to expected anomolies.


if __name__ == '__main__':
  tf.test.main()
