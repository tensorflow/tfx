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
"""Tests for tfx.components.example_validator.executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tensorflow_metadata.proto.v0 import anomalies_pb2
from tfx.components.example_validator import executor
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.utils import io_utils
from tfx.utils import json_utils


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
    validation_output.split_names = artifact_utils.encode_split_names(
        ['train', 'eval'])

    input_dict = {
        executor.STATISTICS_KEY: [eval_stats_artifact],
        executor.SCHEMA_KEY: [schema_artifact],
    }

    exec_properties = {
        # List needs to be serialized before being passed into Do function.
        executor.EXCLUDE_SPLITS_KEY:
            json_utils.dumps(['test'])
    }

    output_dict = {
        executor.ANOMALIES_KEY: [validation_output],
    }

    example_validator_executor = executor.Executor()
    example_validator_executor.Do(input_dict, output_dict, exec_properties)

    # Check example_validator outputs.
    train_anomalies_path = os.path.join(validation_output.uri, 'train',
                                        'anomalies.pbtxt')
    eval_anomalies_path = os.path.join(validation_output.uri, 'eval',
                                       'anomalies.pbtxt')
    self.assertTrue(tf.io.gfile.exists(train_anomalies_path))
    self.assertTrue(tf.io.gfile.exists(eval_anomalies_path))
    train_anomalies = io_utils.parse_pbtxt_file(train_anomalies_path,
                                                anomalies_pb2.Anomalies())
    eval_anomalies = io_utils.parse_pbtxt_file(eval_anomalies_path,
                                               anomalies_pb2.Anomalies())
    self.assertEqual(0, len(train_anomalies.anomaly_info))
    self.assertEqual(0, len(eval_anomalies.anomaly_info))

    # Assert 'test' split is excluded.
    train_file_path = os.path.join(validation_output.uri, 'test',
                                   'anomalies.pbtxt')
    self.assertFalse(tf.io.gfile.exists(train_file_path))
    # TODO(zhitaoli): Add comparison to expected anomolies.


if __name__ == '__main__':
  tf.test.main()
