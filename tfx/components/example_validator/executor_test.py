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


class ExecutorTest(tf.test.TestCase):

  def setUp(self):
    super(ExecutorTest, self).setUp()
    self.source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')

    self.eval_stats_artifact = standard_artifacts.ExampleStatistics()
    self.eval_stats_artifact.uri = os.path.join(
        self.source_data_dir, 'statistics_gen')
    self.eval_stats_artifact.split_names = artifact_utils.encode_split_names(
        ['eval'])

    self.schema_artifact = standard_artifacts.Schema()
    self.schema_artifact.uri = os.path.join(self.source_data_dir, 'schema_gen')

    self.output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    self.validation_output = standard_artifacts.ExampleAnomalies()
    self.validation_output.uri = os.path.join(self.output_data_dir, 'output')

  def _do(self):
    example_validator_executor = executor.Executor()
    example_validator_executor.Do(
        self.input_dict, self.output_dict, self.exec_properties)
    self.assertEqual(
        ['anomalies.pbtxt'],
        tf.io.gfile.listdir(self.validation_output.uri))
    anomalies = io_utils.parse_pbtxt_file(
        os.path.join(self.validation_output.uri, 'anomalies.pbtxt'),
        anomalies_pb2.Anomalies())
    self.assertNotEqual(0, len(anomalies.anomaly_info))

  def testDo(self):
    self.input_dict = {
        executor.STATISTICS_KEY: [self.eval_stats_artifact],
        executor.SCHEMA_KEY: [self.schema_artifact],
    }
    self.output_dict = {
        executor.ANOMALIES_KEY: [self.validation_output],
    }

    self.exec_properties = {}

    self._do()
    # TODO(zhitaoli): Add comparison to expected anomolies.

  def testDoSkewDetection(self):
    training_statistics = standard_artifacts.ExampleStatistics()
    training_statistics.uri = os.path.join(
        self.source_data_dir, 'trainer/current')
    training_statistics.split_names = artifact_utils.encode_split_names(
        ['eval_model_dir'])

    self.input_dict = {
        executor.STATISTICS_KEY: [self.eval_stats_artifact],
        executor.SCHEMA_KEY: [self.schema_artifact],
        executor.TRAINING_STATISTICS_KEY: [training_statistics],
    }
    self.output_dict = {
        executor.ANOMALIES_KEY: [self.validation_output],
    }

    self.exec_properties = {}

    self._do()

if __name__ == '__main__':
  tf.test.main()
