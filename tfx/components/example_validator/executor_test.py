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
from tfx.utils import io_utils
from tfx.utils import types


class ExecutorTest(tf.test.TestCase):

  def test_do(self):
    source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')

    eval_stats_artifact = types.TfxType('ExampleStatsPath', split='eval')
    eval_stats_artifact.uri = os.path.join(source_data_dir,
                                           'statistics_gen/eval/')

    schema_artifact = types.TfxType('SchemaPath')
    schema_artifact.uri = os.path.join(source_data_dir, 'schema_gen/')

    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    validation_output = types.TfxType('ExampleValidationPath')
    validation_output.uri = os.path.join(output_data_dir, 'output')

    input_dict = {
        'stats': [eval_stats_artifact],
        'schema': [schema_artifact],
    }
    output_dict = {
        'output': [validation_output],
    }

    exec_properties = {}

    example_validator_executor = executor.Executor()
    example_validator_executor.Do(input_dict, output_dict, exec_properties)
    self.assertEqual(['anomalies.pbtxt'],
                     tf.gfile.ListDirectory(validation_output.uri))
    anomalies = io_utils.parse_pbtxt_file(
        os.path.join(validation_output.uri, 'anomalies.pbtxt'),
        anomalies_pb2.Anomalies())
    self.assertNotEqual(0, len(anomalies.anomaly_info))
    # TODO(zhitaoli): Add comparison to expected anomolies.


if __name__ == '__main__':
  tf.test.main()
