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
"""Tests for tfx.components.example_gen.csv_example_gen.driver."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
import tensorflow as tf
from ml_metadata.proto import metadata_store_pb2
from tfx.components.example_gen.csv_example_gen import driver
from tfx.utils import logging_utils
from tfx.utils import types


class DriverTest(tf.test.TestCase):

  def test_prepare_input_for_processing(self):
    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    self._logger_config = logging_utils.LoggerConfig(
        log_root=os.path.join(output_data_dir, 'log_dir'))

    # Mock metadata.
    mock_metadata = tf.test.mock.Mock()
    csv_example_gen = driver.Driver(self._logger_config, mock_metadata)

    # Mock artifact.
    artifacts = []
    for i in [4, 3, 2, 1]:
      artifact = metadata_store_pb2.Artifact()
      artifact.id = i
      # Only odd ids will be matched to input_base.uri.
      artifact.uri = 'path-{}'.format(i % 2)
      artifacts.append(artifact)

    # Create input dict.
    input_base = types.TfxType(type_name='ExternalPath')
    input_base.uri = 'path-1'
    input_dict = {'input-base': [input_base]}

    # Cache not hit.
    mock_metadata.get_all_artifacts.return_value = []
    mock_metadata.publish_artifacts.return_value = [artifacts[3]]
    updated_input_dict = csv_example_gen._prepare_input_for_processing(
        copy.deepcopy(input_dict))
    self.assertEqual(1, len(updated_input_dict))
    self.assertEqual(1, len(updated_input_dict['input-base']))
    updated_input_base = updated_input_dict['input-base'][0]
    self.assertEqual(1, updated_input_base.id)
    self.assertEqual('path-1', updated_input_base.uri)

    # Cache hit.
    mock_metadata.get_all_artifacts.return_value = artifacts
    mock_metadata.publish_artifacts.return_value = []
    updated_input_dict = csv_example_gen._prepare_input_for_processing(
        copy.deepcopy(input_dict))
    self.assertEqual(1, len(updated_input_dict))
    self.assertEqual(1, len(updated_input_dict['input-base']))
    updated_input_base = updated_input_dict['input-base'][0]
    self.assertEqual(3, updated_input_base.id)
    self.assertEqual('path-1', updated_input_base.uri)


if __name__ == '__main__':
  tf.test.main()
