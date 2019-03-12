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
"""Tests for tfx.components.model_validator.driver."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tfx.components.model_validator import driver
from tfx.utils import logging_utils
from tfx.utils import types


class DriverTest(tf.test.TestCase):

  def test_fetch_last_blessed_model(self):
    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    self._logger_config = logging_utils.LoggerConfig(
        log_root=os.path.join(output_data_dir, 'log_dir'))

    # Mock metadata.
    mock_metadata = tf.test.mock.Mock()
    model_validator_driver = driver.Driver(self._logger_config, mock_metadata)

    # No blessed model.
    mock_metadata.get_all_artifacts.return_value = []
    self.assertEqual((None, None),
                     model_validator_driver._fetch_last_blessed_model())

    # Mock blessing artifacts.
    artifacts = []
    for span in [4, 3, 2, 1]:
      model_blessing = types.TfxType(type_name='ModelBlessingPath')
      model_blessing.span = span
      model_blessing.set_string_custom_property('current_model',
                                                'uri-%d' % span)
      model_blessing.set_int_custom_property('current_model_id', span)
      # Only odd spans are "blessed"
      model_blessing.set_int_custom_property('blessed', span % 2)
      artifacts.append(model_blessing.artifact)
    mock_metadata.get_all_artifacts.return_value = artifacts
    self.assertEqual(('uri-3', 3),
                     model_validator_driver._fetch_last_blessed_model())


if __name__ == '__main__':
  tf.test.main()
