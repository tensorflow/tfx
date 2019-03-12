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
"""Tests for tfx.components.trainer.driver."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tfx.components.trainer import driver
from tfx.utils import logging_utils
from tfx.utils import types


class DriverTest(tf.test.TestCase):

  def test_fetch_warm_starting_model(self):
    mock_metadata = tf.test.mock.Mock()
    artifacts = []
    for span in [3, 2, 1]:
      model = types.TfxType(type_name='ModelExportPath')
      model.span = span
      model.uri = 'uri-%d' % span
      artifacts.append(model.artifact)
    mock_metadata.get_all_artifacts.return_value = artifacts
    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    log_root = os.path.join(output_data_dir, 'log_dir')
    logger_config = logging_utils.LoggerConfig(log_root=log_root)
    logger = logging_utils.get_logger(logger_config)
    trainer_driver = driver.Driver(logger, mock_metadata)
    result = trainer_driver._fetch_latest_model()
    self.assertEqual('uri-3', result)


if __name__ == '__main__':
  tf.test.main()
