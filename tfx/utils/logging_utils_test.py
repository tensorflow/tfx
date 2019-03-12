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
"""Tests for tfx.utils.logging_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import os
# Standard Imports
import tensorflow as tf
from tensorflow.python.lib.io import file_io  # pylint: disable=g-direct-tensorflow-import
from tfx.utils import logging_utils


class LoggingUtilsTest(tf.test.TestCase):

  def setUp(self):
    self._log_root = os.path.join(self.get_temp_dir(), 'log_dir')
    self._logger_config = logging_utils.LoggerConfig(log_root=self._log_root)

  def test_logging(self):
    """Ensure a logged string actually appears in the log file."""
    logger = logging_utils.get_logger(self._logger_config)
    logger.info('Test')
    log_file_path = os.path.join(self._log_root)
    f = file_io.FileIO(os.path.join(log_file_path, 'tfx.log'), mode='r')
    self.assertRegexpMatches(
        f.read(),
        r'^\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d,\d\d\d - : \(logging_utils_test.py:\d\d\) - INFO: Test$'
    )

  def test_default_settings(self):
    """Ensure log defaults are set correctly."""
    config = logging_utils.LoggerConfig()
    self.assertEqual(config.log_root, '/var/tmp/tfx/logs')
    self.assertEqual(config.log_level, logging.INFO)
    self.assertEqual(config.pipeline_name, '')
    self.assertEqual(config.worker_name, '')

  def test_override_settings(self):
    """Ensure log overrides are set correctly."""
    config = logging_utils.LoggerConfig(log_root='path', log_level=logging.WARN,
                                        pipeline_name='pipe', worker_name='wrk')
    self.assertEqual(config.log_root, 'path')
    self.assertEqual(config.log_level, logging.WARN)
    self.assertEqual(config.pipeline_name, 'pipe')
    self.assertEqual(config.worker_name, 'wrk')

if __name__ == '__main__':
  tf.test.main()
