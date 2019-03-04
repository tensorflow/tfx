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

import os
# Standard Imports

import tensorflow as tf
from tensorflow.python.lib.io import file_io  # pylint: disable=g-direct-tensorflow-import
from tfx.utils import logging_utils


class LoggingUtilsTest(tf.test.TestCase):

  def setUp(self):
    self._log_dir = os.path.join(self.get_temp_dir(), 'log_dir')

  def test_logging(self):
    logger = logging_utils.get_logger(self._log_dir, 'test')
    logger.info('Test')
    log_file_path = os.path.join(self._log_dir, 'test')
    f = file_io.FileIO(log_file_path, mode='r')
    self.assertRegexpMatches(
        f.read(),
        r'^\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d,\d\d\d - logging_utils_test.py:35 - INFO: Test$'
    )


if __name__ == '__main__':
  tf.test.main()
