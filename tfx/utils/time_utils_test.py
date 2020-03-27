# Lint as: python2, python3
# Copyright 2020 Google LLC. All Rights Reserved.
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
"""Tests for tfx.utils.time_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mock
import tensorflow as tf

from tfx.utils import time_utils


class TimeUtilsTest(tf.test.TestCase):

  def setUp(self):
    super(TimeUtilsTest, self).setUp()
    self.addCleanup(mock.patch.stopall)
    self._sleep_args = []
    mock_sleep = mock.patch('time.sleep').start()
    mock_sleep.side_effect = self._sleep_args.append

  def testExponentialBackoff(self):
    # Evaluate exponential backoff generator
    list(time_utils.exponential_backoff(5))

    # Assert exponential jitter within the range.
    self.assertEqual(len(self._sleep_args), 4)  # Sleep for 4 times.
    self.assertBetween(self._sleep_args[0], 0, 1)
    self.assertBetween(self._sleep_args[1], 0, 2)
    self.assertBetween(self._sleep_args[2], 0, 4)
    self.assertBetween(self._sleep_args[3], 0, 8)

  def testExponentialBackoff_DifferentCoefficient(self):
    # Evaluate exponential backoff generator.
    list(time_utils.exponential_backoff(5, initial_delay_sec=2, multiplier=3))

    # Assert exponential jitter within the range.
    self.assertEqual(len(self._sleep_args), 4)
    self.assertBetween(self._sleep_args[0], 0, 2)
    self.assertBetween(self._sleep_args[1], 0, 6)
    self.assertBetween(self._sleep_args[2], 0, 18)
    self.assertBetween(self._sleep_args[3], 0, 54)

  def testExponentialBackoff_Truncation(self):
    # Evaluate exponential backoff generator.
    list(time_utils.exponential_backoff(20, truncate_after=10))

    # Assert exponential jitter within the range, truncated after 512 = 2^9
    for i, sleep_time in enumerate(self._sleep_args):
      self.assertBetween(sleep_time, 0, min(512, 2 ** i))


if __name__ == '__main__':
  tf.test.main()
