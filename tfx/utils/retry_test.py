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
"""Tests for tfx.utils.retry."""

import time
from unittest import mock

import tensorflow as tf
from tfx.utils import retry


class RetryTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.addCleanup(mock.patch.stopall)
    self.mock_sleep = mock.patch.object(time, 'sleep', autospec=True).start()

  def testSuccessful(self):
    mock_fn = mock.MagicMock()

    @retry.retry()
    def success():
      mock_fn()
      return 42

    self.assertEqual(success(), 42)
    self.assertEqual(mock_fn.call_count, 1)
    self.mock_sleep.assert_not_called()

  def testRetry(self):
    mock_fn = mock.MagicMock()

    @retry.retry()
    def fail():
      mock_fn()
      raise ValueError()

    with self.assertRaises(ValueError):
      fail()
    self.assertEqual(mock_fn.call_count, 1 + 3)
    self.assertEqual(self.mock_sleep.call_count, 3)
    self.mock_sleep.assert_has_calls([mock.call(1)] * 3)

  def testNoRetry(self):
    mock_fn = mock.MagicMock()

    @retry.retry(max_retries=0)
    def fail():
      mock_fn()
      raise ValueError()

    with self.assertRaises(ValueError):
      fail()
    self.assertEqual(mock_fn.call_count, 1)
    self.mock_sleep.assert_not_called()

  def testNoDelay(self):

    @retry.retry(delay_seconds=0)
    def fail():
      raise ValueError()

    with self.assertRaises(ValueError):
      fail()
    self.mock_sleep.assert_not_called()

  def testWrongException(self):
    mock_fn = mock.MagicMock()

    @retry.retry(expected_exception=KeyError)
    def fail():
      mock_fn()
      raise ValueError()

    with self.assertRaises(ValueError):
      fail()
    self.assertEqual(mock_fn.call_count, 1)

  def testIgnoreEventualFailure(self):
    mock_fn = mock.MagicMock()

    @retry.retry(max_retries=2, ignore_eventual_failure=True)
    def fail():
      mock_fn()
      raise ValueError()

    self.assertIsNone(fail())
    self.assertEqual(mock_fn.call_count, 1 + 2)


if __name__ == '__main__':
  tf.test.main()
