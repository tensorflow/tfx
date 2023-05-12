# Copyright 2023 Google LLC. All Rights Reserved.
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
"""Tests for tfx.extensions.google_cloud_ai_platform.utils."""

from google.api_core import exceptions
from google.api_core import retry as retries
from google.auth import exceptions as auth_exceptions
import requests.exceptions
import tensorflow as tf
from tfx.extensions.google_cloud_ai_platform import utils
from tfx.proto import google_cloud_ai_platform_pb2


class UtilsTest(tf.test.TestCase):

  def testMakeRetryFromProto_Default(self):
    retry_proto = google_cloud_ai_platform_pb2.Retry()
    retry = utils.make_retry_from_proto(retry_proto)

    self.assertIsInstance(retry, retries.Retry)
    self.assertAlmostEqual(retry._initial, utils._DEFAULT_INITIAL_DELAY)
    self.assertAlmostEqual(retry._maximum, utils._DEFAULT_MAXIMUM_DELAY)
    self.assertAlmostEqual(retry._multiplier, utils._DEFAULT_DELAY_MULTIPLIER)
    self.assertAlmostEqual(retry._deadline, utils._DEFAULT_DEADLINE)
    for exception in [
        exceptions.InternalServerError,
        exceptions.TooManyRequests,
        exceptions.ServiceUnavailable,
        requests.exceptions.ConnectionError,
        requests.exceptions.ChunkedEncodingError,
        auth_exceptions.TransportError,
    ]:
      self.assertTrue(retry._predicate(exception))

  def testMakeRetryFromProto_ExplicitExecutionTypes(self):
    retry_proto = google_cloud_ai_platform_pb2.Retry(
        predicate=google_cloud_ai_platform_pb2.Retry.Predicate(
            exception_types=['InternalServerError']
        )
    )
    retry = utils.make_retry_from_proto(retry_proto)

    self.assertIsInstance(retry, retries.Retry)
    self.assertTrue(retry._predicate(exceptions.InternalServerError))
    self.assertFalse(retry._predicate(exceptions.TooManyRequests))


if __name__ == '__main__':
  tf.test.main()
