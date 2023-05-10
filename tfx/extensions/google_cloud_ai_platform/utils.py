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
"""Google Cloud AI Platform utilities."""

from google.api_core import retry as retries
from tfx.proto import google_cloud_ai_platform_pb2


_DEFAULT_EXCEPTION_TYPES = [
    'InternalServerError',
    'TooManyRequests',
    'ServiceUnavailable',
    'ConnectionError',
    'ChunkedEncodingError',
    'TransportError',
]
_DEFAULT_INITIAL_DELAY = 1.0  # seconds
_DEFAULT_MAXIMUM_DELAY = 60.0  # seconds
_DEFAULT_DELAY_MULTIPLIER = 2.0
_DEFAULT_DEADLINE = 60.0 * 2.0  # seconds


def make_retry_from_proto(
    retry_proto: google_cloud_ai_platform_pb2.Retry,
) -> retries.Retry:
  exception_types = retry_proto.predicate.exception_types
  if not exception_types:
    exception_types = _DEFAULT_EXCEPTION_TYPES

  def predicate_fn(exception: Exception) -> bool:
    return exception.__name__ in exception_types

  return retries.Retry(
      predicate=predicate_fn,
      initial=retry_proto.initial or _DEFAULT_INITIAL_DELAY,
      maximum=retry_proto.maximum or _DEFAULT_MAXIMUM_DELAY,
      multiplier=retry_proto.multiplier or _DEFAULT_DELAY_MULTIPLIER,
      deadline=retry_proto.deadline or _DEFAULT_DEADLINE,
  )
