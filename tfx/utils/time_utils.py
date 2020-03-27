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
"""Utilities for time related functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import random
import time
from typing import Generator, Optional


def exponential_backoff(
    attempts: int,
    initial_delay_sec: float = 1.0,
    multiplier: float = 2.0,
    truncate_after: Optional[int] = 10) -> Generator[int, None, None]:
  """Exponential backoff retry.

  This function implements a full-jitter exponential backoff algorithm, where
  i'th delay would be:

      delay_i ~ Uniform(0, max_delay)
      where max_delay =
          initial_delay_sec * multiplier ^ min(i, truncate_after - 1)

  Usage:

      for i in exponential_backoff_retry(10):
        do(attempt=i)
        if succeeded:
          break

  Args:
    attempts: Total number of attempts.
    initial_delay_sec: First delay time in seconds. Actual backoff value will be
        smaller as we're jittering the delay. Should be > 0 (default 1.0 sec)
    multiplier: Back off will be exponentially increased with rate `multiplier`.
        Should be >= 1.0 (default 2.0).
    truncate_after: After this number of attempts, maximum delay won't be
        increased. If None, no truncation would occur. Counted from 1 (default
        10).

  Raises:
    ValueError: if argument range is invalid.

  Yields:
    An ordinal of the attempt (starting from zero) after sleeping for
    exponential backoff time.
  """
  if initial_delay_sec <= 0:
    raise ValueError('initial_delay_sec > 0 (got {})'.format(initial_delay_sec))
  if multiplier < 1.0:
    raise ValueError('multiplier >= 1 (got {})'.format(multiplier))
  if truncate_after < 1:
    raise ValueError('truncate_after >= 1 (got {})'.format(truncate_after))

  for attempt in range(attempts):
    yield attempt
    if attempt == attempts - 1:
      return  # No further sleep on final attempts.
    if truncate_after is not None:
      attempt = min(attempt, max(0, truncate_after - 1))
    max_delay = initial_delay_sec * math.pow(multiplier, attempt)
    delay = random.uniform(0, max_delay)  # Full jitter.
    time.sleep(delay)
