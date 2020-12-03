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
"""Simple utility for retries."""

import functools
import time

from typing import Type

from absl import logging


def retry(max_retries: int = 3,
          delay_seconds: int = 1,
          expected_exception: Type[Exception] = Exception,
          ignore_eventual_failure: bool = False):
  """Function decorator to retry a function automatically.

  Example:
    from tfx.utils import retry
    @retry.retry()
    def some_fragile_func():
      ...

  If `ignore_eventual_failure` is False, the last expected exception caught
  will raised from this function. If `ignore_eventual_failure` is True,
  no exception will raised and will return None.

  Args:
    max_retries: number of retries. Total trial count becomes 1 + max_retries.
    delay_seconds: there will be a predefined delay between each trial.
    expected_exception: this exception will be regarded as retriable failures.
    ignore_eventual_failure: See above description.

  Returns:
    A decorator for retrying logic.
  """

  def decorator_retry(func):

    @functools.wraps(func)
    def with_retry(*args, **kwargs):
      last_exception = None
      for retry_no in range(max_retries + 1):
        if retry_no > 0:
          if delay_seconds > 0:
            time.sleep(delay_seconds)
          logging.info('[Retrying "%s" %d/%d]', func.__name__, retry_no,
                       max_retries)
        try:
          return func(*args, **kwargs)
        except expected_exception as err:  # pylint:disable=broad-except
          logging.info('%s', err)
          last_exception = err
      logging.info('Max number of retries(%d) reached for %s.', max_retries,
                   func.__name__)
      if not ignore_eventual_failure:
        raise last_exception

    return with_retry

  return decorator_retry
