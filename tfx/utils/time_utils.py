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

import datetime
import time


def utc_timestamp() -> float:
  """Get timestamp for UTC now.

  Unlike time.time() where you get timezone-aware timestamp of now, this
  function returns the unix timestamp (i.e. timezone-agnostic) of now in
  microseconds precision.

  Returns:
    A microsecond precision UTC timestamp.
  """
  # TODO(b/149535021): for python>=3.3 we have datetime.timestamp() method.
  dt = datetime.datetime.utcnow()
  return time.mktime(dt.timetuple()) + dt.microsecond * 1e-6
