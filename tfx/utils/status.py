# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Status class and related utilities.

Keep the status codes aligned with `google.rpc.Code`:
https://github.com/googleapis/googleapis/blob/master/google/rpc/code.proto
"""

import enum
from typing import Optional

import attr


@enum.unique
class Code(enum.IntEnum):
  """Convenience enum class for status codes that mirrors `google.rpc.Code`.

  Keep the status codes aligned with `google.rpc.Code`:
  https://github.com/googleapis/googleapis/blob/master/google/rpc/code.proto
  """
  OK = 0
  CANCELLED = 1
  UNKNOWN = 2
  INVALID_ARGUMENT = 3
  DEADLINE_EXCEEDED = 4
  NOT_FOUND = 5
  ALREADY_EXISTS = 6
  PERMISSION_DENIED = 7
  RESOURCE_EXHAUSTED = 8
  FAILED_PRECONDITION = 9
  ABORTED = 10
  OUT_OF_RANGE = 11
  UNIMPLEMENTED = 12
  INTERNAL = 13
  UNAVAILABLE = 14
  DATA_LOSS = 15
  UNAUTHENTICATED = 16


@attr.s(auto_attribs=True, frozen=True)
class Status:
  """Class to record status of operations.

  Attributes:
    code: A status code integer. Should be an enum value of `google.rpc.Code`.
    message: An optional message associated with the status.
  """
  code: int
  message: Optional[str] = None


class StatusNotOkError(Exception):
  """Error class useful when status not OK."""

  def __init__(self, code: int, message: str):
    self.code = code
    self.message = message
    Exception.__init__(self, str(self))

  def status(self) -> Status:
    return Status(code=self.code, message=self.message)

  def __str__(self) -> str:
    return f'Error ({self.code}): {self.message}'
