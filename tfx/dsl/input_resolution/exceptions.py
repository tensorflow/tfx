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
"""Module for input resolution exceptions.

Theses exceptions can be raised from ResolverStartegy or ResolverOp
implementation, and each exception is specially handled in input resolution
process. Other errors raised during the input resolution will not be catched
and be propagated to the input resolution caller.
"""
# Disable lint errors that enforces all exception name to end with -Error.
# pylint: disable=g-bad-exception-name


class InputResolutionSignal(Exception):
  """Base class for all input resolution related exception classes."""


class SkipSignal(InputResolutionSignal):
  """Raise SkipSignal to resolve an empty list right away.

  Empty list (ResolutionSucceeded([])) means there are no input dicts to
  process, but how pipeline runner handles it is different in different
  contexts:
  - In synchronous mode, if component execution is skipped, there is no output
    artifacts to run downstream components. Therefore empty input resolution
    result will abort the pipeline.
      - Exception is the Resolver node which can resolve nothing.
  - In asynchronous mode, each component is constantly running as well as
    loosely coupled with other components, so no inputs for this time does not
    prevent executing other components. Empty list is a perfectly valid result.
  """


class IgnoreSignal(InputResolutionSignal):
  """Raise IgnoreSignal to ignore the current node execution.

  Ignore() is a special input resolution result that the pipeline would run as
  if the node of this input resolution result does not exist. It effectively
  erases current node from the pipeline during runtime.

  Unlike SkipSignal, this does not abort the pipeline. Ignore() is only valid
  in synchronous mode.
  """
