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
"""Module for declaring and parsing TFX system flags for python executable."""

from typing import TypeVar

from absl import flags
from tfx.orchestration.portable import data_types
from tfx.orchestration.python_execution_binary import python_execution_binary_utils as flag_utils

_LEGACY_EXECUTION_INVOCATION = flags.DEFINE_string(
    'tfx_execution_info_b64',
    None,
    'url safe base64 encoded tfx.orchestration.ExecutionInvocation proto',
)

_T = TypeVar('_T')


def _require_flag(flag: flags.FlagHolder[_T]) -> _T:
  if not flag.present:
    raise flags.ValidationError(f'Flag --{flag.name} is required.')
  return flag.value


def parse_execution_info() -> data_types.ExecutionInfo:
  exec_invocation_b64 = _require_flag(_LEGACY_EXECUTION_INVOCATION)
  return flag_utils.deserialize_execution_info(exec_invocation_b64)
