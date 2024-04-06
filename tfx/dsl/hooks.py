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

"""DSL for composing execution hooks."""

import abc
from collections.abc import Mapping, Sequence
from typing import Callable, Optional, Union

import attr
from tfx.orchestration import data_types_utils
from tfx.proto.orchestration import execution_hook_pb2

from ml_metadata.proto import metadata_store_pb2


_PrimitiveFlagValueType = Union[int, float, str, bool]
_FlagMap = Union[
    Sequence[tuple[str, _PrimitiveFlagValueType]],
    Mapping[str, _PrimitiveFlagValueType],
]


class PreExecutionOutput(abc.ABC):
  """Python wrapper for the pre execution hook output message."""

  @abc.abstractmethod
  def encode(self) -> execution_hook_pb2.PreExecutionOutput:
    raise NotImplementedError


def _to_value(
    value: _PrimitiveFlagValueType,
) -> metadata_store_pb2.Value:
  result = metadata_store_pb2.Value()
  data_types_utils.set_metadata_value(result, value)
  return result


def _iterate_flags(
    flags: _FlagMap,
) -> Sequence[tuple[str, _PrimitiveFlagValueType]]:
  return list(flags.items()) if isinstance(flags, Mapping) else flags


@attr.define
class BinaryComponentPreOutput(PreExecutionOutput):
  flags: Optional[_FlagMap] = None
  extra_flags: Optional[list[_PrimitiveFlagValueType]] = None
  # TODO(wssong): Add experimental_flags

  def encode(self) -> execution_hook_pb2.PreExecutionOutput:
    return execution_hook_pb2.PreExecutionOutput(
        flags=[
            execution_hook_pb2.PreExecutionOutput.Flag(
                name=key, value=_to_value(value)
            )
            for key, value in _iterate_flags(self.flags or [])
        ],
        extra_flags=[_to_value(value) for value in self.extra_flags or []],
    )


@attr.define
class BCLComponentPreOutput(PreExecutionOutput):
  vars: Optional[_FlagMap] = None
  # TODO(wssong): Add experimental_flags

  def encode(self) -> execution_hook_pb2.PreExecutionOutput:
    return execution_hook_pb2.PreExecutionOutput(
        vars={
            key: _to_value(value)
            for key, value in _iterate_flags(self.vars or [])
        },
    )


@attr.define
class XManagerComponentPreOutput(PreExecutionOutput):
  flags: Optional[_FlagMap] = None
  # TODO(wssong): Add experimental_flags

  def encode(self) -> execution_hook_pb2.PreExecutionOutput:
    return execution_hook_pb2.PreExecutionOutput(
        flags=[
            execution_hook_pb2.PreExecutionOutput.Flag(
                name=key, value=_to_value(value)
            )
            for key, value in _iterate_flags(self.flags or [])
        ],
    )


PreExecutionFunction = Callable[..., Optional[PreExecutionOutput]]
PostExecutionFunction = Callable[..., None]
