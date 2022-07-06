# Copyright 2022 Google LLC. All Rights Reserved.
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
"""Module for pre-importing all known ResolverOp for dependency tracking."""

from typing import Type, Union, Dict, Optional

from tfx.dsl.components.common import resolver
from tfx.dsl.input_resolution import resolver_op
from tfx.dsl.input_resolution.ops import skip_if_empty_op
from tfx.dsl.input_resolution.ops import unnest_op
from tfx.dsl.input_resolution.strategies import conditional_strategy
from tfx.dsl.input_resolution.strategies import latest_artifact_strategy
from tfx.dsl.input_resolution.strategies import latest_blessed_model_strategy
from tfx.dsl.input_resolution.strategies import span_range_strategy
from tfx.utils import name_utils

_ResolverOpType = Type[resolver_op.ResolverOp]
_ResolverStrategyType = Type[resolver.ResolverStrategy]
_OpTypes = Union[_ResolverOpType, _ResolverStrategyType]
_OPS_BY_CLASSPATH: Dict[str, _OpTypes] = {}
_OPS_BY_NAME: Dict[str, _ResolverOpType] = {}


def _register_op(cls: _ResolverOpType, name: Optional[str] = None) -> None:
  class_path = name_utils.get_full_name(cls, strict_check=False)
  _OPS_BY_CLASSPATH[class_path] = cls
  if name is None:
    name = class_path
  if name in _OPS_BY_NAME:
    raise ValueError(f'Duplicated name {name} while registering.')
  _OPS_BY_NAME[name] = cls

# go/keep-sorted start
SkipIfEmpty = skip_if_empty_op.SkipIfEmpty
Unnest = unnest_op.Unnest
# go/keep-sorted end
# go/keep-sorted start
_register_op(SkipIfEmpty, name='tfx.internal.SkipIfEmpty')
_register_op(Unnest, name='tfx.internal.Unnest')
# go/keep-sorted end


def _register_strategy(cls: _ResolverStrategyType) -> None:
  _OPS_BY_CLASSPATH[name_utils.get_full_name(cls, strict_check=False)] = cls

# For ResolverStrategy, register them but do not expose their public name.
# go/keep-sorted start
_register_strategy(conditional_strategy.ConditionalStrategy)
_register_strategy(latest_artifact_strategy.LatestArtifactStrategy)
_register_strategy(latest_blessed_model_strategy.LatestBlessedModelStrategy)
_register_strategy(span_range_strategy.SpanRangeStrategy)
# go/keep-sorted end


def testonly_register(cls: _OpTypes) -> _OpTypes:
  if issubclass(cls, resolver_op.ResolverOp):
    _register_op(cls, name=cls.__name__)
  else:
    _register_strategy(cls)
  return cls


def get_by_class_path(class_path: str) -> _OpTypes:
  """Get ResolverOp class from class path string."""
  return _OPS_BY_CLASSPATH[class_path]


def get_by_name(name: str) -> _ResolverOpType:
  """Get ResolverOp class from registered name."""
  return _OPS_BY_NAME[name]
