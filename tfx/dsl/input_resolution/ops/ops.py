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

from typing import Dict, Type, Union

from tfx.dsl.components.common import resolver
from tfx.dsl.input_resolution import resolver_op
from tfx.dsl.input_resolution.ops import all_spans_op
from tfx.dsl.input_resolution.ops import consecutive_spans_op
from tfx.dsl.input_resolution.ops import exclude_spans_op
from tfx.dsl.input_resolution.ops import latest_create_time_op
from tfx.dsl.input_resolution.ops import latest_pipeline_run_outputs_op
from tfx.dsl.input_resolution.ops import latest_span_op
from tfx.dsl.input_resolution.ops import latest_version_op
from tfx.dsl.input_resolution.ops import skip_if_empty_op
from tfx.dsl.input_resolution.ops import skip_if_less_than_n_spans_op
from tfx.dsl.input_resolution.ops import sliding_window_op
from tfx.dsl.input_resolution.ops import static_span_range_op
from tfx.dsl.input_resolution.ops import temp_latest_create_time_op
from tfx.dsl.input_resolution.ops import temp_latest_span_op
from tfx.dsl.input_resolution.ops import unnest_op
from tfx.dsl.input_resolution.strategies import conditional_strategy
from tfx.dsl.input_resolution.strategies import latest_artifact_strategy
from tfx.dsl.input_resolution.strategies import latest_blessed_model_strategy
from tfx.dsl.input_resolution.strategies import span_range_strategy
from tfx.utils import name_utils
from tfx.utils import typing_utils


_ResolverOpType = Type[resolver_op.ResolverOp]
_ResolverStrategyType = Type[resolver.ResolverStrategy]
_OpTypes = Union[_ResolverOpType, _ResolverStrategyType]
_OPS_BY_CLASSPATH: Dict[str, _OpTypes] = {}
_OPS_BY_NAME: Dict[str, _ResolverOpType] = {}


def _register_op(cls: _ResolverOpType) -> None:
  class_path = name_utils.get_full_name(cls, strict_check=False)
  if class_path in _OPS_BY_CLASSPATH:
    raise ValueError(f'Duplicated class path {class_path} while registering.')
  _OPS_BY_CLASSPATH[class_path] = cls
  if cls.canonical_name in _OPS_BY_NAME:
    raise ValueError(f'Duplicated name {cls.canonical_name} while registering.')
  _OPS_BY_NAME[cls.canonical_name] = cls

# go/keep-sorted start
AllSpans = all_spans_op.AllSpans
ConsecutiveSpans = consecutive_spans_op.ConsecutiveSpans
ExcludeSpans = exclude_spans_op.ExcludeSpans
LatestCreateTime = latest_create_time_op.LatestCreateTime
LatestPipelineRunOutputs = latest_pipeline_run_outputs_op.LatestPipelineRunOutputs
LatestSpan = latest_span_op.LatestSpan
LatestVersion = latest_version_op.LatestVersion
SkipIfEmpty = skip_if_empty_op.SkipIfEmpty
SkipIfLessThanNSpans = skip_if_less_than_n_spans_op.SkipIfLessThanNSpans
SlidingWindow = sliding_window_op.SlidingWindow
StaticSpanRange = static_span_range_op.StaticSpanRange
TempLatestCreateTime = temp_latest_create_time_op.TempLatestCreateTime
TempLatestSpan = temp_latest_span_op.TempLatestSpan
Unnest = unnest_op.Unnest
# go/keep-sorted end
# go/keep-sorted start
_register_op(AllSpans)
_register_op(ConsecutiveSpans)
_register_op(ExcludeSpans)
_register_op(LatestCreateTime)
_register_op(LatestPipelineRunOutputs)
_register_op(LatestSpan)
_register_op(LatestVersion)
_register_op(SkipIfEmpty)
_register_op(SkipIfLessThanNSpans)
_register_op(SlidingWindow)
_register_op(StaticSpanRange)
_register_op(TempLatestCreateTime)
_register_op(TempLatestSpan)
_register_op(Unnest)
# go/keep-sorted end


def _register_strategy(cls: _ResolverStrategyType) -> None:
  class_path = name_utils.get_full_name(cls, strict_check=False)
  if class_path in _OPS_BY_CLASSPATH:
    raise ValueError(f'Duplicated class path {class_path} while registering.')
  _OPS_BY_CLASSPATH[class_path] = cls

# For ResolverStrategy, register them but do not expose their public name.
# go/keep-sorted start
_register_strategy(conditional_strategy.ConditionalStrategy)
_register_strategy(latest_artifact_strategy.LatestArtifactStrategy)
_register_strategy(latest_blessed_model_strategy.LatestBlessedModelStrategy)
_register_strategy(span_range_strategy.SpanRangeStrategy)
# go/keep-sorted end


def testonly_register(cls: _OpTypes) -> _OpTypes:
  if issubclass(cls, resolver_op.ResolverOp):
    _register_op(cls)
  else:
    _register_strategy(cls)
  return cls


def get_by_class_path(class_path: str) -> _OpTypes:
  """Get ResolverOp or ResolverStrategy class from class path string."""
  if class_path in _OPS_BY_CLASSPATH:
    result = _OPS_BY_CLASSPATH[class_path]
  else:
    result = name_utils.resolve_full_name(class_path)
  if not typing_utils.is_compatible(result, _OpTypes):
    raise ValueError(f'{class_path} is not compatible with {_OpTypes}.')
  return result


def get_by_name(name: str) -> _ResolverOpType:
  """Get ResolverOp class from registered name."""
  return _OPS_BY_NAME[name]
