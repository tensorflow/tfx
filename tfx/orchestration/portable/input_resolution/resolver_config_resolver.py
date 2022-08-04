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
"""Module for NodeInputs.resolver_config based input resolution.

Note that ResolverConfig is deprecated; newly compiled pipeline should contain
InputGraph and InputSpec.InputGraphRef instead.
"""

from typing import Iterable, Union, Sequence, cast, Type

from tfx import types
from tfx.dsl.components.common import resolver
from tfx.dsl.input_resolution import resolver_op
from tfx.dsl.input_resolution.ops import ops
from tfx.orchestration.portable.input_resolution import exceptions
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import json_utils
from tfx.utils import typing_utils

import ml_metadata as mlmd


_ResolverOpClass = Union[
    Type[resolver_op.ResolverOp],
    Type[resolver.ResolverStrategy],
]
# Types that can be used as an argument & return value of an resolver op.
_ResolverIOType = Union[
    Sequence[types.Artifact],
    typing_utils.ArtifactMultiMap,
    Sequence[typing_utils.ArtifactMultiMap],
]


def _run_resolver_strategy(
    input_dict: typing_utils.ArtifactMultiMap,
    *,
    strategy: resolver.ResolverStrategy,
    input_keys: Iterable[str],
    store: mlmd.MetadataStore,
) -> typing_utils.ArtifactMultiMap:
  """Runs a single ResolverStrategy with MLMD store."""
  if not typing_utils.is_artifact_multimap(input_dict):
    raise TypeError(f'Invalid argument type: {input_dict!r}. Must be '
                    'Mapping[str, Sequence[Artifact]].')
  valid_keys = input_keys or set(input_dict.keys())
  valid_inputs = {
      key: list(value)
      for key, value in input_dict.items()
      if key in valid_keys
  }
  bypassed_inputs = {
      key: list(value)
      for key, value in input_dict.items()
      if key not in valid_keys
  }
  result = strategy.resolve_artifacts(store, valid_inputs)
  if result is None:
    raise exceptions.InputResolutionError(f'{strategy} returned None.')
  else:
    result.update(bypassed_inputs)
    return result


def _run_resolver_op(
    arg: _ResolverIOType,
    *,
    op: resolver_op.ResolverOp,
    context: resolver_op.Context,
) -> _ResolverIOType:
  """Runs a single ResolverOp with resolver_op.Context."""
  op.set_context(context)
  return op.apply(arg)


def resolve(
    store: mlmd.MetadataStore,
    input_dict: typing_utils.ArtifactMultiMap,
    resolver_config: pipeline_pb2.ResolverConfig,
) -> _ResolverIOType:
  """Run ResolverConfig.resolver_steps with an input_dict."""
  result = input_dict
  context = resolver_op.Context(store=store)
  for step in resolver_config.resolver_steps:
    cls = ops.get_by_class_path(step.class_path)
    if step.config_json:
      kwargs = json_utils.loads(step.config_json)
    else:
      kwargs = {}
    if issubclass(cls, resolver.ResolverStrategy):
      strategy = cls(**kwargs)  # pytype: disable=not-instantiable
      result = _run_resolver_strategy(
          cast(typing_utils.ArtifactMultiMap, result),
          strategy=strategy,
          input_keys=step.input_keys,
          store=store)
    elif issubclass(cls, resolver_op.ResolverOp):
      op = cls.create(**kwargs)
      result = _run_resolver_op(result, op=op, context=context)
    else:
      raise TypeError(f'Invalid class {cls}. Should be a subclass of '
                      'tfx.dsl.components.common.resolver.ResolverStrategy or '
                      'tfx.dsl.input_resolution.resolver_op.ResolverOp.')
  return result
