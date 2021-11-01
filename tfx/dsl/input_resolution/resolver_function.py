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
"""Module for ResolverFunction."""
from typing import Callable

from tfx.dsl.input_resolution import resolver_op
from tfx.utils import doc_controls


@doc_controls.do_not_generate_docs
class ResolverFunction:
  """ResolverFunction represents a traceable function of resolver operators.

  ResolverFunction as a whole, takes an ArtifactMultiMap as an argument and
  returns an ArtifactMultiMap.

  Usage:
      @resolver_function
      def trainer_resolver_fn(root):
        result = FooOp(root, foo=1)
        result = BarOp(result, bar='x')
        return result

      trainer_resolver = Resolver(
          function=trainer_resolver_fn,
          examples=example_gen.outputs['examples'],
          ...)
  """

  def __init__(self, f: Callable[..., resolver_op.OpNode]):
    self._function = f

  def __call__(self, *args, **kwargs):
    raise NotImplementedError('Cannot call resolver_function directly.')

  def trace(self, input_node: resolver_op.OpNode):
    # TODO(b/188023509): Better debug support & error message.
    result = self._function(input_node)
    if not isinstance(result, resolver_op.OpNode):
      raise TypeError(
          'Resolver function should return from the operator output, '
          f'but got {result}.')
    return result


def resolver_function(f: Callable[..., resolver_op.OpNode]):
  """Decorator for the resolver function."""
  return ResolverFunction(f)
