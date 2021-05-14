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


class ResolverFunction:
  """ResolverFunction represents a traced function of resolver operators.

  ResolverFunction as a whole, takes an ArtifactMultimap as an argument and
  returns ArtifactMultimap.

  Usage:
      def TrainerResolver(input_dict):
        result = FooOp(input_dict, foo=1)
        result = BarOp(result, bar='x')
        return result
      rf = ResolverFunction(TrainerResolver)
      rf.trace()
      rf.output_node  # BarOp(FooOp(INPUT_NODE, foo=1), bar='x')
  """

  def __init__(self, f: Callable[..., resolver_op.OpNode]):
    self._function = f
    self._output_node = None

  @property
  def output_node(self) -> resolver_op.OpNode:
    if self._output_node is None:
      raise ValueError('ResolverFunction is not traced.')
    return self._output_node

  def trace(self):
    # TODO(b/188023509): Better debug support & error message.
    output_node = self._function(resolver_op.OpNode.INPUT_NODE)
    if not isinstance(output_node, resolver_op.OpNode):
      raise TypeError(
          'Resolver function should return from the operator output, '
          f'but got {output_node}.')
    self._output_node = output_node
