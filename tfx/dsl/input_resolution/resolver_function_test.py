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
"""Tests for tfx.dsl.input_resolution.resolver_function."""
import tensorflow as tf

from tfx.dsl.components.common import resolver
from tfx.dsl.input_resolution import resolver_function
from tfx.dsl.input_resolution import resolver_op


class Foo(resolver_op.ResolverOp):
  foo = resolver_op.ResolverOpProperty(type=int)

  def apply(self, input_dict):
    return input_dict


class Bar(resolver_op.ResolverOp):
  bar = resolver_op.ResolverOpProperty(type=str, default='bar')

  def apply(self, input_dict):
    return input_dict


class FooStrategy(resolver.ResolverStrategy):

  def __init__(self, foo: int):
    self._foo = foo

  def resolve_artifacts(self, store, input_dict):
    return input_dict


class ResolverFunctionTest(tf.test.TestCase):

  def testTrace(self):

    def resolve(input_dict):
      result = Foo(input_dict, foo=1)
      result = Bar(result, bar='x')
      return result

    rf = resolver_function.ResolverFunction(resolve)
    output_node = rf.trace(resolver_op.OpNode.INPUT_NODE)
    self.assertEqual(repr(output_node),
                     "Bar(Foo(INPUT_NODE, foo=1), bar='x')")

  def testTrace_ResolverOpAndResolverStrategyInterop(self):

    def resolve(input_dict):
      result = Foo(input_dict, foo=1)
      result = FooStrategy.as_resolver_op(result, foo=2)
      result = Bar(result, bar='x')
      return result

    rf = resolver_function.ResolverFunction(resolve)
    output_node = rf.trace(resolver_op.OpNode.INPUT_NODE)
    self.assertEqual(
        repr(output_node),
        "Bar(FooStrategy(Foo(INPUT_NODE, foo=1), foo=2), bar='x')")

  def testTrace_BadReturnValue(self):

    def resolve(unused_input_dict):
      return {'examples': []}

    rf = resolver_function.ResolverFunction(resolve)
    with self.assertRaises(TypeError):
      rf.trace(resolver_op.OpNode.INPUT_NODE)


if __name__ == '__main__':
  tf.test.main()
