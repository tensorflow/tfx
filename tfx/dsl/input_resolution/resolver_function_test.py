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
from tfx.dsl.input_resolution import resolver_function
from tfx.dsl.input_resolution import resolver_op


class Foo(resolver_op.ResolverOp):
  foo = resolver_op.Property(type=int)

  def apply(self, input_dict):
    return input_dict


class Bar(resolver_op.ResolverOp):
  bar = resolver_op.Property(type=str, default='bar')

  def apply(self, input_dict):
    return input_dict


DUMMY_INPUT_NODE = resolver_op.InputNode(
    None, resolver_op.DataType.ARTIFACT_MULTIMAP)


class ResolverFunctionTest(tf.test.TestCase):

  def testTrace(self):

    def resolve(input_dict):
      result = Foo(input_dict, foo=1)
      result = Bar(result, bar='x')
      return result

    rf = resolver_function.ResolverFunction(resolve)
    output_node = rf.trace(DUMMY_INPUT_NODE)
    self.assertEqual(repr(output_node),
                     "Bar(Foo(Input(), foo=1), bar='x')")

  def testTrace_BadReturnValue(self):

    def resolve(unused_input_dict):
      return {'examples': []}

    rf = resolver_function.ResolverFunction(resolve)
    with self.assertRaises(TypeError):
      rf.trace(DUMMY_INPUT_NODE)


if __name__ == '__main__':
  tf.test.main()
