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
"""Tests for tfx.dsl.input_resolution.resolver_op."""
import copy
from typing import Optional, Mapping

import tensorflow as tf

from tfx.dsl.input_resolution import resolver_op


class Foo(resolver_op.ResolverOp):
  foo = resolver_op.ResolverOpProperty(type=int)

  def apply(self, input_dict):
    return input_dict


class Bar(resolver_op.ResolverOp):
  bar = resolver_op.ResolverOpProperty(type=str, default='bar')

  def apply(self, input_dict):
    return input_dict


class Repeat(
    resolver_op.ResolverOp,
    return_data_type=resolver_op.DataTypes.ARTIFACT_MULTIMAP_LIST):
  n = resolver_op.ResolverOpProperty(type=int)

  def apply(self, input_dict):
    return [copy.deepcopy(input_dict) for _ in range(self.n)]


class TakeLast(
    resolver_op.ResolverOp,
    arg_data_types=(resolver_op.DataTypes.ARTIFACT_MULTIMAP_LIST,)):

  def apply(self, input_dicts):
    return input_dicts[-1]


class ResolverOpTest(tf.test.TestCase):

  def testDefineOp_PropertyDefaultViolatesType(self):
    with self.assertRaises(TypeError):
      class BadProperty(resolver_op.ResolverOp):  # pylint: disable=unused-variable
        str_prop = resolver_op.ResolverOpProperty(type=str, default=42)

  def testOpCall_ReturnsOpNode(self):
    result = Foo(resolver_op.OpNode.INPUT_NODE, foo=42)
    self.assertIsInstance(result, resolver_op.OpNode)
    self.assertEqual(repr(result), 'Foo(INPUT_NODE, foo=42)')

  def testOpCall_MissingRequiredProperty(self):
    with self.assertRaisesRegex(
        ValueError, 'Required property foo is missing.'):
      Foo(resolver_op.OpNode.INPUT_NODE)

  def testOpCall_UnknownProperty(self):
    with self.assertRaisesRegex(
        KeyError, 'Unknown property bar.'):
      Foo(resolver_op.OpNode.INPUT_NODE, foo=42, bar='zz')

  def testOpCall_PropertyTypeCheck(self):
    with self.assertRaisesRegex(
        TypeError, "foo should be <class 'int'> but got '42'."):
      Foo(resolver_op.OpNode.INPUT_NODE, foo='42')

  def testOpCreate_ReturnsOp(self):
    result = Foo.create(foo=42)
    self.assertIsInstance(result, Foo)
    self.assertEqual(result.foo, 42)

  def testOpCreate_CannotTakeMultipleArgs(self):
    foo1 = Foo(resolver_op.OpNode.INPUT_NODE, foo=1)
    foo2 = Foo(resolver_op.OpNode.INPUT_NODE, foo=2)
    with self.assertRaises(TypeError):
      Bar(foo1, foo2)

  def testOpCreate_InvalidArg(self):
    with self.assertRaisesRegex(
        ValueError, 'Cannot directly call ResolverOp with real values.'):
      Bar({'foo': []})

  def testOpCreate_MissingRequiredProperty(self):
    with self.assertRaisesRegex(
        ValueError, 'Required property foo is missing.'):
      Foo.create()

  def testOpCreate_UnknownProperty(self):
    with self.assertRaisesRegex(
        KeyError, 'Unknown property bar.'):
      Foo.create(foo=42, bar='zz')

  def testOpCreate_PropertyTypeCheck(self):
    with self.assertRaisesRegex(
        TypeError, "foo should be <class 'int'> but got '42'."):
      Foo.create(foo='42')

  def testOpCreate_ArgumentTypeCheck(self):
    input_node = resolver_op.OpNode.INPUT_NODE

    with self.subTest('Need List[Dict] but got Dict.'):
      with self.assertRaisesRegex(
          TypeError, 'TakeLast takes ARTIFACT_MULTIMAP_LIST type but got '
          'ARTIFACT_MULTIMAP instead.'):
        TakeLast(input_node)

    with self.subTest('No Error'):
      TakeLast(Repeat(input_node, n=2))

  def testOpProperty_DefaultValue(self):
    result = Bar.create()
    self.assertEqual(result.bar, 'bar')

  def testOpProperty_TypeCheckOnSet(self):
    foo = Foo.create(foo=42)
    with self.assertRaises(TypeError):
      foo.foo = '42'

  def testOpProperty_ComplexTypeCheck(self):

    class CustomOp(resolver_op.ResolverOp):
      optional_mapping = resolver_op.ResolverOpProperty(
          type=Optional[Mapping[str, str]], default=None)

      def apply(self, input_dict):
        return input_dict

    # No errors for Optional[Mapping]
    CustomOp.create()
    CustomOp.create(optional_mapping=None)
    CustomOp.create(optional_mapping={'hello': 'world'})
    # Errors for non Optional[Mapping]
    with self.assertRaises(TypeError):
      CustomOp.create(optional_mapping='meh')
    with self.assertRaises(TypeError):
      CustomOp.create(optional_mapping=[])


class OpNodeTest(tf.test.TestCase):

  def testOpNode_Repr(self):
    input_node = resolver_op.OpNode.INPUT_NODE
    foo = resolver_op.OpNode(
        op_type=Foo, arg=input_node, kwargs={'foo': 42})
    bar = resolver_op.OpNode(
        op_type=Bar, arg=foo, kwargs={'bar': 'z'})

    self.assertEqual(repr(bar), "Bar(Foo(INPUT_NODE, foo=42), bar='z')")

  def testOpNode_OpTypeMustBeResolverOpSubclass(self):
    class NotResolverOp:
      pass

    with self.assertRaises(TypeError):
      resolver_op.OpNode(op_type=NotResolverOp)

    with self.assertRaises(TypeError):
      resolver_op.OpNode(op_type=Foo.create(foo=42))  # Not an instance!

  def testOpNode_ArgsMustBeOpNodeSequence(self):
    node = resolver_op.OpNode.INPUT_NODE

    with self.assertRaises(TypeError):
      resolver_op.OpNode(op_type=Foo, args=node)  # Not a Sequence!

    with self.assertRaises(TypeError):
      resolver_op.OpNode(op_type=Foo, args=[42])  # Element is not OpNode!

  def testOpNode_IsInputNode(self):
    input_node = resolver_op.OpNode.INPUT_NODE
    foo = Foo(input_node, foo=42)

    self.assertTrue(input_node.is_input_node)
    self.assertFalse(foo.is_input_node)


if __name__ == '__main__':
  tf.test.main()
