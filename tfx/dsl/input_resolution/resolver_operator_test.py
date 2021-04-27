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
"""Tests for tfx.dsl.input_resolution.resolver_operator."""
from typing import Optional, Mapping, Dict

import tensorflow as tf

from tfx.dsl.input_resolution import resolver_operator

ArtifactMultimap = resolver_operator.ArtifactMultimap


class Foo(resolver_operator.ResolverOp):
  foo = resolver_operator.ResolverOpProperty(type=int)

  def apply(self, input_dict: ArtifactMultimap) -> ArtifactMultimap:
    return input_dict


class Bar(resolver_operator.ResolverOp):
  bar = resolver_operator.ResolverOpProperty(type=str, default='bar')

  def apply(self, input_dict: ArtifactMultimap) -> ArtifactMultimap:
    return input_dict


class ResolverOpTest(tf.test.TestCase):

  def testDefineOp_PropertyDefaultViolatesType(self):
    with self.assertRaises(TypeError):
      class BadProperty(resolver_operator.ResolverOp):  # pylint: disable=unused-variable
        str_prop = resolver_operator.ResolverOpProperty(type=str, default=42)

  def testDefineOp_NoAnnotation(self):
    # pylint: disable=g-wrong-blank-lines
    # pylint: disable=unused-variable

    with self.assertRaises(TypeError):
      class NoArgTypeAnnotation(resolver_operator.ResolverOp):
        def apply(self, input_dict) -> ArtifactMultimap:
          return input_dict

    with self.assertRaises(TypeError):
      class NoReturnTypeAnnotation(resolver_operator.ResolverOp):
        def apply(self, input_dict: ArtifactMultimap):
          return input_dict

  def testDefineOp_InvalidSignature(self):
    # pylint: disable=g-wrong-blank-lines
    # pylint: disable=unused-variable

    with self.assertRaises(TypeError):
      class BadReturnType(resolver_operator.ResolverOp):
        def apply(self, input_dict: ArtifactMultimap) -> int:
          return 42

    with self.assertRaises(TypeError):
      class BadArgType(resolver_operator.ResolverOp):
        def apply(self, input_dict: Dict[str, int]) -> ArtifactMultimap:
          return input_dict

    with self.assertRaises(TypeError):
      class TwoArgs(resolver_operator.ResolverOp):
        def apply(self, input_dict: ArtifactMultimap,
                  another_dict: ArtifactMultimap) -> ArtifactMultimap:
          return input_dict

  def testOpCall_ReturnsOpNode(self):
    result = Foo(resolver_operator.OpNode.INPUT_NODE, foo=42)
    self.assertIsInstance(result, resolver_operator.OpNode)
    self.assertEqual(repr(result), 'Foo(INPUT_NODE, foo=42)')

  def testOpCall_MissingRequiredProperty(self):
    with self.assertRaisesRegex(
        ValueError, 'Required property foo is missing.'):
      Foo(resolver_operator.OpNode.INPUT_NODE)

  def testOpCall_UnknownProperty(self):
    with self.assertRaisesRegex(
        KeyError, 'Unknown property bar.'):
      Foo(resolver_operator.OpNode.INPUT_NODE, foo=42, bar='zz')

  def testOpCall_PropertyTypeCheck(self):
    with self.assertRaisesRegex(
        TypeError, "foo should be <class 'int'> but got '42'."):
      Foo(resolver_operator.OpNode.INPUT_NODE, foo='42')

  def testOpCreate_ReturnsOp(self):
    result = Foo.create(foo=42)
    self.assertIsInstance(result, Foo)
    self.assertEqual(result.foo, 42)

  def testOpCreate_CannotTakeMultipleArgs(self):
    foo1 = Foo(resolver_operator.OpNode.INPUT_NODE, foo=1)
    foo2 = Foo(resolver_operator.OpNode.INPUT_NODE, foo=2)
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

  def testOpProperty_DefaultValue(self):
    result = Bar.create()
    self.assertEqual(result.bar, 'bar')

  def testOpProperty_TypeCheckOnSet(self):
    foo = Foo.create(foo=42)
    with self.assertRaises(TypeError):
      foo.foo = '42'

  def testOpProperty_ComplexTypeCheck(self):

    class CustomOp(resolver_operator.ResolverOp):
      optional_mapping = resolver_operator.ResolverOpProperty(
          type=Optional[Mapping[str, str]], default=None)

      def apply(self, input_dict: ArtifactMultimap) -> ArtifactMultimap:
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
    input_node = resolver_operator.OpNode.INPUT_NODE
    foo = resolver_operator.OpNode(
        op_type=Foo, arg=input_node, kwargs={'foo': 42})
    bar = resolver_operator.OpNode(
        op_type=Bar, arg=foo, kwargs={'bar': 'z'})

    self.assertEqual(repr(bar), "Bar(Foo(INPUT_NODE, foo=42), bar='z')")

  def testOpNode_OpTypeMustBeResolverOpSubclass(self):
    class NotResolverOp:
      pass

    with self.assertRaises(TypeError):
      resolver_operator.OpNode(op_type=NotResolverOp)

    with self.assertRaises(TypeError):
      resolver_operator.OpNode(op_type=Foo.create(foo=42))  # Not an instance!

  def testOpNode_ArgsMustBeOpNodeSequence(self):
    node = resolver_operator.OpNode.INPUT_NODE

    with self.assertRaises(TypeError):
      resolver_operator.OpNode(op_type=Foo, args=node)  # Not a Sequence!

    with self.assertRaises(TypeError):
      resolver_operator.OpNode(op_type=Foo, args=[42])  # Element is not OpNode!

  def testOpNode_IsInputNode(self):
    input_node = resolver_operator.OpNode.INPUT_NODE
    foo = Foo(input_node, foo=42)

    self.assertTrue(input_node.is_input_node)
    self.assertFalse(foo.is_input_node)


if __name__ == '__main__':
  tf.test.main()
