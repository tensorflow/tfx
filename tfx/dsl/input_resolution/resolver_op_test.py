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
import tfx.types
from tfx.types import standard_artifacts


class Foo(resolver_op.ResolverOp):
  foo = resolver_op.Property(type=int)

  def apply(self, input_dict):
    return input_dict


class Bar(resolver_op.ResolverOp):
  bar = resolver_op.Property(type=str, default='bar')

  def apply(self, input_dict):
    return input_dict


class EmptyArtifactList(
    resolver_op.ResolverOp,
    canonical_name='testing.Baz',
    arg_data_types=(),
    return_data_type=resolver_op.DataType.ARTIFACT_LIST):

  def apply(self):
    return []


class Repeat(
    resolver_op.ResolverOp,
    return_data_type=resolver_op.DataType.ARTIFACT_MULTIMAP_LIST):
  n = resolver_op.Property(type=int)

  def apply(self, input_dict):
    return [copy.deepcopy(input_dict) for _ in range(self.n)]


class TakeLast(
    resolver_op.ResolverOp,
    arg_data_types=(resolver_op.DataType.ARTIFACT_MULTIMAP_LIST,)):

  def apply(self, input_dicts):
    return input_dicts[-1]


class ManyArtifacts(
    resolver_op.ResolverOp,
    arg_data_types=(),
    return_data_type=resolver_op.DataType.ARTIFACT_LIST):

  def apply(self):
    return []


DUMMY_INPUT_NODE = resolver_op.InputNode(
    None, resolver_op.DataType.ARTIFACT_MULTIMAP)


class ResolverOpTest(tf.test.TestCase):

  def testDefineOp_PropertyDefaultViolatesType(self):
    with self.assertRaises(TypeError):
      class BadProperty(resolver_op.ResolverOp):  # pylint: disable=unused-variable
        str_prop = resolver_op.Property(type=str, default=42)

  def testOpCall_ReturnsOpNode(self):
    result = Foo(DUMMY_INPUT_NODE, foo=42)
    self.assertIsInstance(result, resolver_op.OpNode)
    self.assertEqual(repr(result), 'Foo(Input(), foo=42)')

  def testOpCall_MissingRequiredProperty(self):
    with self.assertRaisesRegex(
        ValueError, 'Required property foo is missing.'):
      Foo(DUMMY_INPUT_NODE)

  def testOpCall_UnknownProperty(self):
    with self.assertRaisesRegex(
        KeyError, 'Unknown property bar.'):
      Foo(DUMMY_INPUT_NODE, foo=42, bar='zz')

  def testOpCall_PropertyTypeCheck(self):
    with self.assertRaisesRegex(
        TypeError, "foo should be <class 'int'> but got '42'."):
      Foo(DUMMY_INPUT_NODE, foo='42')

  def testOpCreate_ReturnsOp(self):
    result = Foo.create(foo=42)
    self.assertIsInstance(result, Foo)
    self.assertEqual(result.foo, 42)

  def testOpCreate_CannotTakeMultipleArgs(self):
    foo1 = Foo(DUMMY_INPUT_NODE, foo=1)
    foo2 = Foo(DUMMY_INPUT_NODE, foo=2)
    with self.assertRaises(ValueError):
      Bar(foo1, foo2)

  def testOpCreate_InvalidArg(self):
    with self.assertRaisesRegex(
        ValueError, r'Expected dict\[str, Node\]'):
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
    input_node = DUMMY_INPUT_NODE

    with self.subTest('Need List[Dict] but got Dict.'):
      with self.assertRaisesRegex(
          TypeError, 'TakeLast takes ARTIFACT_MULTIMAP_LIST type but got '
          'ARTIFACT_MULTIMAP instead.'):
        TakeLast(input_node)

    with self.subTest('No Error'):
      TakeLast(Repeat(input_node, n=2))

  def testOpCreate_DictArg_ConvertedToDictNode(self):
    result = Bar({'foo': ManyArtifacts()})
    self.assertEqual(repr(result), 'Bar(Dict(foo=ManyArtifacts()))')

  def testOpProperty_DefaultValue(self):
    result = Bar.create()
    self.assertEqual(result.bar, 'bar')

  def testOpProperty_TypeCheckOnSet(self):
    foo = Foo.create(foo=42)
    with self.assertRaises(TypeError):
      foo.foo = '42'

  def testOpProperty_ComplexTypeCheck(self):

    class CustomOp(resolver_op.ResolverOp):
      optional_mapping = resolver_op.Property(
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


class NodeTest(tf.test.TestCase):

  def testOpNode_Repr(self):
    input_node = DUMMY_INPUT_NODE
    foo = resolver_op.OpNode(
        op_type=Foo, args=[input_node], kwargs={'foo': 42})
    bar = resolver_op.OpNode(
        op_type=Bar, args=[foo], kwargs={'bar': 'z'})

    self.assertEqual(repr(bar), "Bar(Foo(Input(), foo=42), bar='z')")

  def testOpNode_ArgsMustBeOpNodeSequence(self):
    node = DUMMY_INPUT_NODE

    with self.assertRaises(TypeError):
      resolver_op.OpNode(op_type=Foo, args=node)  # Not a Sequence!

    with self.assertRaises(TypeError):
      resolver_op.OpNode(op_type=Foo, args=[42])  # Element is not OpNode!

  def testEqualityAndHash(self):
    with self.subTest('Node'):
      n1 = resolver_op.Node()
      n2 = resolver_op.Node()
      self.assertNotEqual(n1, n2)

    with self.subTest('OpNode'):
      n3 = resolver_op.OpNode(
          op_type=EmptyArtifactList,
          output_data_type=resolver_op.DataType.ARTIFACT_LIST,
          args=())
      n4 = resolver_op.OpNode(
          op_type=EmptyArtifactList,
          output_data_type=resolver_op.DataType.ARTIFACT_LIST,
          args=())
      self.assertNotEqual(n3, n4)

    with self.subTest('DictNode'):
      n1.output_data_type = resolver_op.DataType.ARTIFACT_LIST
      n2.output_data_type = resolver_op.DataType.ARTIFACT_LIST
      n5 = resolver_op.DictNode({'x': n1, 'y': n3})
      n6 = resolver_op.DictNode({'x': n1, 'y': n3})
      n7 = resolver_op.DictNode({'x': n2, 'y': n4})
      self.assertEqual(n5, n6)
      self.assertNotEqual(n5, n7)
      self.assertLen({n5, n6, n7}, 2)

    class DummyChannel(tfx.types.BaseChannel):

      def __hash__(self):
        return hash(self.type)

      def __eq__(self, other):
        return isinstance(other, DummyChannel) and self.type == other.type

    with self.subTest('InputNode'):
      n8 = resolver_op.InputNode(
          wrapped=DummyChannel(standard_artifacts.Model),
          output_data_type=resolver_op.DataType.ARTIFACT_LIST)
      n9 = resolver_op.InputNode(
          wrapped=DummyChannel(standard_artifacts.Model),
          output_data_type=resolver_op.DataType.ARTIFACT_LIST)
      n10 = resolver_op.InputNode(
          wrapped=DummyChannel(standard_artifacts.Examples),
          output_data_type=resolver_op.DataType.ARTIFACT_LIST)
      self.assertEqual(n8, n9)
      self.assertNotEqual(n8, n10)
      self.assertLen({n8, n9, n10}, 2)

if __name__ == '__main__':
  tf.test.main()
