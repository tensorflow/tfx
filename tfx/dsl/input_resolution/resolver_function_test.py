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

from typing import Mapping

import tensorflow as tf
from tfx.dsl.input_resolution import resolver_function
from tfx.dsl.input_resolution import resolver_op
import tfx.types
from tfx.types import resolved_channel
from tfx.utils import typing_utils


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


class DummyChannel(tfx.types.BaseChannel):

  def __eq__(self, other):
    return isinstance(other, DummyChannel) and self.type == other.type


class DummyNode(resolver_op.Node):

  def __init__(self, output_data_type):
    self.output_data_type = output_data_type


class X(tfx.types.Artifact):
  TYPE_NAME = 'X'


class Y(tfx.types.Artifact):
  TYPE_NAME = 'Y'


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
      return 'Not a node'

    rf = resolver_function.ResolverFunction(resolve)
    with self.assertRaises(RuntimeError):
      rf.trace(DUMMY_INPUT_NODE)

  def testCall_WithStaticTypeHint(self):

    @resolver_function.resolver_function(type_hint={'x': X, 'y': Y})
    def resolve(input_dict):
      return input_dict

    output = resolve({})

    self.assertTrue(
        typing_utils.is_compatible(output, Mapping[str, tfx.types.BaseChannel]))
    self.assertEqual(output['x'].type, X)
    self.assertEqual(output['y'].type, Y)

  def testCall_WithDynamicTypeHint(self):

    @resolver_function.resolver_function
    def resolve(input_dict):
      return input_dict

    output = resolve.with_type_hint({'x': X, 'y': Y})({})

    self.assertTrue(
        typing_utils.is_compatible(output, Mapping[str, tfx.types.BaseChannel]))
    self.assertEqual(output['x'].type, X)
    self.assertEqual(output['y'].type, Y)

    with self.subTest('Invalid type_hint'):
      with self.assertRaisesRegex(ValueError, 'Invalid type_hint'):
        resolve.with_type_hint('i am not a type hint')

  def testCall_WithNoTypeHint(self):

    @resolver_function.resolver_function
    def resolve(*args, **kwargs):
      del args, kwargs
      return DummyNode(resolver_op.DataType.ARTIFACT_MULTIMAP)

    with self.subTest(
        'Infer type hint if len(args) = 1 and args[0] is Mapping[str, '
        'BaseChannel].'):
      output = resolve({
          'x': DummyChannel(X),
          'y': DummyChannel(Y),
      })
      self.assertTrue(
          typing_utils.is_compatible(
              output, Mapping[str, tfx.types.BaseChannel]))
      self.assertEqual(output['x'].type, X)
      self.assertEqual(output['y'].type, Y)

    with self.subTest('Cannot infer other kind of args'):
      with self.assertRaisesRegex(RuntimeError, 'type_hint not set'):
        output = resolve(x=DummyChannel(X), y=DummyChannel(Y))

  def testCall_TypeHintCompatibility(self):

    @resolver_function.resolver_function
    def resolve():
      return DummyNode(resolver_op.DataType.ARTIFACT_MULTIMAP)

    def okay(type_hint):
      with self.subTest('Should Pass', type_hint=type_hint):
        resolve.with_type_hint(type_hint)

    def fail(type_hint):
      with self.subTest('Should Fail', type_hint=type_hint):
        with self.assertRaisesRegex(ValueError, 'Invalid type_hint'):
          resolve.with_type_hint(type_hint)

    # Type[Artifact] is okay
    okay(tfx.types.Artifact)
    okay(X)
    fail(X())

    # Dict[str, Type[Artifact]] is okay
    okay({})
    okay({'x': X})
    fail({'x': X()})
    fail({0: X})
    fail({'x': 'x'})

    # Other types fail
    fail(1)
    fail([])
    fail([X, Y])
    fail('not a type hint')

  def testCall_ArgConversion(self):
    holder = []

    @resolver_function.resolver_function(type_hint={'x': X})
    def resolve(*args, **kwargs):
      holder.append((args, kwargs))
      return DummyNode(resolver_op.DataType.ARTIFACT_MULTIMAP)

    resolve(
        {'x': DummyChannel(X)},  # args[0],
        'hello',  # args[1]
        DummyChannel(X),  # args[2]
        x='x',  # kwargs['x']
        y=DummyChannel(Y),  # kwargs['y']
    )

    args, kwargs = holder[0]

    with self.subTest('args[0]', value=args[0]):
      self.assertIsInstance(args[0], resolver_op.InputNode)
      self.assertEqual(args[0].wrapped, {'x': DummyChannel(X)})
      self.assertEqual(
          args[0].output_data_type, resolver_op.DataType.ARTIFACT_MULTIMAP)

    with self.subTest('args[1]', value=args[1]):
      self.assertIsInstance(args[1], str)
      self.assertEqual(args[1], 'hello')

    with self.subTest('args[2]', value=args[2]):
      self.assertIsInstance(args[2], resolver_op.InputNode)
      self.assertEqual(args[2].wrapped, DummyChannel(X))
      self.assertEqual(
          args[2].output_data_type, resolver_op.DataType.ARTIFACT_LIST)

    with self.subTest('kwargs[x]', value=kwargs['x']):
      self.assertEqual(kwargs['x'], 'x')

    with self.subTest('kwargs[y]', value=kwargs['y']):
      self.assertIsInstance(kwargs['y'], resolver_op.InputNode)
      self.assertEqual(kwargs['y'].wrapped, DummyChannel(Y))

  def testCall_ResultConversion(self):

    @resolver_function.resolver_function(type_hint={'x': X})
    def resolve():
      return {
          'x': DummyNode(resolver_op.DataType.ARTIFACT_LIST)
      }

    output = resolve()
    self.assertIsInstance(output, dict)
    self.assertIsInstance(output['x'], resolved_channel.ResolvedChannel)
    self.assertIsInstance(output['x'].output_node, resolver_op.DictNode)
    self.assertEqual(output['x'].output_key, 'x')

  def testCall_TypeHintAndOutputDataTypeMatch(self):

    @resolver_function.resolver_function
    def resolve_artifact_list():
      return DummyNode(resolver_op.DataType.ARTIFACT_LIST)

    @resolver_function.resolver_function
    def resolve_artifact_multimap():
      return DummyNode(resolver_op.DataType.ARTIFACT_MULTIMAP)

    @resolver_function.resolver_function
    def resolve_artifact_multimap_list():
      return DummyNode(resolver_op.DataType.ARTIFACT_MULTIMAP_LIST)

    with self.subTest('ARTIFACT_LIST with dict type hint'):
      with self.assertRaises(RuntimeError):
        resolve_artifact_list.with_type_hint({'x': X})()

    with self.subTest('ARTIFACT_LIST with a single type hint'):
      result = resolve_artifact_list.with_type_hint(X)()
      self.assertIsInstance(result, resolved_channel.ResolvedChannel)
      self.assertEqual(result.type, X)

    with self.subTest('ARTIFACT_MULTIMAP with dict type hint'):
      result = resolve_artifact_multimap.with_type_hint({'x': X})()
      self.assertIsInstance(result, dict)
      self.assertIsInstance(result['x'], resolved_channel.ResolvedChannel)
      self.assertEqual(result['x'].type, X)

    with self.subTest('ARTIFACT_MULTIMAP with a single type hint'):
      with self.assertRaises(RuntimeError):
        resolve_artifact_multimap.with_type_hint(X)()

    with self.subTest('ARTIFACT_MULTIMAP_LIST with dict type hint'):
      with self.assertRaises(NotImplementedError):
        resolve_artifact_multimap_list.with_type_hint({'x': X})()

    with self.subTest('ARTIFACT_MULTIMAP_LIST with a single type hint'):
      with self.assertRaises(RuntimeError):
        resolve_artifact_multimap_list.with_type_hint(X)()

  def testGetInputNodes_And_GetDependentChannels(self):
    x = DummyChannel(X)
    y1 = DummyChannel(Y)
    y2 = DummyChannel(Y)
    input_x = resolver_op.InputNode(x, resolver_op.DataType.ARTIFACT_LIST)
    input_y = resolver_op.InputNode(y1, resolver_op.DataType.ARTIFACT_LIST)
    input_xy = resolver_op.InputNode(
        {'x': x, 'y': y2}, resolver_op.DataType.ARTIFACT_MULTIMAP)

    x_plus_y = resolver_op.OpNode(
        op_type='add',
        output_data_type=resolver_op.DataType.ARTIFACT_LIST,
        args=(input_x, input_y))
    z = resolver_op.DictNode({'z': x_plus_y})
    result = resolver_op.OpNode(
        op_type='merge',
        output_data_type=resolver_op.DataType.ARTIFACT_MULTIMAP,
        args=(input_xy, z))

    self.assertCountEqual(
        resolver_function.get_input_nodes(result),
        [input_x, input_y, input_xy])
    self.assertCountEqual(
        resolver_function.get_dependent_channels(result),
        [x, y1, y2])


if __name__ == '__main__':
  tf.test.main()
