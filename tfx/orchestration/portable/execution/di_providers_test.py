# Copyright 2023 Google LLC. All Rights Reserved.
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
"""Tests for tfx.orchestration.portable.execution.di_providers."""

from collections.abc import Sequence
from typing import Any, Optional

import tensorflow as tf
from tfx.orchestration.portable import data_types
from tfx.orchestration.portable.execution import di_providers
from tfx.types import artifact as artifact_lib
from tfx.types import standard_artifacts
from tfx.utils.di import errors
from tfx.utils.di import module


class Foo(artifact_lib.Artifact):
  TYPE_NAME = 'Foo'

  def __init__(self, identifier: int = 1):
    super().__init__()
    self.id = identifier

  def __eq__(self, others):
    return isinstance(others, Foo) and self.id == others.id


class IAmAlsoFoo(artifact_lib.Artifact):
  TYPE_NAME = 'Foo'


class Bar(artifact_lib.Artifact):
  TYPE_NAME = 'Bar'


def _value_artifact(
    artifact_type: type[standard_artifacts.ValueArtifact], value: Any
):
  artifact = artifact_type()
  artifact.read = lambda: None  # Make artifact.read() to be a no-op.
  artifact.write = lambda _: None  # Make artifact.write() to be a no-op.
  artifact.value = value
  return artifact


class ProvidersTest(tf.test.TestCase):

  def testFlatExecutionInfoProvider_Input(self):
    m = module.DependencyModule()
    m.add_provider(di_providers.FlatExecutionInfoProvider(['foo']))
    m.provide_value(
        data_types.ExecutionInfo(
            input_dict={
                'foo': [Foo()],
            }
        )
    )

    result = m.get('foo', Foo)
    self.assertEqual(result, Foo())

  def testFlatExecutionInfoProvider_Output(self):
    m = module.DependencyModule()
    m.add_provider(di_providers.FlatExecutionInfoProvider(['foo']))
    m.provide_value(
        data_types.ExecutionInfo(
            output_dict={
                'foo': [Foo()],
            }
        )
    )

    result = m.get('foo', Foo)
    self.assertEqual(result, Foo())

  def testFlatExecutionInfoProvider_ExecProperty(self):
    m = module.DependencyModule()
    m.add_provider(
        di_providers.FlatExecutionInfoProvider(
            ['my_int', 'my_string', 'my_list']
        )
    )
    m.provide_value(
        data_types.ExecutionInfo(
            exec_properties={
                'my_int': 1,
                'my_string': 'hello',
                'my_list': [1, 2, 3],
            }
        )
    )

    self.assertEqual(m.get('my_int', int), 1)
    self.assertEqual(m.get('my_string', str), 'hello')
    self.assertEqual(m.get('my_list', list[int]), [1, 2, 3])

  def testFlatExecutionInfoProvider_ArtifactCompositeType(self):
    m = module.DependencyModule()
    m.add_provider(
        di_providers.FlatExecutionInfoProvider(['zero', 'one', 'two'])
    )
    m.provide_value(
        data_types.ExecutionInfo(
            input_dict={
                'zero': [],
                'one': [Foo(1)],
                'two': [Foo(1), Foo(2)],
            }
        )
    )

    with self.subTest('Artifact type can change as long as TYPE_NAME is same.'):
      self.assertIsInstance(m.get('one', Foo), Foo)
      self.assertIsInstance(m.get('one', IAmAlsoFoo), IAmAlsoFoo)
      with self.assertRaises(errors.InvalidTypeHintError):
        m.get('one', Bar)

    with self.subTest('Optional[T]'):
      self.assertIsNone(m.get('zero', Optional[Foo]))
      self.assertEqual(m.get('one', Optional[Foo]), Foo())
      with self.assertRaises(errors.InvalidTypeHintError):
        m.get('two', Optional[Foo])

    with self.subTest('list[T] or Sequence[T]'):
      self.assertLen(m.get('zero', list[Foo]), 0)  # pylint: disable=g-generic-assert
      self.assertLen(m.get('one', list[Foo]), 1)
      self.assertLen(m.get('two', list[Foo]), 2)
      self.assertLen(m.get('two', Sequence[Foo]), 2)

    with self.subTest('No type_hint'):
      self.assertEqual(m.get('zero', None), [])
      self.assertEqual(m.get('one', None), [Foo(1)])
      self.assertEqual(m.get('two', None), [Foo(1), Foo(2)])

  def testFlatExecutionInfoProvider_ValueArtifactPrimitiveType(self):
    m = module.DependencyModule()
    m.add_provider(
        di_providers.FlatExecutionInfoProvider([
            'empty',
            'int',
            'float',
            'str',
            'bytes',
            'bool',
            'jsonable',
            'many',
            'out',
        ])
    )
    m.provide_value(
        data_types.ExecutionInfo(
            input_dict={
                'empty': [],
                'int': [_value_artifact(standard_artifacts.Integer, 1)],
                'float': [_value_artifact(standard_artifacts.Float, 2.5)],
                'str': [_value_artifact(standard_artifacts.String, 'hello')],
                'bytes': [_value_artifact(standard_artifacts.Bytes, b'world')],
                'bool': [_value_artifact(standard_artifacts.Boolean, True)],
                'jsonable': [
                    _value_artifact(
                        standard_artifacts.JsonValue, {'hello': 'world'}
                    )
                ],
                'many': [
                    _value_artifact(standard_artifacts.Integer, 1),
                    _value_artifact(standard_artifacts.Integer, 2),
                ],
            },
            output_dict={
                'out': [_value_artifact(standard_artifacts.Integer, 3)],
            },
        )
    )

    with self.subTest('From input dict'):
      self.assertEqual(m.get('int', int), 1)
      self.assertEqual(m.get('float', float), 2.5)
      self.assertEqual(m.get('str', str), 'hello')
      self.assertEqual(m.get('bytes', bytes), b'world')
      self.assertEqual(m.get('bool', bool), True)
      self.assertDictEqual(
          m.get('jsonable', dict[str, str]), {'hello': 'world'}
      )

    with self.subTest('Optional input dict'):
      self.assertEqual(m.get('int', Optional[int]), 1)
      self.assertEqual(m.get('float', Optional[float]), 2.5)
      self.assertEqual(m.get('str', Optional[str]), 'hello')
      self.assertEqual(m.get('bytes', Optional[bytes]), b'world')
      self.assertEqual(m.get('bool', Optional[bool]), True)
      self.assertIsNone(m.get('empty', Optional[int]))

    with self.subTest('Unsupported primitive type'):
      with self.assertRaises(errors.InvalidTypeHintError):
        m.get('empty', int)
      with self.assertRaises(errors.InvalidTypeHintError):
        m.get('many', int)

    with self.subTest('Unsupported output dict'):
      with self.assertRaises(errors.InvalidTypeHintError):
        m.get('out', int)

  def testFlatExecutionInfoProvider_ExecProperty_StrictTypeCheck(self):
    m = module.DependencyModule()
    m.add_provider(
        di_providers.FlatExecutionInfoProvider(
            ['my_int', 'my_string', 'my_list'], strict=True
        )
    )
    m.provide_value(
        data_types.ExecutionInfo(
            exec_properties={
                'my_int': 1,
                'my_string': 'hello',
                'my_list': [1, 2, 3],
            }
        )
    )

    self.assertEqual(m.get('my_int', int), 1)
    with self.assertRaises(errors.InvalidTypeHintError):
      m.get('my_int', str)

    self.assertEqual(m.get('my_string', str), 'hello')
    with self.assertRaises(errors.InvalidTypeHintError):
      m.get('my_string', int)

    self.assertEqual(m.get('my_list', list[int]), [1, 2, 3])
    with self.assertRaises(errors.InvalidTypeHintError):
      m.get('my_list', list[str])
