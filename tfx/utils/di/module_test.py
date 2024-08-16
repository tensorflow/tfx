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
"""Tests for tfx.utils.di.module."""

import dataclasses
from typing import Optional, Union

import tensorflow as tf
from tfx.utils.di import errors
from tfx.utils.di import module


ANY_NAME = 'this_is_not_a_name'


class ModuleTest(tf.test.TestCase):

  def testGet_Simple(self):
    mod = module.DependencyModule()
    mod.provide_value(42, name='x')
    self.assertEqual(mod.get('x', int), 42)

  def testGet_MissingDependency(self):
    mod = module.DependencyModule()
    with self.assertRaises(errors.NotProvidedError):
      mod.get('x', int)

  def testGet_MissingDependency_OptionalIsNone(self):
    mod = module.DependencyModule()
    self.assertIsNone(mod.get('x', Optional[int]))

  def testInject_Simple(self):
    mod = module.DependencyModule()
    mod.provide_value(42, name='x')

    def add_one(x: int):
      return x + 1

    self.assertEqual(mod.call(add_one), 43)

  def testInject_MissingDependency(self):
    mod = module.DependencyModule()
    mod.provide_value(42, name='x')

    def add(x: int, y: int):
      return x + y

    with self.assertRaises(errors.NotProvidedError):
      mod.call(add)

  def testInject_MissingDependency_DefaultArgumentUsed(self):
    mod = module.DependencyModule()
    mod.provide_value(42, name='x')

    def add(x: int, y: int = 1):
      return x + y

    self.assertEqual(mod.call(add), 43)

  def testProvideValue_TypeShouldMatch(self):
    mod = module.DependencyModule()
    mod.provide_value(42, name='x')
    self.assertEqual(mod.get('x', int), 42)
    self.assertEqual(mod.get('x', None), 42)  # When type_hint is not given.
    with self.assertRaises(errors.NotProvidedError):
      mod.get('x', float)

  def testProvideAnonymousValue_GetByType(self):

    @dataclasses.dataclass
    class Foo:
      x: int

    mod = module.DependencyModule()
    mod.provide_value(Foo(x=42))
    self.assertEqual(mod.get(ANY_NAME, Foo), Foo(x=42))

  def testProvideClass_AnyNameIsOk(self):
    class Foo:
      pass

    mod = module.DependencyModule()
    mod.provide_class(Foo)

    foo = mod.get(ANY_NAME, Foo)

    self.assertIsInstance(foo, Foo)

  def testProvideClass_InitArgumentInjected(self):
    class Foo:

      def __init__(self, x: int):
        self.x = x

    mod = module.DependencyModule()
    mod.provide_value(42, name='x')
    mod.provide_class(Foo)

    foo = mod.get(ANY_NAME, Foo)
    self.assertEqual(foo.x, 42)

  def testProvideClass_UnprovidedDependency(self):
    class Foo:

      def __init__(self, x: int):
        self.x = x

    mod = module.DependencyModule()
    mod.provide_class(Foo)

    with self.assertRaises(errors.NotProvidedError):
      mod.get(ANY_NAME, Foo)

  def testProvideClass_Subclass(self):
    class Foo:
      pass

    class SubFoo(Foo):
      pass

    mod = module.DependencyModule()
    mod.provide_class(SubFoo)

    self.assertIsInstance(mod.get(ANY_NAME, Foo), SubFoo)
    self.assertIsInstance(mod.get(ANY_NAME, SubFoo), SubFoo)

  def testProvideClass_NonSingleton(self):
    class Foo:
      pass

    mod = module.DependencyModule()
    mod.provide_class(Foo, singleton=False)
    self.assertIsNot(
        mod.get(ANY_NAME, Foo),
        mod.get(ANY_NAME, Foo),
    )

  def testProvideClass_Singleton(self):
    class Foo:
      pass

    mod = module.DependencyModule()
    mod.provide_class(Foo, singleton=True)
    self.assertIs(
        mod.get(ANY_NAME, Foo),
        mod.get(ANY_NAME, Foo),
    )

  def testProvideClass_CompositeType(self):
    class Foo:
      pass

    mod = module.DependencyModule()
    mod.provide_class(Foo)

    # self.assertIsInstance(mod.get(ANY_NAME, Foo | None), Foo)
    self.assertIsInstance(mod.get(ANY_NAME, Optional[Foo]), Foo)
    # self.assertIsInstance(mod.get(ANY_NAME, Foo | int), Foo)
    self.assertIsInstance(mod.get(ANY_NAME, Union[Foo, int]), Foo)

  def testProvideClass_Dataclass(self):
    @dataclasses.dataclass
    class Foo:
      x: int
      y: str

    mod = module.DependencyModule()
    mod.provide_class(Foo)
    mod.provide_value(42, name='x')
    mod.provide_value('foo', name='y')

    foo = mod.get(ANY_NAME, Foo)
    self.assertEqual(foo.x, 42)
    self.assertEqual(foo.y, 'foo')

  def testProvideNamedClass(self):
    class Foo:
      pass

    class SubFoo(Foo):
      pass

    mod = module.DependencyModule()
    mod.provide_named_class('foo', Foo)
    mod.provide_class(SubFoo)

    self.assertIsInstance(mod.get('foo', Foo), Foo)
    self.assertIsInstance(mod.get('bar', Foo), SubFoo)
    self.assertIsInstance(mod.get('bar', SubFoo), SubFoo)

  def testProvideNamedClass_NotFoundIfNameMismatch(self):
    class Foo:
      pass

    mod = module.DependencyModule()
    mod.provide_named_class('foo', Foo)
    with self.assertRaises(errors.NotProvidedError):
      mod.get('bar', Foo)

  def testProvideNamedClass_NonSingleton(self):
    class Foo:
      pass

    mod = module.DependencyModule()
    mod.provide_named_class('foo', Foo, singleton=False)
    self.assertIsNot(mod.get('foo', Foo), mod.get('foo', Foo))

  def testProvideNamedClass_Singleton(self):
    class Foo:
      pass

    mod = module.DependencyModule()
    mod.provide_named_class('foo', Foo, singleton=True)
    self.assertIs(mod.get('foo', Foo), mod.get('foo', Foo))
