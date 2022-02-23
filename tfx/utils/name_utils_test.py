# Copyright 2022 Google LLC. All Rights Reserved.
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
"""Tests for tfx.utils.name_utils."""

import types

import tensorflow as tf

from tfx.utils import name_utils


class Foo:
  class Bar:
    pass


def fun():
  pass

VALUE = 42


class ClassUtilsTest(tf.test.TestCase):

  def testGetFullName_GoodExamples(self):
    self.assertEqual(name_utils.get_full_name(str), 'builtins.str')
    self.assertEqual(name_utils.get_full_name(Foo), f'{__name__}.Foo')
    self.assertEqual(name_utils.get_full_name(Foo.Bar), f'{__name__}.Foo.Bar')
    self.assertEqual(name_utils.get_full_name(fun), f'{__name__}.fun')

  def testGetFullName_BadExamples(self):
    with self.assertRaisesRegex(ValueError, 'does not have a name'):
      name_utils.get_full_name(VALUE)

    with self.assertRaisesRegex(ValueError, 'does not have a qualified name'):
      class DynamicClass:
        pass
      name_utils.get_full_name(DynamicClass)

    with self.assertRaisesRegex(ValueError, 'is not importable'):
      dynamic_class = types.new_class('DynamicClass')
      name_utils.get_full_name(dynamic_class)

  def testGetClass_GoodExamples(self):
    self.assertIs(name_utils.resolve_full_name('builtins.str'), str)
    self.assertIs(name_utils.resolve_full_name(f'{__name__}.Foo'), Foo)
    self.assertIs(name_utils.resolve_full_name(f'{__name__}.Foo.Bar'), Foo.Bar)
    self.assertIs(name_utils.resolve_full_name(f'{__name__}.fun'), fun)

  def testGetClass_BadExamples(self):
    with self.assertRaisesRegex(ValueError, 'not a valid name.'):
      name_utils.resolve_full_name(42)

    with self.assertRaisesRegex(ValueError, 'not a valid name.'):
      name_utils.resolve_full_name('foo^ax.1234')

    with self.assertRaisesRegex(ValueError, 'Cannot find'):
      name_utils.resolve_full_name('non_existing_module_name.meh.FakeClass')


if __name__ == '__main__':
  tf.test.main()
