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
"""Tests for tfx.dsl.context_managers.for_each."""

import tensorflow as tf
from tfx import types
from tfx.dsl.components.base import base_node
from tfx.dsl.context_managers import context_manager
from tfx.dsl.context_managers import for_each


class AA(types.Artifact):
  TYPE_NAME = 'AA'


class BB(types.Artifact):
  TYPE_NAME = 'BB'


class FakeNode(base_node.BaseNode):

  def __init__(self, **input_dict):
    self._input_dict = input_dict
    self._output_dict = {
        'aa': types.Channel(type=AA),
        'bb': types.Channel(type=BB),
    }
    self._output_dict_patched = False
    super().__init__()

  @property
  def inputs(self):
    return self._input_dict

  @property
  def outputs(self):
    if not self._output_dict_patched:
      for output_key, channel in self._output_dict.items():
        channel.producer_component_id = self.id
        channel.output_key = output_key
      self._output_dict_patched = True
    return self._output_dict

  @property
  def exec_properties(self):
    return {}


class A(FakeNode):
  pass


class B(FakeNode):
  pass


class C(FakeNode):
  pass


class ForEachTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.reset_registry()

  def reset_registry(self) -> context_manager._DslContextRegistry:
    result = context_manager._registry = context_manager._DslContextRegistry()
    return result

  def testForEach_As_GivesLoopVariable(self):
    a = A().with_id('Apple')
    with for_each.ForEach(a.outputs['aa']) as aa:
      pass

    self.assertIsInstance(aa, for_each.SlicedChannel)
    self.assertEqual(aa.context_id, 'ForEachContext:1')
    self.assertEqual(aa.type, AA)
    self.assertEqual(aa.producer_component_id, 'Apple')
    self.assertEqual(aa.output_key, 'aa')

  def testForEach_LoopVariableNotUsed_Disallowed(self):
    with self.subTest('Not using at all'):
      with self.assertRaises(ValueError):
        a = A()
        with for_each.ForEach(a.outputs['aa']) as aa:  # pylint: disable=unused-variable
          b = B()  # aa is not used. # pylint: disable=unused-variable

    with self.subTest('Source channel is not a loop variable.'):
      with self.assertRaises(ValueError):
        a = A()
        with for_each.ForEach(a.outputs['aa']) as aa:
          b = B(aa=a.outputs['aa'])  # Should use loop var "aa" directly.

  def testForEach_UsingOriginalChannel_Disallowed(self):
    with self.assertRaises(ValueError):
      a = A()
      with for_each.ForEach(a.outputs['aa']) as aa:  # pylint: disable=unused-variable
        b = B(aa=a.outputs['aa'])  # != aa. # pylint: disable=unused-variable

  def testForEach_MultipleNodes_NotImplemented(self):
    with self.assertRaises(NotImplementedError):
      a = A()
      with for_each.ForEach(a.outputs['aa']) as aa:
        b = B(aa=aa)
        c = C(bb=b.outputs['bb'])  # pylint: disable=unused-variable

  def testForEach_NestedForEach_NotImplemented(self):
    with self.assertRaises(NotImplementedError):
      a = A()
      b = B()
      with for_each.ForEach(a.outputs['aa']) as aa:
        with for_each.ForEach(b.outputs['bb']) as bb:
          c = C(aa=aa, bb=bb)  # pylint: disable=unused-variable

  def testForEach_DifferentLoop_HasDifferentContext(self):
    a = A()
    with for_each.ForEach(a.outputs['aa']) as aa1:
      b1 = B(aa=aa1)
    with for_each.ForEach(a.outputs['aa']) as aa2:
      b2 = B(aa=aa2)

    c1 = context_manager.get_contexts(b1)[-1]
    c2 = context_manager.get_contexts(b2)[-1]
    self.assertNotEqual(c1.id, c2.id)


if __name__ == '__main__':
  tf.test.main()
