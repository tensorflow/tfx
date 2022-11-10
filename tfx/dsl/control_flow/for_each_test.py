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
import unittest

import tensorflow as tf
from tfx import types
from tfx.dsl.components.base import base_node
from tfx.dsl.context_managers import dsl_context_registry
from tfx.dsl.control_flow import for_each
from tfx.utils import test_case_utils


class AA(types.Artifact):
  TYPE_NAME = 'AA'


class BB(types.Artifact):
  TYPE_NAME = 'BB'


class FakeNode(base_node.BaseNode):

  def __init__(self, **input_dict):
    self._input_dict = input_dict
    self._output_dict = {
        'aa': types.OutputChannel(AA, self, 'aa'),
        'bb': types.OutputChannel(BB, self, 'bb'),
    }
    self._output_dict_patched = False
    super().__init__()

  @property
  def inputs(self):
    return self._input_dict

  @property
  def outputs(self):
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


class ForEachTest(test_case_utils.TfxTest):

  def setUp(self):
    super().setUp()
    self.enter_context(dsl_context_registry.new_registry())

  def testForEach_As_GivesLoopVariable(self):
    reg = dsl_context_registry.get()

    a = A().with_id('Apple')
    with for_each.ForEach(a.outputs['aa']) as aa:
      f = reg.peek_context()

    self.assertIsInstance(aa, types.LoopVarChannel)
    self.assertEqual(aa.for_each_context, f)
    self.assertEqual(aa.type, AA)
    self.assertEqual(aa.wrapped.producer_component_id, 'Apple')
    self.assertEqual(aa.wrapped.output_key, 'aa')

  @unittest.skip('Not implemented.')
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
    b = B()
    with for_each.ForEach(a.outputs['aa']) as aa1:
      c1 = C(aa=aa1, bb=b.outputs['bb'])  # pylint: disable=unused-variable
    with for_each.ForEach(a.outputs['aa']) as aa2:
      c2 = C(aa=aa2, bb=b.outputs['bb'])  # pylint: disable=unused-variable

    context1 = dsl_context_registry.get().get_contexts(c1)[-1]
    context2 = dsl_context_registry.get().get_contexts(c2)[-1]
    self.assertNotEqual(context1, context2)


if __name__ == '__main__':
  tf.test.main()
