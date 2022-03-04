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
"""Tests for tfx.dsl.context_manager."""

from typing import Dict, Any

import tensorflow as tf
from tfx.dsl.components.base import base_node
from tfx.dsl.context_managers import dsl_context
from tfx.dsl.context_managers import dsl_context_manager
from tfx.dsl.context_managers import dsl_context_registry
from tfx.utils import test_case_utils


class _FakeNode(base_node.BaseNode):

  @property
  def inputs(self) -> Dict[str, Any]:
    return {}

  @property
  def outputs(self) -> Dict[str, Any]:
    return {}

  @property
  def exec_properties(self) -> Dict[str, Any]:
    return {}


class _FakeContext(dsl_context.DslContext):
  pass


class _FakeContextManager(dsl_context_manager.DslContextManager):

  def create_context(self) -> _FakeContext:
    return _FakeContext()

  def enter(self, context: _FakeContext) -> _FakeContext:
    return context


class ContextManagerTest(test_case_utils.TfxTest):

  def setUp(self):
    super().setUp()
    self.reg = self.enter_context(dsl_context_registry.new_registry())

  def testContext_ParentAttrFactory(self):
    with _FakeContextManager() as c1:
      self.assertIsNone(c1.parent)
      with _FakeContextManager() as c2:
        self.assertIs(c2.parent, c1)
        with _FakeContextManager() as c3:
          self.assertEqual(c3.parent, c2)
      with _FakeContextManager() as c4:
        self.assertIs(c4.parent, c1)

  def testContext_Ancestors(self):
    with _FakeContextManager() as c1:
      self.assertEqual(list(c1.ancestors), [])
      with _FakeContextManager() as c2:
        self.assertEqual(list(c2.ancestors), [c1])
        with _FakeContextManager() as c3:
          self.assertEqual(list(c3.ancestors), [c1, c2])
      with _FakeContextManager() as c4:
        self.assertEqual(list(c4.ancestors), [c1])

  def testRegistry_AllContexts(self):
    reg = dsl_context_registry.get()

    with _FakeContextManager() as c1:
      self.assertEqual(reg.all_contexts, [c1])
      with _FakeContextManager() as c2:
        self.assertEqual(reg.all_contexts, [c1, c2])
        with _FakeContextManager() as c3:
          self.assertEqual(reg.all_contexts, [c1, c2, c3])
      with _FakeContextManager() as c4:
        self.assertEqual(reg.all_contexts, [c1, c2, c3, c4])

  def testRegistry_ActiveContexts(self):
    reg = dsl_context_registry.get()

    self.assertEqual(reg.active_contexts, [])
    with _FakeContextManager() as c1:
      self.assertEqual(reg.active_contexts, [c1])
      with _FakeContextManager() as c2:
        self.assertEqual(reg.active_contexts, [c1, c2])
        with _FakeContextManager() as c3:
          self.assertEqual(reg.active_contexts, [c1, c2, c3])
      with _FakeContextManager() as c4:
        self.assertEqual(reg.active_contexts, [c1, c4])

  def testRegistry_NodeAndContextAssociations(self):
    reg = dsl_context_registry.get()

    n0 = _FakeNode()
    with _FakeContextManager() as c1:
      n1 = _FakeNode()
      with _FakeContextManager() as c2:
        n2 = _FakeNode()
        with _FakeContextManager() as c3:
          n3 = _FakeNode()
      with _FakeContextManager() as c4:
        n4 = _FakeNode()

    # Associated nodes for each context
    self.assertEqual(reg.all_nodes, [n0, n1, n2, n3, n4])
    self.assertEqual(reg.get_nodes(c1), [n1, n2, n3, n4])
    self.assertEqual(reg.get_nodes(c2), [n2, n3])
    self.assertEqual(reg.get_nodes(c3), [n3])
    self.assertEqual(reg.get_nodes(c4), [n4])
    # Associated contexts for each node
    self.assertEqual(reg.all_contexts, [c1, c2, c3, c4])
    self.assertEqual(reg.get_contexts(n0), [])
    self.assertEqual(reg.get_contexts(n1), [c1])
    self.assertEqual(reg.get_contexts(n2), [c1, c2])
    self.assertEqual(reg.get_contexts(n3), [c1, c2, c3])
    self.assertEqual(reg.get_contexts(n4), [c1, c4])

  def testContextManager_EnterMultipleTimes(self):
    cm = _FakeContextManager()
    with cm as c1:
      pass
    with cm as c2:
      self.assertNotEqual(c1, c2)
      with self.assertRaises(RuntimeError):
        with cm:
          pass

  def testContextManager_EnterReturnValue(self):

    class UltimateContextManager(_FakeContextManager):

      def enter(self, context: _FakeContext) -> int:
        return 42

    with UltimateContextManager() as captured:
      self.assertEqual(captured, 42)

  def testNewRegistry_InnerRegistryIsolated(self):
    outer = dsl_context_registry.get()

    n0 = _FakeNode()
    with _FakeContextManager() as c1:
      n1 = _FakeNode()
      with dsl_context_registry.new_registry() as inner:
        n2 = _FakeNode()
        with _FakeContextManager() as c2:
          n3 = _FakeNode()

    self.assertEqual(outer.all_nodes, [n0, n1])
    self.assertEqual(inner.all_nodes, [n2, n3])
    self.assertEqual(outer.get_nodes(c1), [n1])
    self.assertEqual(inner.get_nodes(c2), [n3])
    self.assertEqual(outer.get_contexts(n0), [])
    self.assertEqual(outer.get_contexts(n1), [c1])
    self.assertEqual(inner.get_contexts(n2), [])
    self.assertEqual(inner.get_contexts(n3), [c2])

    for reg, node in [(inner, n0), (inner, n1), (outer, n2), (outer, n3)]:
      with self.assertRaisesRegex(ValueError, 'does not exist in the registry'):
        reg.get_contexts(node)

    for reg, context in [(inner, c1), (outer, c2)]:
      with self.assertRaisesRegex(ValueError, 'does not exist in the registry'):
        reg.get_nodes(context)


if __name__ == '__main__':
  tf.test.main()
