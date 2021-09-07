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
"""Tests for tfx.dsl.context_manager."""

import threading
from typing import Dict, Any

import tensorflow as tf
from tfx.dsl.components.base import base_node
from tfx.dsl.context_managers import context_manager


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


class _FakeContext(context_manager.DslContext):
  pass


class _FakeContextManager(context_manager.DslContextManager):

  def create_context(self) -> _FakeContext:
    return _FakeContext()

  def enter(self, context: _FakeContext) -> _FakeContext:
    return context


class ContextManagerTest(tf.test.TestCase):

  def reset_registry(self) -> context_manager._DslContextRegistry:
    result = context_manager._registry = context_manager._DslContextRegistry()
    return result

  def testContext_ContextIdAttrFactory(self):
    # ID is in format <classname>:<num> where <num> is incremental
    # regardless of the depth of the contexts.
    self.reset_registry()
    with _FakeContextManager() as c1:
      self.assertEqual(c1.id, '_FakeContext:1')
      with _FakeContextManager() as c2:
        self.assertEqual(c2.id, '_FakeContext:2')
        with _FakeContextManager() as c3:
          self.assertEqual(c3.id, '_FakeContext:3')
    with _FakeContextManager() as c4:
      self.assertEqual(c4.id, '_FakeContext:4')
    # ID count is reset after registry reset.
    self.reset_registry()
    with _FakeContextManager() as c5:
      self.assertEqual(c5.id, '_FakeContext:1')

  def testContext_ParentAttrFactory(self):
    registry = self.reset_registry()
    bg = registry.background_context

    with _FakeContextManager() as c1:
      self.assertIs(c1.parent, bg)
      with _FakeContextManager() as c2:
        self.assertIs(c2.parent, c1)
        with _FakeContextManager() as c3:
          self.assertEqual(c3.parent, c2)
      with _FakeContextManager() as c4:
        self.assertIs(c4.parent, c1)

  def testContext_Ancestors(self):
    registry = self.reset_registry()
    bg = registry.background_context

    self.assertEqual(list(bg.ancestors), [])
    with _FakeContextManager() as c1:
      self.assertEqual(list(c1.ancestors), [bg])
      with _FakeContextManager() as c2:
        self.assertEqual(list(c2.ancestors), [bg, c1])
        with _FakeContextManager() as c3:
          self.assertEqual(list(c3.ancestors), [bg, c1, c2])
      with _FakeContextManager() as c4:
        self.assertEqual(list(c4.ancestors), [bg, c1])

  def testRegistry_AllContexts(self):
    registry = self.reset_registry()
    bg = registry.background_context

    self.assertEqual(registry.all_contexts, [bg])
    with _FakeContextManager() as c1:
      self.assertEqual(registry.all_contexts, [bg, c1])
      with _FakeContextManager() as c2:
        self.assertEqual(registry.all_contexts, [bg, c1, c2])
        with _FakeContextManager() as c3:
          self.assertEqual(registry.all_contexts, [bg, c1, c2, c3])
      with _FakeContextManager() as c4:
        self.assertEqual(registry.all_contexts, [bg, c1, c2, c3, c4])

  def testRegistry_ActiveContexts(self):
    registry = self.reset_registry()
    bg = registry.background_context

    self.assertEqual(registry.active_contexts, [bg])
    with _FakeContextManager() as c1:
      self.assertEqual(registry.active_contexts, [bg, c1])
      with _FakeContextManager() as c2:
        self.assertEqual(registry.active_contexts, [bg, c1, c2])
        with _FakeContextManager() as c3:
          self.assertEqual(registry.active_contexts, [bg, c1, c2, c3])
      with _FakeContextManager() as c4:
        self.assertEqual(registry.active_contexts, [bg, c1, c4])

  def testRegistry_NodeAndContextAssociations(self):
    registry = self.reset_registry()
    bg = registry.background_context

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
    self.assertEqual(registry.get_nodes(bg), [n0, n1, n2, n3, n4])
    self.assertEqual(registry.get_nodes(c1), [n1, n2, n3, n4])
    self.assertEqual(registry.get_nodes(c2), [n2, n3])
    self.assertEqual(registry.get_nodes(c3), [n3])
    self.assertEqual(registry.get_nodes(c4), [n4])
    # Convenient property for calling registry.get_nodes()
    self.assertEqual(bg.nodes, [n0, n1, n2, n3, n4])
    self.assertEqual(c1.nodes, [n1, n2, n3, n4])
    self.assertEqual(c2.nodes, [n2, n3])
    self.assertEqual(c3.nodes, [n3])
    self.assertEqual(c4.nodes, [n4])
    # Associated contexts for each node
    self.assertEqual(registry.get_contexts(n0), [bg])
    self.assertEqual(registry.get_contexts(n1), [bg, c1])
    self.assertEqual(registry.get_contexts(n2), [bg, c1, c2])
    self.assertEqual(registry.get_contexts(n3), [bg, c1, c2, c3])
    self.assertEqual(registry.get_contexts(n4), [bg, c1, c4])

  def testContextManager_EnterMultipleTimes(self):
    cm = _FakeContextManager()
    with cm as c1:
      pass
    with cm as c2:
      self.assertNotEqual(c1, c2)
      with cm as c3:
        self.assertNotEqual(c2, c3)

  def testContextManager_EnterReturnValue(self):

    class UltimateContextManager(_FakeContextManager):

      def enter(self, context: _FakeContext) -> int:
        return 42

    with UltimateContextManager() as captured:
      self.assertEqual(captured, 42)

  def testRegistry_MultiThreads(self):
    num_threads = 2
    b = threading.Barrier(num_threads)

    def test():
      with _FakeContextManager() as c1:
        self.assertEqual(c1.id, '_FakeContext:1')
        b.wait()
        with _FakeContextManager() as c2:
          self.assertEqual(c2.id, '_FakeContext:2')
          self.assertEqual(c2.parent, c1)

    threads = [threading.Thread(target=test) for i in range(num_threads)]
    for thread in threads:
      thread.start()
    for thread in threads:
      thread.join()  # Expects no unhandled exceptions.

  def testGetNodes(self):
    self.reset_registry()

    n0 = _FakeNode()
    with _FakeContextManager() as c1:
      n1 = _FakeNode()
      with _FakeContextManager() as c2:
        n2 = _FakeNode()
        with _FakeContextManager() as c3:
          n3 = _FakeNode()
      with _FakeContextManager() as c4:
        n4 = _FakeNode()

    with self.subTest('With argument'):
      self.assertEqual(context_manager.get_nodes(c1), [n1, n2, n3, n4])
      self.assertEqual(context_manager.get_nodes(c2), [n2, n3])
      self.assertEqual(context_manager.get_nodes(c3), [n3])
      self.assertEqual(context_manager.get_nodes(c4), [n4])

    with self.subTest('Without argument'):
      # get_nodes() without argument queries nodes for background context,
      # which works as a node registry
      self.assertEqual(context_manager.get_nodes(), [n0, n1, n2, n3, n4])


if __name__ == '__main__':
  tf.test.main()
