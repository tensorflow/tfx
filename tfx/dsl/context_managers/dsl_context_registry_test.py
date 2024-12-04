# Copyright 2024 Google LLC. All Rights Reserved.
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
"""Tests for tfx.dsl.context_managers.dsl_context_registry."""

import tensorflow as tf
from tfx.dsl.context_managers import dsl_context_registry
from tfx.dsl.context_managers import test_utils

Node = test_utils.Node
TestContext = test_utils.TestContext


class DslContextRegistryTest(tf.test.TestCase):

  def assert_registry_equal(self, reg, expected):
    test_utils.assert_registry_equal(self, reg, expected)

  def testMultipleNodesAndContexts(self):
    with dsl_context_registry.new_registry() as reg:
      a = Node('A')
      with TestContext('Ctx1') as foo1:
        b = Node('B')
        c = Node('C')
        with TestContext('Ctx2') as foo2:
          d = Node('D')
        e = Node('E')
      f = Node('F')

    self.assert_registry_equal(
        reg,
        """
        A
        Ctx1 {
          B
          C
          Ctx2 {
            D
          }
          E
        }
        F
        """,
    )
    self.assertEqual(reg.all_nodes, [a, b, c, d, e, f])
    self.assertEqual(reg.all_contexts, [foo1, foo2])
    self.assertEqual(reg.get_contexts(b), [foo1])
    self.assertEqual(reg.get_contexts(d), [foo1, foo2])
    self.assertEqual(reg.get_contexts(e), [foo1])
    self.assertEqual(reg.get_contexts(f), [])

  def testExtractForPipeline(self):
    with dsl_context_registry.new_registry() as reg:
      Node('A')
      with TestContext('Ctx1'):
        b = Node('B')
        c = Node('C')
        with TestContext('Ctx2'):
          d = Node('D')
        sub_reg = reg.extract_for_pipeline([b, c, d])
        Node('E')
      Node('F')

    self.assert_registry_equal(
        reg,
        """
        A
        Ctx1 {
          E
        }
        F
        """,
    )
    self.assert_registry_equal(
        sub_reg,
        """
        B
        C
        Ctx2 {
          D
        }
        """,
    )

  def testExtractForPipeline_ActiveContextMatters(self):
    with dsl_context_registry.new_registry() as reg:
      with TestContext('Ctx1'):
        a = Node('A')
        reg1 = reg.extract_for_pipeline([a])
        with TestContext('Ctx2'):
          b = Node('B')
          reg2 = reg.extract_for_pipeline([b])
        with TestContext('Ctx3'):
          c = Node('C')
        reg3 = reg.extract_for_pipeline([c])
        with TestContext('Ctx4'):
          d = Node('D')
      reg4 = reg.extract_for_pipeline([d])

    self.assert_registry_equal(reg, '')
    self.assert_registry_equal(reg1, 'A')
    self.assert_registry_equal(reg2, 'B')
    self.assert_registry_equal(
        reg3,
        """
        Ctx3 {
          C
        }
        """,
    )
    self.assert_registry_equal(
        reg4,
        """
        Ctx1 {
          Ctx4 {
            D
          }
        }
        """,
    )

  def testExtractForPipeline_OnlyExtractsRelevantContexts(self):
    with dsl_context_registry.new_registry() as reg:
      with TestContext('Ctx1'):
        Node('A')
        with TestContext('Ctx2'):
          Node('B')
        with TestContext('Ctx3'):
          with TestContext('Ctx4'):
            c = Node('C')
          sub_reg = reg.extract_for_pipeline([c])
          Node('D')

    self.assert_registry_equal(
        reg,
        """
        Ctx1 {
          A
          Ctx2 {
            B
          }
          Ctx3 {
            D
          }
        }
        """,
    )
    self.assert_registry_equal(
        sub_reg,
        """
        Ctx4 {
          C
        }
        """,
    )

  def testReplaceNode(self):
    with dsl_context_registry.new_registry() as reg:
      a = Node('A')
      with TestContext('Ctx1'):
        b = Node('B')
        c = Node('C')
        with TestContext('Ctx2'):
          d = Node('D')
        e = Node('E')
      f = Node('F')

    reg.replace_node(a, Node('A_copy'))
    reg.replace_node(b, Node('B_copy'))
    reg.replace_node(c, Node('C_copy'))
    reg.replace_node(d, Node('D_copy'))
    reg.replace_node(e, Node('E_copy'))
    reg.replace_node(f, Node('F_copy'))

    self.assert_registry_equal(
        reg,
        """
        A_copy
        Ctx1 {
          B_copy
          C_copy
          Ctx2 {
            D_copy
          }
          E_copy
        }
        F_copy
        """,
    )

  def testFinalize(self):
    with dsl_context_registry.new_registry() as reg:
      Node('A')
      reg.finalize()
      with self.assertRaises(RuntimeError):
        Node('B')
