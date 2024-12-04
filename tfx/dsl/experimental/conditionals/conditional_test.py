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
"""Tests for tfx.dsl.experimental.conditionals.conditional."""

import tensorflow as tf
from tfx.dsl.context_managers import dsl_context_registry
from tfx.dsl.context_managers import test_utils
from tfx.dsl.experimental.conditionals import conditional
from tfx.dsl.placeholder import placeholder
from tfx.orchestration import pipeline


Node = test_utils.Node


class ConditionalTest(tf.test.TestCase):

  def assertPredicatesEqual(self, node, *expected_predicates):
    self.assertEqual(
        conditional.get_predicates(node, dsl_context_registry.get()),
        expected_predicates)

  def testSingleCondition(self):
    pred = placeholder.input('foo') == 'bar'
    with conditional.Cond(pred):
      node1 = Node('node1')
      node2 = Node('node2')
    self.assertPredicatesEqual(node1, pred)
    self.assertPredicatesEqual(node2, pred)

  def testNestedCondition(self):
    pred1 = placeholder.input('foo') == 'bar'
    pred2 = placeholder.input('foo') == 'baz'
    with conditional.Cond(pred1):
      node1 = Node('node1')
      with conditional.Cond(pred2):
        node2 = Node('node2')
    self.assertPredicatesEqual(node1, pred1)
    self.assertPredicatesEqual(node2, pred1, pred2)

  def testReusePredicate(self):
    pred = placeholder.input('foo') == 'bar'
    with conditional.Cond(pred):
      node1 = Node('node1')
    with conditional.Cond(pred):
      node2 = Node('node2')
    self.assertPredicatesEqual(node1, pred)
    self.assertPredicatesEqual(node2, pred)

  def testNestedConditionWithDuplicatePredicates_SameInstance(self):
    pred = placeholder.input('foo') == 'bar'
    with self.assertRaisesRegex(
        ValueError, 'Nested conditionals with duplicate predicates'):
      with conditional.Cond(pred):
        unused_node1 = Node('node1') # noqa: F841
        with conditional.Cond(pred):
          unused_node2 = Node('node2') # noqa: F841

  def testNestedConditionWithDuplicatePredicates_EquivalentPredicate(self):
    with self.assertRaisesRegex(
        ValueError, 'Nested conditionals with duplicate predicates'
    ):
      with conditional.Cond(placeholder.input('foo') == 'bar'):
        unused_node1 = Node('node1') # noqa: F841
        with conditional.Cond(placeholder.input('foo') == 'bar'):
          unused_node2 = Node('node2') # noqa: F841

  def testCond_Subpipeline(self):
    pred = placeholder.input('foo') == 'bar'
    with conditional.Cond(pred):
      a = Node('A')
      b = Node('B')
      p = pipeline.Pipeline(pipeline_name='p', components=[a, b])
    p_out = pipeline.Pipeline(pipeline_name='p_out', components=[p])

    # Nodes from subpipeline are isolated from the outer pipeline context thus
    # are not conditional.
    self.assertEmpty(conditional.get_predicates(a, p.dsl_context_registry))

    # Subpipeline is under conditional of the outer pipeline.
    self.assertCountEqual(
        conditional.get_predicates(p, p_out.dsl_context_registry), [pred]
    )
