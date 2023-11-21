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

from typing import Any, Dict

import tensorflow as tf
from tfx.dsl.components.base import base_node
from tfx.dsl.context_managers import dsl_context_registry
from tfx.dsl.experimental.conditionals import conditional
from tfx.dsl.placeholder import placeholder


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


class ConditionalTest(tf.test.TestCase):

  def assertPredicatesEqual(self, node, *expected_predicates):
    self.assertEqual(
        conditional.get_predicates(node, dsl_context_registry.get()),
        expected_predicates)

  def testSingleCondition(self):
    pred = placeholder.input('foo') == 'bar'
    with conditional.Cond(pred):
      node1 = _FakeNode().with_id('node1')
      node2 = _FakeNode().with_id('node2')
    self.assertPredicatesEqual(node1, pred)
    self.assertPredicatesEqual(node2, pred)

  def testNestedCondition(self):
    pred1 = placeholder.input('foo') == 'bar'
    pred2 = placeholder.input('foo') == 'baz'
    with conditional.Cond(pred1):
      node1 = _FakeNode().with_id('node1')
      with conditional.Cond(pred2):
        node2 = _FakeNode().with_id('node2')
    self.assertPredicatesEqual(node1, pred1)
    self.assertPredicatesEqual(node2, pred1, pred2)

  def testReusePredicate(self):
    pred = placeholder.input('foo') == 'bar'
    with conditional.Cond(pred):
      node1 = _FakeNode().with_id('node1')
    with conditional.Cond(pred):
      node2 = _FakeNode().with_id('node2')
    self.assertPredicatesEqual(node1, pred)
    self.assertPredicatesEqual(node2, pred)

  def testNestedConditionWithDuplicatePredicates(self):
    # Note: This only catches the duplication if the _same_ predicate (in terms
    # of Python object identity) is used. Ideally we would also detect
    # equivalent predicates (like __eq__) but placeholders cannot implement
    # __eq__ itself (due to its special function in creating predicates from
    # ChannelWrappedPlaceholder) and placeholders also don't offer another
    # equality function at the moment.
    pred = placeholder.input('foo') == 'bar'
    with self.assertRaisesRegex(
        ValueError, 'Nested conditionals with duplicate predicates'):
      with conditional.Cond(pred):
        unused_node1 = _FakeNode().with_id('node1')
        with conditional.Cond(pred):
          unused_node2 = _FakeNode().with_id('node2')


if __name__ == '__main__':
  tf.test.main()
