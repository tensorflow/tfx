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
"""Tests for tfx.dsl.experimental.conditionals.predicate."""

from absl.testing import parameterized
import tensorflow as tf

from tfx.dsl.experimental.conditionals import predicate
from tfx.dsl.placeholder import placeholder as ph


class PredicateTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      {'compare_op': predicate.CompareOp.EQUAL},
      {'compare_op': predicate.CompareOp.NOT_EQUAL},
      {'compare_op': predicate.CompareOp.LESS_THAN},
      {'compare_op': predicate.CompareOp.LESS_THAN_OR_EQUAL},
      {'compare_op': predicate.CompareOp.GREATER_THAN},
      {'compare_op': predicate.CompareOp.GREATER_THAN_OR_EQUAL})
  def testFromComparison(self, compare_op):
    pred = predicate.Predicate.from_comparison(compare_op, ph.input('left'),
                                               ph.input('right'))
    self.assertIsInstance(pred, predicate.Predicate)
    pred_dict = pred.to_json_dict()
    self.assertEqual(pred_dict['cmp_op'], str(compare_op))
    self.assertEqual(pred_dict['left'], ph.input('left').encode())
    self.assertEqual(pred_dict['right'], ph.input('right').encode())

  @parameterized.parameters({'primitive': 1}, {'primitive': 1.1},
                            {'primitive': 'one'})
  def testFromComparisonOnePrimitive(self, primitive):
    pred = predicate.Predicate.from_comparison(predicate.CompareOp.EQUAL,
                                               ph.input('left'), primitive)
    self.assertIsInstance(pred, predicate.Predicate)
    pred_dict = pred.to_json_dict()
    self.assertEqual(pred_dict['cmp_op'], str(predicate.CompareOp.EQUAL))
    self.assertEqual(pred_dict['left'], ph.input('left').encode())
    self.assertEqual(pred_dict['right'], primitive)

  def testNegation(self):
    pred = predicate.Predicate.from_comparison(predicate.CompareOp.EQUAL,
                                               ph.input('left'),
                                               ph.input('right'))
    not_pred = predicate.logical_not(pred)
    self.assertIsInstance(not_pred, predicate.Predicate)
    not_pred_dict = not_pred.to_json_dict()
    self.assertEqual(not_pred_dict['logical_op'], str(predicate.LogicalOp.NOT))
    self.assertEqual(not_pred_dict['left'], pred.to_json_dict())
    self.assertIsNone(not_pred_dict['right'])

  @parameterized.parameters(
      {
          'func_name': 'logical_and',
          'logical_op': predicate.LogicalOp.AND
      }, {
          'func_name': 'logical_or',
          'logical_op': predicate.LogicalOp.OR
      })
  def testBinaryLogicalOps(self, func_name, logical_op):
    pred1 = predicate.Predicate.from_comparison(predicate.CompareOp.EQUAL,
                                                ph.input('left1'),
                                                ph.input('right1'))

    pred2 = predicate.Predicate.from_comparison(predicate.CompareOp.EQUAL,
                                                ph.input('left2'),
                                                ph.input('right2'))
    result = getattr(predicate, func_name)(pred1, pred2)
    self.assertIsInstance(result, predicate.Predicate)
    result_dict = result.to_json_dict()
    self.assertEqual(result_dict['logical_op'], str(logical_op))
    self.assertEqual(result_dict['left'], pred1.to_json_dict())
    self.assertEqual(result_dict['right'], pred2.to_json_dict())


if __name__ == '__main__':
  tf.test.main()
