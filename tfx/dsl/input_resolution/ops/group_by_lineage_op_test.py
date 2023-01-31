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
"""Tests for tfx.dsl.input_resolution.ops.group_by_lineage_op."""

import random

from absl.testing import parameterized
import tensorflow as tf
from tfx.dsl.input_resolution.ops import group_by_lineage_op
from tfx.dsl.input_resolution.ops import test_utils
from tfx.orchestration.portable.input_resolution import exceptions
from tfx.types import artifact_utils
from tfx.utils import test_case_utils


def _shuffle(values):
  values = list(values)
  random.shuffle(values)
  return values


class _LineageUtils(test_case_utils.MlmdMixins):

  def _prepare_tfx_artifacts(self, n):
    artifacts = [self.put_artifact('MyArtifact') for i in range(n)]
    artifact_type = self.store.get_artifact_type('MyArtifact')
    return artifact_utils.deserialize_artifacts(artifact_type, artifacts)

  def _unwrap_mlmd_artifacts(self, tfx_artifacts):
    if isinstance(tfx_artifacts, list):
      return [t.mlmd_artifact for t in tfx_artifacts]
    else:
      return [tfx_artifacts.mlmd_artifact]

  def _put_lineage(self, *path):
    while len(path) >= 2:
      self.put_execution(
          'MyExecution',
          inputs={'in': self._unwrap_mlmd_artifacts(path[0])},
          outputs={'out': self._unwrap_mlmd_artifacts(path[1])},
      )
      path = path[1:]


class GroupByDisjointLineageTest(
    tf.test.TestCase,
    parameterized.TestCase,
    _LineageUtils,
):

  def setUp(self):
    super().setUp()
    self.init_mlmd()

  def _group_by_disjoint_lineage(self, *args, **kwargs):
    return test_utils.strict_run_resolver_op(
        group_by_lineage_op.GroupByDisjointLineage,
        args=args,
        kwargs=kwargs,
        store=self.store,
    )

  @parameterized.parameters(
      (range(3), [], [[0], [1], [2]]),
      (range(3), [(0, 1)], [[0, 1], [2]]),
      (range(10), [], [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]),
      (range(10),
       [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)],
       [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]),
      (range(10),
       [(0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8), (7, 9)],
       [[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]]),
      (range(10),
       [(2, 4), (5, 3), (6, 4), (3, 4)],
       [[0], [1], [2, 3, 4, 5, 6], [7], [8], [9]]),
  )  # pyformat: disable
  def testFindDisjointSets(self, verts, edges, expected_disjoint_sets):
    actual = group_by_lineage_op._find_disjoint_sets(
        _shuffle(verts), _shuffle(edges)
    )
    self.assertEqual(actual, expected_disjoint_sets)

  def testGroupByDisjointLineage(self):
    a1, a2, a3, b1, b2, b3, b4, c1, c2, c3, c4 = self._prepare_tfx_artifacts(11)
    self._put_lineage(a1, b1, c1)
    self._put_lineage(a2, b2, c2)
    self._put_lineage(a2, b3, c3)
    self._put_lineage(a3, b4, c4)

    result = self._group_by_disjoint_lineage({
        'a': [a1, a2, a3],
        'b': [b1, b2, b3, b4],
        'c': [c1, c2, c3, c4],
    })

    self.assertEqual(result, [
        {'a': [a1], 'b': [b1], 'c': [c1]},
        {'a': [a2], 'b': [b2, b3], 'c': [c2, c3]},
        {'a': [a3], 'b': [b4], 'c': [c4]},
    ])  # pyformat: disable

  def testGroupByDisjointLineage_RequireAll(self):
    a1, a2, a3, b1, b2, b4, c1, c3, c4 = self._prepare_tfx_artifacts(9)
    self._put_lineage(a1, [b1, c1])
    self._put_lineage(a2, b2)
    self._put_lineage(a3, c3)
    self._put_lineage([], [b4, c4])
    inputs = {
        'a': [a1, a2, a3],
        'b': [b1, b2, b4],
        'c': [c1, c3, c4],
    }
    with self.subTest('with require_all = False'):
      result = self._group_by_disjoint_lineage(inputs, require_all=False)
      self.assertEqual(result, [
          {'a': [a1], 'b': [b1], 'c': [c1]},
          {'a': [a2], 'b': [b2], 'c': []},
          {'a': [a3], 'b': [], 'c': [c3]},
          {'a': [], 'b': [b4], 'c': [c4]},
      ])  # pyformat: disable

    with self.subTest('with require_all = True'):
      result = self._group_by_disjoint_lineage(inputs, require_all=True)
      self.assertEqual(result, [
          {'a': [a1], 'b': [b1], 'c': [c1]},
      ])  # pyformat: disable

  def testGroupByDisjointLineage_SiblingsAreConnected(self):
    a1, a2, b1, b2 = self._prepare_tfx_artifacts(4)
    self._put_lineage([], [a1, b1])
    self._put_lineage([], [a2, b2])
    result = self._group_by_disjoint_lineage({'a': [a1, a2], 'b': [b1, b2]})
    self.assertEqual(result, [
        {'a': [a1], 'b': [b1]},
        {'a': [a2], 'b': [b2]},
    ])  # pyformat: disable

  def testGroupByDisjointLineage_InputAndOutputAreConnected(self):
    a1, a2, b1, b2 = self._prepare_tfx_artifacts(4)
    self._put_lineage(a1, b1)
    self._put_lineage(a2, b2)
    result = self._group_by_disjoint_lineage({'a': [a1, a2], 'b': [b1, b2]})
    self.assertEqual(result, [
        {'a': [a1], 'b': [b1]},
        {'a': [a2], 'b': [b2]},
    ])  # pyformat: disable

  def testGroupByDisjointLineage_ChainingIsConnected(self):
    a1, a2, b1, b2, c1, c2 = self._prepare_tfx_artifacts(6)
    self._put_lineage(a1, b1, c1)
    self._put_lineage(a2, b2, c2)
    result = self._group_by_disjoint_lineage(
        {'a': [a1, a2], 'b': [b1, b2], 'c': [c1, c2]}
    )
    self.assertEqual(result, [
        {'a': [a1], 'b': [b1], 'c': [c1]},
        {'a': [a2], 'b': [b2], 'c': [c2]},
    ])  # pyformat: disable

  def testGroupByDisjointLineage_MoreThanTwoHopsAreDisjoint(self):
    a1, a2, b1, b2, c1, c2 = self._prepare_tfx_artifacts(6)
    self._put_lineage(a1, b1, c1)
    self._put_lineage(a2, b2, c2)
    result = self._group_by_disjoint_lineage(
        {'a': [a1, a2], 'c': [c1, c2]}, require_all=False
    )
    self.assertEqual(result, [
        {'a': [a1], 'c': []},
        {'a': [a2], 'c': []},
        {'a': [], 'c': [c1]},
        {'a': [], 'c': [c2]},
    ])  # pyformat: disable

  def testGroupByDisjointLineage_ResultOrder(self):
    a_list = self._prepare_tfx_artifacts(10)
    b_list = self._prepare_tfx_artifacts(10)
    c_list = self._prepare_tfx_artifacts(10)
    for a, b in zip(a_list, _shuffle(b_list)):
      self._put_lineage(a, b)
    for b, c in zip(b_list, _shuffle(c_list)):
      self._put_lineage(b, c)

    result = self._group_by_disjoint_lineage(
        {'a': _shuffle(a_list), 'b': _shuffle(b_list), 'c': _shuffle(c_list)}
    )

    self.assertEqual(a_list, [result_item['a'][0] for result_item in result])

  def testGroupByDisjointLineage_EmptyInput(self):
    self.assertEmpty(self._group_by_disjoint_lineage({}))
    self.assertEmpty(self._group_by_disjoint_lineage({'a': []}))

  def testGroupByDisjointLineage_SameArtifactInMultipleKeys(self):
    [a] = self._prepare_tfx_artifacts(1)
    result = self._group_by_disjoint_lineage({'a1': [a], 'a2': [a]})
    self.assertEqual(result, [{'a1': [a], 'a2': [a]}])

  def testGroupByDisjointLineage_DuplicatedArtifacts_Deduplicated(self):
    [a] = self._prepare_tfx_artifacts(1)
    result = self._group_by_disjoint_lineage({'a': [a, a]})
    self.assertEqual(result, [{'a': [a]}])


class GroupByPivotTest(tf.test.TestCase, _LineageUtils):

  def setUp(self):
    super().setUp()
    self.init_mlmd()

  def _group_by_pivot(self, *args, **kwargs):
    return test_utils.strict_run_resolver_op(
        group_by_lineage_op.GroupByPivot,
        args=args,
        kwargs=kwargs,
        store=self.store,
    )

  def testGroupByPivot(self):
    a1, a2, a3, b1, b2, b3, b4, c1, c2, c3, c4 = self._prepare_tfx_artifacts(11)
    self._put_lineage(a1, b1, c1)
    self._put_lineage(a2, b2, c2)
    self._put_lineage(a2, b3, c3)
    self._put_lineage(a3, b4, c4)
    inputs = {
        'a': [a1, a2, a3],
        'b': [b1, b2, b3, b4],
        'c': [c1, c2, c3, c4],
    }

    with self.subTest('Group by a'):
      result = self._group_by_pivot(inputs, pivot_key='a')

      self.assertEqual(result, [
          {'a': [a1], 'b': [b1], 'c': []},
          {'a': [a2], 'b': [b2, b3], 'c': []},
          {'a': [a3], 'b': [b4], 'c': []},
      ])  # pyformat: disable

    with self.subTest('Group by b'):
      result = self._group_by_pivot(inputs, pivot_key='b')

      self.assertEqual(result, [
          {'a': [a1], 'b': [b1], 'c': [c1]},
          {'a': [a2], 'b': [b2], 'c': [c2]},
          {'a': [a2], 'b': [b3], 'c': [c3]},
          {'a': [a3], 'b': [b4], 'c': [c4]},
      ])  # pyformat: disable

    with self.subTest('Group by c'):
      result = self._group_by_pivot(inputs, pivot_key='c')

      self.assertEqual(result, [
          {'a': [], 'b': [b1], 'c': [c1]},
          {'a': [], 'b': [b2], 'c': [c2]},
          {'a': [], 'b': [b3], 'c': [c3]},
          {'a': [], 'b': [b4], 'c': [c4]},
      ])  # pyformat: disable

  def testGroupByPivot_InvalidPivot(self):
    a, b = self._prepare_tfx_artifacts(2)
    self._put_lineage(a, b)
    inputs = {'a': [a], 'b': [b]}

    with self.assertRaises(exceptions.FailedPreconditionError):
      self._group_by_pivot(inputs, pivot_key='invalid_pivot')

  def testGroupByPivot_EmptyPivot(self):
    a, b = self._prepare_tfx_artifacts(2)
    self._put_lineage(a, b)
    inputs = {'a': [a], 'b': [b], 'c': []}

    with self.subTest('Empty pivot'):
      result = self._group_by_pivot(inputs, pivot_key='c')
      self.assertEmpty(result)

    with self.subTest('Non-empty pivot'):
      result = self._group_by_pivot(inputs, pivot_key='a')
      self.assertEqual(result, [{'a': [a], 'b': [b], 'c': []}])

  def testGroupByPivot_RequireAll(self):
    a1, a2, a3, b1, b2, b4, c1, c3, c4 = self._prepare_tfx_artifacts(9)
    self._put_lineage(a1, [b1, c1])
    self._put_lineage(a2, b2)
    self._put_lineage(a3, c3)
    self._put_lineage([], [b4, c4])
    inputs = {
        'a': [a1, a2, a3],
        'b': [b1, b2, b4],
        'c': [c1, c3, c4],
    }

    with self.subTest('Group by b'):
      result = self._group_by_pivot(inputs, pivot_key='b')
      self.assertEqual(result, [
          {'a': [a1], 'b': [b1], 'c': [c1]},
          {'a': [a2], 'b': [b2], 'c': []},
          {'a': [], 'b': [b4], 'c': [c4]},
      ])  # pyformat: disable

    with self.subTest('Require all'):
      result = self._group_by_pivot(inputs, pivot_key='b', require_all=True)
      self.assertEqual(result, [
          {'a': [a1], 'b': [b1], 'c': [c1]}
      ])  # pyformat: disable

  def testGroupByPivot_SiblingsAreConnected(self):
    a, b = self._prepare_tfx_artifacts(2)
    self._put_lineage([], [a, b])
    result = self._group_by_pivot({'a': [a], 'b': [b]}, pivot_key='a')
    self.assertEqual(result, [{'a': [a], 'b': [b]}])

  def testGroupByPivot_InputAndOutputAreConnected(self):
    a, b = self._prepare_tfx_artifacts(2)
    self._put_lineage(a, b)
    result = self._group_by_pivot({'a': [a], 'b': [b]}, pivot_key='a')
    self.assertEqual(result, [{'a': [a], 'b': [b]}])

  def testGroupByPivot_ChainingIsNotConnected(self):
    a, b, c = self._prepare_tfx_artifacts(3)
    self._put_lineage(a, b, c)
    with self.subTest('All connected'):
      result = self._group_by_pivot(
          {'a': [a], 'b': [b], 'c': [c]}, pivot_key='a'
      )
      self.assertEqual(result, [{'a': [a], 'b': [b], 'c': []}])

  def testGroupByPivot_SelfIsNotNeighbor(self):
    [a] = self._prepare_tfx_artifacts(1)
    result = self._group_by_pivot({'a1': [a], 'a2': [a]}, pivot_key='a1')
    self.assertEqual(result, [{'a1': [a], 'a2': []}])

  def testGroupByPivot_DuplicatedPivotPreserved(self):
    [a] = self._prepare_tfx_artifacts(1)
    result = self._group_by_pivot({'a': [a, a]}, pivot_key='a')
    self.assertEqual(result, [{'a': [a]}, {'a': [a]}])


if __name__ == '__main__':
  tf.test.main()
