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
"""Tests for tfx.orchestration.portable.input_resolution.partition_utils."""

import tensorflow as tf
from tfx.orchestration.portable.input_resolution import partition_utils


def partition(**kwargs):
  return partition_utils.Partition(kwargs)


class PartitionUtilsTest(tf.test.TestCase):

  def testCompositeKey(self):
    k = partition(x=1, y=2, z=3)
    self.assertEqual(k.dimensions, ('x', 'y', 'z'))
    self.assertEqual(k.partial(['y', 'x']), (2, 1))
    self.assertEqual(k.partial([]), ())
    self.assertTrue(k)
    self.assertEqual(partition(x=1, y=2), partition(y=2, x=1))
    self.assertEqual(
        partition(x=1, y=2) | partition(y=2, z=3),
        partition(x=1, y=2, z=3))
    self.assertEqual(
        partition(x=1) | partition(y=2, z=3),
        partition(x=1, y=2, z=3))
    with self.assertRaises(ValueError):
      partition(x=1, y=2) | partition(y=1, z=3)  # pylint: disable=expression-not-assigned

  def testNoPartition(self):
    empty = partition_utils.NO_PARTITION
    self.assertEqual(empty.dimensions, ())
    self.assertEqual(empty.partial([]), ())
    self.assertFalse(empty)

  def testJoin(self):

    def check(lhs, rhs, expected, merge_fn=lambda x, y: x + y):
      with self.subTest(lhs=lhs, rhs=rhs, expected=expected):
        result = partition_utils.join(
            lhs, rhs, merge_fn=merge_fn)
        self.assertEqual(result, expected)

    check(
        lhs=[(partition(), 'a'), (partition(), 'b')],
        rhs=[(partition(), '1'), (partition(), '2')],
        expected=[
            (partition(), 'a1'),
            (partition(), 'a2'),
            (partition(), 'b1'),
            (partition(), 'b2'),
        ]
    )

    check(
        lhs=[(partition(), 'a'), (partition(), 'b')],
        rhs=[],
        expected=[]
    )

    check(
        lhs=[],
        rhs=[(partition(), '1'), (partition(), '2')],
        expected=[]
    )

    check(
        lhs=[(partition(x=1), 'x1'), (partition(x=2), 'x2')],
        rhs=[(partition(y=1), 'y1'), (partition(y=2), 'y2')],
        expected=[
            (partition(x=1, y=1), 'x1y1'),
            (partition(x=1, y=2), 'x1y2'),
            (partition(x=2, y=1), 'x2y1'),
            (partition(x=2, y=2), 'x2y2'),
        ]
    )

    check(
        lhs=[(partition(x=1), 'a'), (partition(x=2), 'b')],
        rhs=[(partition(x=1), 'pple'), (partition(x=2), 'anana')],
        expected=[
            (partition(x=1), 'apple'),
            (partition(x=2), 'banana'),
        ]
    )

    check(
        lhs=[(partition(x=1, z=1), 'x1'), (partition(x=2, z=2), 'x2')],
        rhs=[(partition(y=1, z=1), 'y1'), (partition(y=2, z=2), 'y2')],
        expected=[
            (partition(x=1, y=1, z=1), 'x1y1'),
            (partition(x=2, y=2, z=2), 'x2y2'),
        ]
    )

    check(
        lhs=[
            (partition(x=1, y=1), 'x1y1'),
            (partition(x=1, y=2), 'x1y2'),
            (partition(x=2, y=1), 'x2y1'),
            (partition(x=2, y=2), 'x2y2'),
        ],
        rhs=[
            (partition(x=1, z=1), 'z1'),
            (partition(x=2, z=2), 'z2'),
        ],
        expected=[
            (partition(x=1, y=1, z=1), 'x1y1z1'),
            (partition(x=1, y=2, z=1), 'x1y2z1'),
            (partition(x=2, y=1, z=2), 'x2y1z2'),
            (partition(x=2, y=2, z=2), 'x2y2z2'),
        ]
    )

    check(
        lhs=[
            (partition(x=1, y=1), 'x1y1'),
            (partition(x=1, y=2), 'x1y2'),
            (partition(x=2, y=1), 'x2y1'),
            (partition(x=2, y=2), 'x2y2'),
        ],
        rhs=[
            (partition(x=1, z=1), 'z1'),
            (partition(x=1, z=2), 'z2'),
            (partition(x=2, z=3), 'z3'),
            (partition(x=2, z=4), 'z4'),
        ],
        expected=[
            (partition(x=1, y=1, z=1), 'x1y1z1'),
            (partition(x=1, y=1, z=2), 'x1y1z2'),
            (partition(x=1, y=2, z=1), 'x1y2z1'),
            (partition(x=1, y=2, z=2), 'x1y2z2'),
            (partition(x=2, y=1, z=3), 'x2y1z3'),
            (partition(x=2, y=1, z=4), 'x2y1z4'),
            (partition(x=2, y=2, z=3), 'x2y2z3'),
            (partition(x=2, y=2, z=4), 'x2y2z4'),
        ]
    )


if __name__ == '__main__':
  tf.test.main()
