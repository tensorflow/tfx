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
"""Tests for tfx.dsl.input_resolution.ops.latest_create_time_op."""

import tensorflow as tf

from tfx.dsl.input_resolution.ops import ops
from tfx.dsl.input_resolution.ops import test_utils


class LatestCreateTimeOpTest(tf.test.TestCase):

  def _latest_create_time(self, *args, **kwargs):
    return test_utils.strict_run_resolver_op(
        ops.LatestCreateTime, args=args, kwargs=kwargs
    )

  def testLatestCreateTime_Empty(self):
    actual = self._latest_create_time([])
    self.assertEqual(actual, [])

  def testLatestCreateTime_SingleEntry(self):
    a1 = test_utils.DummyArtifact(id=1, create_time_since_epoch=1)
    actual = self._latest_create_time([a1])
    self.assertEqual(actual, [a1])

  def testLatestCreateTime_TieBreak(self):
    a1 = test_utils.DummyArtifact(id=1, create_time_since_epoch=5)
    a2 = test_utils.DummyArtifact(id=2, create_time_since_epoch=10)
    a3 = test_utils.DummyArtifact(id=3, create_time_since_epoch=10)

    actual = self._latest_create_time([a1, a2, a3])
    self.assertEqual(actual, [a3])

    actual = self._latest_create_time([a1, a2, a3], n=2)
    self.assertEqual(actual, [a2, a3])

  def testLatestSpan_InvalidN(self):
    a1 = test_utils.DummyArtifact()

    with self.assertRaisesRegex(ValueError, 'n must be > 0'):
      self._latest_create_time([a1], n=-1)


if __name__ == '__main__':
  tf.test.main()
