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
"""Tests for tfx.dsl.input_resolution.ops.static_span_range_op."""

import tensorflow as tf

from tfx.dsl.input_resolution.ops import ops
from tfx.dsl.input_resolution.ops import test_utils


class StaticSpanRangeOpTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()

    self.a1 = test_utils.DummyArtifact()
    self.a2 = test_utils.DummyArtifact()
    self.a3 = test_utils.DummyArtifact()
    self.a4 = test_utils.DummyArtifact()
    self.a5 = test_utils.DummyArtifact()
    self.a6 = test_utils.DummyArtifact()

    self.artifacts = [self.a1, self.a2, self.a3, self.a4, self.a5, self.a6]

    spans = [1, 2, 3, 4, 4, 4]
    versions = [0, 0, 1, 0, 1, 2]
    for i, artifact in enumerate(self.artifacts):
      artifact.span = spans[i]
      artifact.version = versions[i]

  def testStaticSpanRange_Empty(self):
    actual = test_utils.run_resolver_op(ops.StaticSpanRange, [])
    self.assertEqual(actual, [])

  def testStaticSpanRange_NoStartEndSpan(self):
    actual = test_utils.run_resolver_op(ops.StaticSpanRange, self.artifacts)
    self.assertEqual(actual, [self.a1, self.a2, self.a3, self.a6])

  def testStaticSpanRange_KeepAll(self):
    actual = test_utils.run_resolver_op(
        ops.StaticSpanRange,
        self.artifacts,
        start_span=3,
        end_span=4,
        keep_all_versions=True)
    self.assertEqual(actual, [self.a3, self.a4, self.a5, self.a6])

  def testStaticSpanRange_OutOfBoundStartEndSpan(self):
    actual = test_utils.run_resolver_op(
        ops.StaticSpanRange, self.artifacts, start_span=-1, end_span=10)
    self.assertEqual(actual, [self.a1, self.a2, self.a3, self.a6])

  def testStaticSpanRange(self):
    actual = test_utils.run_resolver_op(
        ops.StaticSpanRange, self.artifacts, start_span=1, end_span=3)
    self.assertEqual(actual, [self.a1, self.a2, self.a3])


if __name__ == '__main__':
  tf.test.main()
