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
"""Tests for tfx.dsl.input_resolution.ops.sliding_window_op."""

import tensorflow as tf

from tfx.dsl.input_resolution.ops import ops
from tfx.dsl.input_resolution.ops import test_utils


class SlidingWindowOpTest(tf.test.TestCase):

  def testSlidingWindow_Empty(self):
    actual = test_utils.run_resolver_op(ops.SlidingWindow, [])
    self.assertEqual(actual, [])

  def testSlidingWindow_NonPositiveN(self):
    a1 = test_utils.DummyArtifact()

    expected_error = "sliding_window must be > 0"
    with self.assertRaisesRegex(ValueError, expected_error):
      test_utils.run_resolver_op(ops.SlidingWindow, [a1], window_size=0)

    with self.assertRaisesRegex(ValueError, expected_error):
      test_utils.run_resolver_op(ops.SlidingWindow, [a1], window_size=-1)

  def testSlidingWindow_SingleEntry(self):
    a1 = test_utils.DummyArtifact()

    actual = test_utils.run_resolver_op(ops.SlidingWindow, [a1])
    self.assertEqual(actual, [{"window": [a1]}])

    actual = test_utils.run_resolver_op(
        ops.SlidingWindow, [a1], window_size=1, output_key="key")
    self.assertEqual(actual, [{"key": [a1]}])

    # The final window size will be 0, so no artifacts will be returned.
    actual = test_utils.run_resolver_op(ops.SlidingWindow, [a1], window_size=2)
    self.assertEqual(actual, [])

  def testSlidingWindow_MultipleEntries(self):
    a1 = test_utils.DummyArtifact()
    a2 = test_utils.DummyArtifact()
    a3 = test_utils.DummyArtifact()
    a4 = test_utils.DummyArtifact()

    artifacts = [a1, a2, a3, a4]

    actual = test_utils.run_resolver_op(ops.SlidingWindow, artifacts)
    self.assertEqual(actual, [
        {
            "window": [a1]
        },
        {
            "window": [a2]
        },
        {
            "window": [a3]
        },
        {
            "window": [a4]
        },
    ])

    actual = test_utils.run_resolver_op(
        ops.SlidingWindow, artifacts, window_size=1)
    self.assertEqual(actual, [
        {
            "window": [a1]
        },
        {
            "window": [a2]
        },
        {
            "window": [a3]
        },
        {
            "window": [a4]
        },
    ])

    actual = test_utils.run_resolver_op(
        ops.SlidingWindow, artifacts, window_size=2)
    self.assertEqual(actual, [
        {
            "window": [a1, a2]
        },
        {
            "window": [a2, a3]
        },
        {
            "window": [a3, a4]
        },
    ])

    actual = test_utils.run_resolver_op(
        ops.SlidingWindow, artifacts, window_size=3)
    self.assertEqual(actual, [
        {
            "window": [a1, a2, a3]
        },
        {
            "window": [a2, a3, a4]
        },
    ])

    actual = test_utils.run_resolver_op(
        ops.SlidingWindow, artifacts, window_size=4)
    self.assertEqual(actual, [{"window": [a1, a2, a3, a4]}])

    # The final window size will be 0, so no artifacts will be returned.
    actual = test_utils.run_resolver_op(
        ops.SlidingWindow, artifacts, window_size=5)
    self.assertEqual(actual, [])


if __name__ == "__main__":
  tf.test.main()
