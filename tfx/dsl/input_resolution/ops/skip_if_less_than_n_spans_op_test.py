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
"""Tests for tfx.dsl.input_resolution.ops.skip_if_less_than_n_spans_op."""

import tensorflow as tf

from tfx.dsl.input_resolution.ops import ops
from tfx.dsl.input_resolution.ops import test_utils
from tfx.orchestration.portable.input_resolution import exceptions


class SkipIfLessThanNSpansOpTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()

    a1 = test_utils.DummyArtifact()
    a2 = test_utils.DummyArtifact()
    a3 = test_utils.DummyArtifact()
    a4 = test_utils.DummyArtifact()
    self.artifacts = [a1, a2, a3, a4]

    spans = [1, 3, 3, 7]
    for artifact, span in zip(self.artifacts, spans):
      artifact.span = span

  def testSkipIfLessThanNSpans_LessThanNSpans_RaisesSkipSignal(self):
    with self.assertRaises(exceptions.SkipSignal):
      test_utils.run_resolver_op(ops.SkipIfLessThanNSpans, self.artifacts, n=4)

    with self.assertRaises(exceptions.SkipSignal):
      test_utils.run_resolver_op(ops.SkipIfLessThanNSpans, self.artifacts, n=5)

    with self.assertRaises(exceptions.SkipSignal):
      test_utils.run_resolver_op(ops.SkipIfLessThanNSpans, self.artifacts, n=10)

  def testSkipIfLessThanNSpans_OnNonEmpty_ReturnsAsIs(self):
    result = test_utils.run_resolver_op(
        ops.SkipIfLessThanNSpans, self.artifacts, n=3)
    self.assertEqual(result, self.artifacts)

    result = test_utils.run_resolver_op(
        ops.SkipIfLessThanNSpans, self.artifacts, n=2)
    self.assertEqual(result, self.artifacts)

    result = test_utils.run_resolver_op(
        ops.SkipIfLessThanNSpans, self.artifacts, n=1)
    self.assertEqual(result, self.artifacts)

    result = test_utils.run_resolver_op(
        ops.SkipIfLessThanNSpans, self.artifacts, n=0)
    self.assertEqual(result, self.artifacts)

    result = test_utils.run_resolver_op(
        ops.SkipIfLessThanNSpans, self.artifacts, n=-1)
    self.assertEqual(result, self.artifacts)

if __name__ == '__main__':
  tf.test.main()
