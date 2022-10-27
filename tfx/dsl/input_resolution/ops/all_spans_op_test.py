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
"""Tests for tfx.dsl.input_resolution.ops.all_spans_op."""

import tensorflow as tf

from tfx.dsl.input_resolution.ops import ops
from tfx.dsl.input_resolution.ops import test_utils


class AllSpansOpTest(tf.test.TestCase):

  def testAllSpans_OnEmpty_ReturnsEmpty(self):
    actual = test_utils.run_resolver_op(ops.AllSpans, [])
    self.assertEqual(actual, [])

  def testAllSpans_OnNonEmpty_ReturnsAllSortedSpans(self):
    a10 = test_utils.DummyArtifact()
    a20 = test_utils.DummyArtifact()
    a31 = test_utils.DummyArtifact()
    a30 = test_utils.DummyArtifact()
    a71 = test_utils.DummyArtifact()
    a82 = test_utils.DummyArtifact()

    artifacts = [a10, a20, a31, a30, a71, a82]

    spans = [1, 2, 3, 3, 7, 8]
    versions = [0, 0, 1, 0, 1, 2]
    for dummy_artifact, span, version in zip(artifacts, spans, versions):
      dummy_artifact.span = span
      dummy_artifact.version = version

    # Rotate the artifacts list so that it is not pre-sorted.
    for _ in range(4):
      artifacts.append(artifacts.pop(0))

    actual = test_utils.run_resolver_op(ops.AllSpans, artifacts)
    self.assertEqual(actual, [a10, a20, a31, a71, a82])

    actual = test_utils.run_resolver_op(
        ops.AllSpans, artifacts, span_descending=False, keep_all_versions=False)
    self.assertEqual(actual, [a10, a20, a31, a71, a82])

    actual = test_utils.run_resolver_op(
        ops.AllSpans, artifacts, span_descending=False, keep_all_versions=True)
    self.assertEqual(actual, [a10, a20, a30, a31, a71, a82])

    actual = test_utils.run_resolver_op(
        ops.AllSpans, artifacts, span_descending=True, keep_all_versions=False)
    self.assertEqual(actual, [a82, a71, a31, a20, a10])

    actual = test_utils.run_resolver_op(
        ops.AllSpans, artifacts, span_descending=True, keep_all_versions=True)
    self.assertEqual(actual, [a82, a71, a30, a31, a20, a10])


if __name__ == '__main__':
  tf.test.main()
