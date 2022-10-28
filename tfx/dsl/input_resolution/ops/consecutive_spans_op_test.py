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
"""Tests for tfx.dsl.input_resolution.ops.consecutive_spans_op."""

from typing import Sequence

import tensorflow as tf

from tfx import types
from tfx.dsl.input_resolution.ops import ops
from tfx.dsl.input_resolution.ops import test_utils
from tfx.types import artifact


class ArtifactWithoutSpanOrVersion(types.Artifact):
  """An Artifact without "span" or "version" as a PROPERTY."""
  TYPE_NAME = 'ArtifactWithoutSpanOrVersion'


class ArtifactWithoutSpan(types.Artifact):
  """An Artifact without "span" as a PROPERTY."""
  TYPE_NAME = 'ArtifactWithoutSpan'

  PROPERTIES = {
      'version': artifact.Property(type=artifact.PropertyType.INT),
  }


class ConsecutiveSpansOpTest(tf.test.TestCase):

  def _get_artifacts_for_sequential_rolling_range_tests(
      self) -> Sequence[types.Artifact]:
    a10 = test_utils.DummyArtifact()
    a20 = test_utils.DummyArtifact()
    a31 = test_utils.DummyArtifact()
    a30 = test_utils.DummyArtifact()
    a40 = test_utils.DummyArtifact()
    a50 = test_utils.DummyArtifact()

    # There is a gap in the artifact spans, with a span of 6 missing.
    # ConsecutiveSpans should only ever return at most artifacts
    # [a10, a20, a30, a40, a50].
    a70 = test_utils.DummyArtifact()
    a80 = test_utils.DummyArtifact()

    artifacts = [a10, a20, a31, a30, a40, a50, a70, a80]

    spans = [1, 2, 3, 3, 4, 5, 7, 8]
    versions = [0, 0, 1, 0, 0, 0, 0, 0]
    for dummy_artifact, span, version in zip(artifacts, spans, versions):
      dummy_artifact.span = span
      dummy_artifact.version = version

    return artifacts

  def testConsecutiveSpans_Empty(self):
    actual = test_utils.run_resolver_op(ops.ConsecutiveSpans, [])
    self.assertEqual(actual, [])

  def testConsecutiveSpans_SingleEntry(self):
    a1 = test_utils.DummyArtifact()
    a1.span = 1

    actual = test_utils.run_resolver_op(ops.ConsecutiveSpans, [a1])
    self.assertEqual(actual, [a1])

  def testConsecutiveSpans_ArtifactsWithoutSpanAndVersion(self):
    a11 = test_utils.DummyArtifact()
    a21 = test_utils.DummyArtifact()
    a31 = test_utils.DummyArtifact()
    a_1 = ArtifactWithoutSpan()
    a__ = ArtifactWithoutSpanOrVersion()

    a11.span = 1
    a21.span = 2
    a31.span = 3

    a11.version = 1
    a21.version = 1
    a31.version = 1
    a_1.version = 1

    artifacts = [a11, a21, a31, a_1, a__]

    actual = test_utils.run_resolver_op(ops.ConsecutiveSpans, artifacts)
    self.assertEqual(actual, [a11, a21, a31])

  def testConsecutiveSpans_AllSameSpanSameVersion(self):
    a1 = test_utils.DummyArtifact()
    a2 = test_utils.DummyArtifact()
    a3 = test_utils.DummyArtifact()

    a1.span = 1
    a2.span = 1
    a3.span = 1

    a1.version = 1
    a2.version = 1
    a3.version = 1

    # The tie should be broken by the id.
    a1.id = 1
    a2.id = 2
    a3.id = 3

    artifacts = [a1, a2, a3]

    actual = test_utils.run_resolver_op(ops.ConsecutiveSpans, artifacts)
    self.assertEqual(actual, [a3])

    actual = test_utils.run_resolver_op(
        ops.ConsecutiveSpans, artifacts, keep_all_versions=True)
    self.assertEqual(actual, [a1, a2, a3])

  def testConsecutiveSpans_MultipleGaps(self):
    a1 = test_utils.DummyArtifact()
    a3 = test_utils.DummyArtifact()
    a5 = test_utils.DummyArtifact()

    a1.span = 1
    a3.span = 3
    a5.span = 5

    artifacts = [a5, a1, a3]

    actual = test_utils.run_resolver_op(ops.ConsecutiveSpans, artifacts)
    self.assertEqual(actual, [a1])

  def testConsecutiveSpans_SkipLastN(self):
    artifacts = self._get_artifacts_for_sequential_rolling_range_tests()
    a10, a20, a31, a30, a40, a50, _, _ = artifacts

    actual = test_utils.run_resolver_op(ops.ConsecutiveSpans, artifacts)
    self.assertEqual(actual, [a10, a20, a31, a40, a50])

    actual = test_utils.run_resolver_op(
        ops.ConsecutiveSpans, artifacts, skip_last_n=1)
    self.assertEqual(actual, [a10, a20, a31, a40, a50])

    actual = test_utils.run_resolver_op(
        ops.ConsecutiveSpans, artifacts, skip_last_n=2)
    self.assertEqual(actual, [a10, a20, a31, a40, a50])

    actual = test_utils.run_resolver_op(
        ops.ConsecutiveSpans, artifacts, skip_last_n=3)
    self.assertEqual(actual, [a10, a20, a31, a40, a50])

    actual = test_utils.run_resolver_op(
        ops.ConsecutiveSpans, artifacts, skip_last_n=4)
    self.assertEqual(actual, [a10, a20, a31, a40])

    actual = test_utils.run_resolver_op(
        ops.ConsecutiveSpans, artifacts, skip_last_n=5)
    self.assertEqual(actual, [a10, a20, a31])

    actual = test_utils.run_resolver_op(
        ops.ConsecutiveSpans, artifacts, skip_last_n=6)
    self.assertEqual(actual, [a10, a20])

    actual = test_utils.run_resolver_op(
        ops.ConsecutiveSpans, artifacts, skip_last_n=7)
    self.assertEqual(actual, [a10])

    actual = test_utils.run_resolver_op(
        ops.ConsecutiveSpans, artifacts, skip_last_n=8)
    self.assertEqual(actual, [])

    # Tests version conflicts when keep_all_versions=True.
    actual = test_utils.run_resolver_op(
        ops.ConsecutiveSpans, artifacts, keep_all_versions=True)
    self.assertEqual(actual, [a10, a20, a30, a31, a40, a50])

    actual = test_utils.run_resolver_op(
        ops.ConsecutiveSpans, artifacts, skip_last_n=5, keep_all_versions=True)
    self.assertEqual(actual, [a10, a20, a30, a31])

    actual = test_utils.run_resolver_op(
        ops.ConsecutiveSpans, artifacts, skip_last_n=6, keep_all_versions=True)
    self.assertEqual(actual, [a10, a20])

    # skip_last_n=9 is greater than the largest spans availble (8), so an
    # invalid range [0, -1] is created and no artifacts are returned.
    actual = test_utils.run_resolver_op(
        ops.ConsecutiveSpans, artifacts, skip_last_n=9)
    self.assertEqual(actual, [])

  def testConsecutiveSpans_FirstSpan(self):
    artifacts = self._get_artifacts_for_sequential_rolling_range_tests()
    a10, a20, a31, a30, a40, a50, _, _ = artifacts

    actual = test_utils.run_resolver_op(
        ops.ConsecutiveSpans, artifacts, first_span=1)
    self.assertEqual(actual, [a10, a20, a31, a40, a50])

    actual = test_utils.run_resolver_op(
        ops.ConsecutiveSpans, artifacts, first_span=2)
    self.assertEqual(actual, [a20, a31, a40, a50])

    actual = test_utils.run_resolver_op(
        ops.ConsecutiveSpans, artifacts, first_span=3)
    self.assertEqual(actual, [a31, a40, a50])

    actual = test_utils.run_resolver_op(
        ops.ConsecutiveSpans, artifacts, first_span=4)
    self.assertEqual(actual, [a40, a50])

    actual = test_utils.run_resolver_op(
        ops.ConsecutiveSpans, artifacts, first_span=5)
    self.assertEqual(actual, [a50])

    # first_span=6 is greater than the largest spans availble (5), so an invalid
    # [6, 5] is created and no artifacts are returned.
    actual = test_utils.run_resolver_op(
        ops.ConsecutiveSpans, artifacts, first_span=6)
    self.assertEqual(actual, [])

    # Tests version conflicts when keep_all_versions=True.
    actual = test_utils.run_resolver_op(
        ops.ConsecutiveSpans, artifacts, first_span=3, keep_all_versions=True)
    self.assertEqual(actual, [a30, a31, a40, a50])

    actual = test_utils.run_resolver_op(
        ops.ConsecutiveSpans, artifacts, first_span=4, keep_all_versions=True)
    self.assertEqual(actual, [a40, a50])

  def testConsecutiveSpans_Denylist(self):
    artifacts = self._get_artifacts_for_sequential_rolling_range_tests()
    a10, a20, a31, _, a40, a50, _, _ = artifacts

    actual = test_utils.run_resolver_op(
        ops.ConsecutiveSpans, artifacts, first_span=1, skip_last_n=0)
    self.assertEqual(actual, [a10, a20, a31, a40, a50])

    actual = test_utils.run_resolver_op(
        ops.ConsecutiveSpans,
        artifacts,
        first_span=1,
        skip_last_n=0,
        denylist=[1])
    self.assertEqual(actual, [a20, a31, a40, a50])

    actual = test_utils.run_resolver_op(
        ops.ConsecutiveSpans,
        artifacts,
        first_span=1,
        skip_last_n=0,
        denylist=[1, 2])
    self.assertEqual(actual, [a31, a40, a50])

    actual = test_utils.run_resolver_op(
        ops.ConsecutiveSpans,
        artifacts,
        first_span=1,
        skip_last_n=0,
        denylist=[1, 3])
    self.assertEqual(actual, [a20])

    actual = test_utils.run_resolver_op(
        ops.ConsecutiveSpans,
        artifacts,
        first_span=1,
        skip_last_n=0,
        denylist=[1, 2, 5])
    self.assertEqual(actual, [a31, a40])

    actual = test_utils.run_resolver_op(
        ops.ConsecutiveSpans,
        artifacts,
        first_span=1,
        skip_last_n=5,
        denylist=[1, 2, 5])
    self.assertEqual(actual, [a31])

    actual = test_utils.run_resolver_op(
        ops.ConsecutiveSpans,
        artifacts,
        first_span=3,
        skip_last_n=5,
        denylist=[3])
    self.assertEqual(actual, [])

    actual = test_utils.run_resolver_op(
        ops.ConsecutiveSpans,
        artifacts,
        first_span=1,
        skip_last_n=0,
        denylist=[3])
    self.assertEqual(actual, [a10, a20])

  def testConsecutiveSpans_SmallValidSpanRange(self):
    artifacts = self._get_artifacts_for_sequential_rolling_range_tests()
    _, _, a31, a30, _, _, _, _ = artifacts

    actual = test_utils.run_resolver_op(
        ops.ConsecutiveSpans,
        artifacts,
        first_span=3,
        skip_last_n=5,
        keep_all_versions=True)
    self.assertEqual(actual, [a30, a31])

    # The arguments lead to the invalid spans range [3, 0], so no artifacts are
    # returned.
    actual = test_utils.run_resolver_op(
        ops.ConsecutiveSpans,
        artifacts,
        first_span=3,
        skip_last_n=8,
        keep_all_versions=True)
    self.assertEqual(actual, [])


if __name__ == '__main__':
  tf.test.main()
