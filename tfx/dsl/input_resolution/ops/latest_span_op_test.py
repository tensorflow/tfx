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
"""Tests for tfx.dsl.input_resolution.ops.latest_span_op."""

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


class LatestSpanOpTest(tf.test.TestCase):

  def testLatestSpan_Empty(self):
    actual = test_utils.run_resolver_op(ops.LatestSpan, [])
    self.assertEqual(actual, [])

  def testLatestSpan_SingleEntry(self):
    a1 = test_utils.DummyArtifact()
    a1.span = 1

    actual = test_utils.run_resolver_op(ops.LatestSpan, [a1])
    self.assertEqual(actual, [a1])

  def testLatestSpan(self):
    a1 = test_utils.DummyArtifact()
    a2 = test_utils.DummyArtifact()
    a3 = test_utils.DummyArtifact()
    a4 = test_utils.DummyArtifact()
    a5 = ArtifactWithoutSpan()
    a6 = ArtifactWithoutSpanOrVersion()

    a1.span = 1
    a2.span = 2
    a3.span = 2
    a4.span = 3

    a1.version = 1
    a2.version = 1
    a3.version = 2
    a4.version = 1
    a5.version = 1

    artifacts = [a1, a2, a3, a4, a5, a6]

    actual = test_utils.run_resolver_op(ops.LatestSpan, artifacts, n=1)
    self.assertEqual(actual, [a4])

    actual = test_utils.run_resolver_op(ops.LatestSpan, artifacts, n=2)
    self.assertEqual(actual, [a4, a3])

    actual = test_utils.run_resolver_op(
        ops.LatestSpan, artifacts, n=2, keep_all_versions=True)
    self.assertEqual(actual, [a4, a2, a3])

    actual = test_utils.run_resolver_op(ops.LatestSpan, artifacts, n=3)
    self.assertEqual(actual, [a4, a3, a1])

    actual = test_utils.run_resolver_op(
        ops.LatestSpan, [a1, a2, a3, a4, a5], n=3, keep_all_versions=True)
    self.assertEqual(actual, [a4, a2, a3, a1])

    actual = test_utils.run_resolver_op(
        ops.LatestSpan, [a1, a2, a3, a4, a5], n=4)
    self.assertEqual(actual, [a4, a3, a1])

    actual = test_utils.run_resolver_op(
        ops.LatestSpan, [a1, a2, a3, a4, a5], n=4, keep_all_versions=True)
    self.assertEqual(actual, [a4, a2, a3, a1])

    actual = test_utils.run_resolver_op(
        ops.LatestSpan, [a1, a2, a3, a4, a5], keep_all_spans=True)
    self.assertEqual(actual, [a4, a3, a1])

    actual = test_utils.run_resolver_op(
        ops.LatestSpan, [a1, a2, a3, a4, a5],
        keep_all_spans=True,
        keep_all_versions=True)
    self.assertEqual(actual, [a4, a2, a3, a1])

  def testLatestSpan_AllSameSpanSameVersion(self):
    a1 = test_utils.DummyArtifact()
    a2 = test_utils.DummyArtifact()
    a3 = test_utils.DummyArtifact()

    a1.span = 1
    a2.span = 1
    a3.span = 1

    a1.version = 1
    a2.version = 1
    a3.version = 1

    a1.id = 1
    a2.id = 2
    a3.id = 3

    artifacts = [a1, a2, a3]

    actual = test_utils.run_resolver_op(ops.LatestSpan, artifacts, n=1)
    self.assertEqual(actual, [a3])

    actual = test_utils.run_resolver_op(
        ops.LatestSpan, artifacts, n=1, keep_all_versions=True)
    self.assertEqual(actual, [a1, a2, a3])

  def testLatestSpan_InvalidN(self):
    a1 = test_utils.DummyArtifact()

    with self.assertRaisesRegex(ValueError, 'n must be > 0'):
      test_utils.run_resolver_op(ops.LatestSpan, [a1], n=-1)

if __name__ == '__main__':
  tf.test.main()
