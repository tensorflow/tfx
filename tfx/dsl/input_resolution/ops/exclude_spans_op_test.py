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
"""Tests for tfx.dsl.input_resolution.ops.exclude_spans_op."""

import tensorflow as tf

from tfx import types
from tfx.dsl.input_resolution.ops import ops
from tfx.dsl.input_resolution.ops import test_utils


class ArtifactWithoutSpan(types.Artifact):
  """An Artifact without "span" as a PROPERTY."""
  TYPE_NAME = 'ArtifactWithoutSpan'


class ExcludeSpansOpTest(tf.test.TestCase):

  def testExcludeSpans_Empty(self):
    actual = test_utils.run_resolver_op(ops.ExcludeSpans, [])
    self.assertEqual(actual, [])

  def testExcludeSpans_SingleEntry(self):
    a1 = test_utils.DummyArtifact()
    a1.span = 1
    artifacts = [a1]

    actual = test_utils.run_resolver_op(ops.ExcludeSpans, artifacts)
    self.assertEqual(actual, [a1])

    actual = test_utils.run_resolver_op(
        ops.ExcludeSpans, artifacts, denylist=[1])
    self.assertEqual(actual, [])

    actual = test_utils.run_resolver_op(
        ops.ExcludeSpans, artifacts, denylist=[2])
    self.assertEqual(actual, [a1])

  def testExcludeSpans(self):
    a1 = test_utils.DummyArtifact()
    a2 = test_utils.DummyArtifact()
    a3 = test_utils.DummyArtifact()
    a4 = ArtifactWithoutSpan()

    a1.span = 1
    a2.span = 2
    a3.span = 2

    artifacts = [a1, a2, a3, a4]

    actual = test_utils.run_resolver_op(
        ops.ExcludeSpans, artifacts, denylist=[])
    self.assertEqual(actual, [a1, a2, a3])

    actual = test_utils.run_resolver_op(
        ops.ExcludeSpans, artifacts, denylist=[1])
    self.assertEqual(actual, [a2, a3])

    actual = test_utils.run_resolver_op(
        ops.ExcludeSpans, artifacts, denylist=[2])
    self.assertEqual(actual, [a1])

    actual = test_utils.run_resolver_op(
        ops.ExcludeSpans, artifacts, denylist=[1, 2])
    self.assertEqual(actual, [])


if __name__ == '__main__':
  tf.test.main()
