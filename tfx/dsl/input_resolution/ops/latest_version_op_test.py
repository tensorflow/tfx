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
"""Tests for tfx.dsl.input_resolution.ops.latest_version_op."""

import tensorflow as tf

from tfx import types
from tfx.dsl.input_resolution.ops import ops
from tfx.dsl.input_resolution.ops import test_utils


class ArtifactWithoutVersion(types.Artifact):
  """An Artifact without "version" as a PROPERTY."""

  TYPE_NAME = 'ArtifactWithoutVersion'


class LatestVersionOpTest(tf.test.TestCase):

  def _latest_version(self, *args, **kwargs):
    return test_utils.strict_run_resolver_op(
        ops.LatestVersion, args=args, kwargs=kwargs
    )

  def testLatestVersion_Empty(self):
    actual = self._latest_version([])
    self.assertEqual(actual, [])

  def testLatestVersion_SingleEntry(self):
    a1 = test_utils.DummyArtifact()
    a1.version = 1

    actual = self._latest_version([a1])
    self.assertEqual(actual, [a1])

  def testLatestVersion_SameSpan(self):
    a1 = test_utils.DummyArtifact()
    a2 = test_utils.DummyArtifact()
    a3 = test_utils.DummyArtifact()
    a4 = ArtifactWithoutVersion()

    a1.id = 1
    a2.id = 2
    a3.id = 3

    a1.version = 1
    a2.version = 2
    a3.version = 2

    artifacts = [a1, a3, a2, a4]

    actual = self._latest_version(artifacts)
    self.assertEqual(actual, [a3])

    actual = self._latest_version(artifacts, n=2)
    self.assertEqual(actual, [a2, a3])

    actual = self._latest_version(artifacts, n=3)
    self.assertEqual(actual, [a1, a2, a3])

    # Although n = 4, only 3 artifacts are returned because only 3 are
    # available.
    actual = self._latest_version(artifacts, n=4)
    self.assertEqual(actual, [a1, a2, a3])

  def testLatestVersion_DifferentSpans(self):
    a10 = test_utils.DummyArtifact()
    a11 = test_utils.DummyArtifact()
    a20 = test_utils.DummyArtifact()
    a21 = test_utils.DummyArtifact()

    a10.span = 1
    a11.span = 1
    a20.span = 2
    a21.span = 2

    a10.version = 0
    a11.version = 1
    a20.version = 0
    a21.version = 1

    artifacts = [a10, a20, a21, a11]

    actual = self._latest_version(artifacts)
    self.assertEqual(actual, [a21])

    actual = self._latest_version(artifacts, n=2)
    self.assertEqual(actual, [a20, a21])

    actual = self._latest_version(artifacts, n=3)
    self.assertEqual(actual, [a11, a20, a21])

    actual = self._latest_version(artifacts, n=4)
    self.assertEqual(actual, [a10, a11, a20, a21])

  def testLatestSpan_InvalidN(self):
    a1 = test_utils.DummyArtifact()
    a1.version = 1

    with self.assertRaisesRegex(ValueError, 'n must be > 0'):
      self._latest_version([a1], n=-1)


if __name__ == '__main__':
  tf.test.main()
