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

  def testLatestVersion_Empty(self):
    actual = test_utils.run_resolver_op(ops.LatestVersion, [])
    self.assertEqual(actual, [])

  def testLatestVersion_SingleEntry(self):
    a1 = test_utils.DummyArtifact()
    a1.version = 1

    actual = test_utils.run_resolver_op(ops.LatestVersion, [a1])
    self.assertEqual(actual, [a1])

  def testLatestVersion(self):
    a1 = test_utils.DummyArtifact()
    a2 = test_utils.DummyArtifact()
    a3 = test_utils.DummyArtifact()
    a4 = ArtifactWithoutVersion()

    a1.version = 1
    a2.version = 2
    a3.version = 3

    artifacts = [a1, a3, a2, a4]

    actual = test_utils.run_resolver_op(ops.LatestVersion, artifacts)
    self.assertEqual(actual, [a3])

    actual = test_utils.run_resolver_op(ops.LatestVersion, artifacts, n=2)
    self.assertEqual(actual, [a3, a2])

    actual = test_utils.run_resolver_op(
        ops.LatestVersion, artifacts, keep_all=True)
    self.assertEqual(actual, [a3, a2, a1])


if __name__ == '__main__':
  tf.test.main()
