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
"""Tests for tfx.dsl.input_resolution.ops.shuffle_op."""

import tensorflow as tf

from tfx.dsl.input_resolution.ops import ops
from tfx.dsl.input_resolution.ops import test_utils


class ShuffleOpTest(tf.test.TestCase):

  def _shuffle(self, *args, **kwargs):
    return test_utils.strict_run_resolver_op(
        ops.Shuffle, args=args, kwargs=kwargs
    )

  def testShuffle(self):
    a10 = test_utils.DummyArtifact()
    a20 = test_utils.DummyArtifact()
    a31 = test_utils.DummyArtifact()
    a30 = test_utils.DummyArtifact()
    a40 = test_utils.DummyArtifact()
    a50 = test_utils.DummyArtifact()

    artifacts = [a10, a20, a31, a30, a40, a50]

    spans = [1, 2, 3, 3, 4, 5]
    versions = [0, 0, 1, 0, 0, 0]
    for dummy_artifact, span, version in zip(artifacts, spans, versions):
      dummy_artifact.span = span
      dummy_artifact.version = version

    idxs = [1, 4, 3, 2, 0, 5]
    artifacts = [artifacts[i] for i in idxs]

    actual = self._shuffle(artifacts)
    self.assertCountEqual(actual, artifacts)

  def testShuffle_NoArtifacts(self):
    actual = self._shuffle([])
    self.assertEqual(actual, [])


if __name__ == '__main__':
  tf.test.main()
