# Copyright 2023 Google LLC. All Rights Reserved.
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
"""Tests for tfx.dsl.input_resolution.ops.paired_spans_op."""

from typing import List, Sequence, Tuple

import tensorflow as tf
from tfx.dsl.input_resolution.ops import ops
from tfx.dsl.input_resolution.ops import test_utils


def _get_artifacts(
    spans_and_versions: List[Tuple[int, int]]
) -> Sequence[test_utils.DummyArtifact]:
  artifacts = []
  for span, version in spans_and_versions:
    art = test_utils.DummyArtifact()
    art.span = span
    art.version = version
    artifacts.append(art)
  return artifacts


class PairedSpansOpTest(tf.test.TestCase):

  def _paired_spans(self, *args, **kwargs):
    return test_utils.strict_run_resolver_op(
        ops.PairedSpans, args=args, kwargs=kwargs
    )

  def assertSpanVersion(self, artifact, span, version):
    self.assertEqual(artifact.span, span)
    self.assertEqual(artifact.version, version)

  def assertPairedVersion(self, artifact_dict, span, version) -> None:
    artifacts = [art[0] for art in artifact_dict.values()]
    self.assertTrue(all(x.span == span for x in artifacts))
    self.assertTrue(all(x.version == version for x in artifacts))

  def test_paired_spans(self):
    actual = self._paired_spans({
        'a': _get_artifacts([(0, 0), (1, 0)]),
        'b': _get_artifacts([(0, 0), (1, 0)]),
    })
    self.assertLen(actual, 2)
    self.assertPairedVersion(actual[0], 0, 0)
    self.assertPairedVersion(actual[1], 1, 0)

  def test_mismatched_span_latest_version(self):
    actual = self._paired_spans(
        {
            'a': _get_artifacts([(0, 0)]),
            'b': _get_artifacts([(0, 0), (0, 1)]),
        },
        keep_all_versions=False,
        match_version=True,
    )
    self.assertEmpty(actual)

  def test_mismatched_span_latest_version_allowed(self):
    actual = self._paired_spans(
        {
            'a': _get_artifacts([(0, 0)]),
            'b': _get_artifacts([(0, 0), (0, 1)]),
        },
        keep_all_versions=False,
        match_version=False,
    )
    self.assertLen(actual, 1)
    self.assertSpanVersion(actual[0]['a'][0], 0, 0)
    self.assertSpanVersion(actual[0]['b'][0], 0, 1)  # Picks latest.

  def test_no_common_keys(self):
    actual = self._paired_spans(
        {
            'a': _get_artifacts([(0, 0)]),
            'b': _get_artifacts([(1, 0)]),
        },
        keep_all_versions=False,
    )
    self.assertEmpty(actual)

  def test_all_versions(self):
    actual = self._paired_spans(
        {
            'a': _get_artifacts([(0, 0)]),
            'b': _get_artifacts([(0, 0), (0, 1)]),
        },
        match_version=True,
        keep_all_versions=True,
    )
    self.assertLen(actual, 1)
    self.assertPairedVersion(actual[0], 0, 0)

  def test_extra_span(self):
    actual = self._paired_spans(
        {
            'a': _get_artifacts([(0, 0), (1, 0)]),
            'b': _get_artifacts([(0, 0)]),
        },
        keep_all_versions=False,
    )
    self.assertLen(actual, 1)
    self.assertPairedVersion(actual[0], 0, 0)

  def test_three_inputs_all_versions(self):
    actual = self._paired_spans(
        {
            'a': _get_artifacts([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]),
            'b': _get_artifacts([
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
            ]),
            'c': _get_artifacts([(0, 0), (0, 1), (1, 1)]),
        },
        match_version=True,
        keep_all_versions=True,
    )
    self.assertLen(actual, 3)
    self.assertPairedVersion(actual[0], 0, 0)
    self.assertPairedVersion(actual[1], 0, 1)
    self.assertPairedVersion(actual[2], 1, 1)

  def test_three_inputs_latest_version(self):
    actual = self._paired_spans(
        {
            'a': _get_artifacts([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]),
            'b': _get_artifacts([
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
            ]),
            'c': _get_artifacts([(0, 0), (0, 1), (1, 1)]),
        },
        keep_all_versions=False,
    )
    self.assertLen(actual, 2)
    self.assertPairedVersion(actual[0], 0, 1)
    self.assertPairedVersion(actual[1], 1, 1)
