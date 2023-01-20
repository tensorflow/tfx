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
"""Tests for tfx.dsl.input_resolution.ops.filter_artifacts_op."""

from unittest import mock

import tensorflow as tf
from tfx.dsl.input_resolution import resolver_op
from tfx.dsl.input_resolution.ops import filter_artifacts_op
from tfx.dsl.input_resolution.ops import test_utils
from tfx.dsl.placeholder import placeholder as ph
from tfx.orchestration.portable.input_resolution import exceptions
from tfx.proto.orchestration import placeholder_pb2
from tfx.types import channel as channel_types


class FilterArtifactsOpTest(tf.test.TestCase):

  def _create_dummy_artifact(self, span, version=0):
    result = test_utils.DummyArtifact()
    result.span = span
    if version:
      result.version = version
    return result

  def _filter_artifacts(self, artifacts, predicate_fn):
    return test_utils.run_resolver_op(
        filter_artifacts_op.FilterArtifactsInternal,
        artifacts,
        context=resolver_op.Context(store=mock.MagicMock()),
        encoded_predicate=filter_artifacts_op._encode_predicate(predicate_fn),
    )

  def testFacade(self):
    op_node = filter_artifacts_op.FilterArtifacts(
        resolver_op.InputNode(channel_types.Channel(test_utils.DummyArtifact)),
        predicate_fn=lambda a: a.property('span') == 42,
    )
    self.assertIsInstance(op_node, resolver_op.OpNode)
    self.assertEqual(
        op_node.output_data_type, resolver_op.DataType.ARTIFACT_LIST
    )
    self.assertIsInstance(
        op_node.kwargs['encoded_predicate'],
        placeholder_pb2.PlaceholderExpression,
    )

  def testFilterArtifacts_DummySpans(self):
    # pylint: disable=g-long-lambda
    inputs = []
    for span in range(10):
      for version in range(3):
        inputs.append(self._create_dummy_artifact(span, version))

    with self.subTest('property(span) < 5'):
      result = self._filter_artifacts(
          inputs,
          predicate_fn=lambda a: a.property('span') < 5,
      )
      self.assertLen(result, 15)
      for artifact in result:
        self.assertLess(artifact.span, 5)

    with self.subTest('5 <= property(span) < 8'):
      result = self._filter_artifacts(
          inputs,
          predicate_fn=lambda a: ph.logical_and(
              a.property('span') >= 5, a.property('span') < 8
          ),
      )
      self.assertLen(result, 9)
      for artifact in result:
        self.assertGreaterEqual(artifact.span, 5)
        self.assertLess(artifact.span, 8)

    with self.subTest('property(span) + property(version) = 7'):
      result = self._filter_artifacts(
          inputs,
          predicate_fn=lambda a: ph.logical_or(
              ph.logical_and(
                  a.property('span') == 5, a.property('version') == 2
              ),
              ph.logical_or(
                  ph.logical_and(
                      a.property('span') == 6, a.property('version') == 1
                  ),
                  ph.logical_and(
                      a.property('span') == 7, a.property('version') == 0
                  ),
              ),
          ),
      )
      self.assertLen(result, 3)
      for artifact in result:
        self.assertEqual(artifact.span + artifact.version, 7)

  def testFilterArtifacts_EmptyArtifacts(self):
    self.assertEmpty(
        self._filter_artifacts(
            [], predicate_fn=lambda a: a.property('span') == 42
        )
    )

  def testFilterArtifacts_InvalidPredicate(self):
    a1 = self._create_dummy_artifact(span=1)
    a2 = self._create_dummy_artifact(span=2)
    a3 = self._create_dummy_artifact(span=3)

    with self.assertRaisesRegex(
        TypeError, 'predicate_fn does not return a placeholder.Predicate'
    ):
      self._filter_artifacts([a1, a2, a3], predicate_fn=lambda a: True)

    with self.assertRaisesRegex(
        TypeError, 'predicate_fn does not return a placeholder.Predicate'
    ):
      self._filter_artifacts(
          [a1, a2, a3], predicate_fn=lambda a: a.property('span')
      )

    with self.subTest('Type checking is not done.'):
      self.assertEmpty(
          self._filter_artifacts(
              [a1, a2, a3], predicate_fn=lambda a: a.property('span') == 'str'
          )
      )

    with self.assertRaisesRegex(
        exceptions.InputResolutionError,
        'Failed to resolve placeholder expression: '
        r'\(input\("value"\)\[0\].property\("non-existing"\) == "42"\)'):
      self._filter_artifacts(
          [a1, a2, a3], predicate_fn=lambda a: a.property('non-existing') == 42
      )


if __name__ == '__main__':
  tf.test.main()
