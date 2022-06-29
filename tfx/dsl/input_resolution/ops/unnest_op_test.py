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
"""Tests for tfx.dsl.input_resolution.ops.unnest_op."""

import tensorflow as tf
from tfx.dsl.input_resolution import resolver_function
from tfx.dsl.input_resolution import resolver_op
from tfx.dsl.input_resolution.ops import ops
from tfx.dsl.input_resolution.ops import test_utils
from tfx.orchestration.portable.input_resolution import exceptions


class UnnestOpTest(tf.test.TestCase):

  def create_artifacts(self, uri_prefix: str, n: int):
    return [
        test_utils.DummyArtifact(uri=uri_prefix + str(i))
        for i in range(1, n + 1)
    ]

  def testUnnest_KeyIsRequired(self):

    @resolver_function.resolver_function
    def f(root):
      return ops.Unnest(root)

    input_node = resolver_op.InputNode(
        None, resolver_op.DataType.ARTIFACT_MULTIMAP)

    with self.assertRaisesRegex(ValueError, 'Required property key is missing'):
      f.trace(input_node)

  def testUnnest_KeyChannel_Unnested(self):
    [x1, x2, x3] = self.create_artifacts(uri_prefix='x/', n=3)
    input_dict = {'x': [x1, x2, x3]}

    result = test_utils.run_resolver_op(ops.Unnest, input_dict, key='x')

    self.assertEqual(result, [{'x': [x1]}, {'x': [x2]}, {'x': [x3]}])

  def testUnnest_NonKeyChannel_IsNotUnnested(self):
    [x1, x2, x3] = self.create_artifacts(uri_prefix='x/', n=3)
    ys = self.create_artifacts(uri_prefix='y/', n=2)
    input_dict = {'x': [x1, x2, x3], 'y': ys}

    result = test_utils.run_resolver_op(ops.Unnest, input_dict, key='x')

    self.assertEqual(result, [
        {'x': [x1], 'y': ys},
        {'x': [x2], 'y': ys},
        {'x': [x3], 'y': ys},
    ])

  def testUnnest_NonExistingKey(self):
    [x1, x2, x3] = self.create_artifacts(uri_prefix='x/', n=3)
    input_dict = {'x': [x1, x2, x3]}

    with self.assertRaisesRegex(
        exceptions.FailedPreconditionError,
        'Input dict does not contain the key y.',
    ):
      test_utils.run_resolver_op(ops.Unnest, input_dict, key='y')

  def testUnnest_EmptyChannel_ReturnsEmptyList(self):
    input_dict = {'x': []}

    result = test_utils.run_resolver_op(ops.Unnest, input_dict, key='x')

    self.assertEmpty(result)


if __name__ == '__main__':
  tf.test.main()
