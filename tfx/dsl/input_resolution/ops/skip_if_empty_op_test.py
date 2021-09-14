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
"""Tests for tfx.dsl.input_resolution.ops.skip_if_empty_op."""

import tensorflow as tf

from tfx.dsl.input_resolution import resolver_function
from tfx.dsl.input_resolution.ops import skip_if_empty_op
from tfx.dsl.input_resolution.ops import test_utils
from tfx.dsl.input_resolution.ops import unnest_op
from tfx.orchestration.portable.input_resolution import exceptions


class SkipIfEmptyOpTest(tf.test.TestCase):

  def create_artifacts(self, uri_prefix: str, n: int):
    return [
        test_utils.create_dummy_artifact(uri=uri_prefix + str(i))
        for i in range(1, n + 1)
    ]

  def testSkipIfEmpty_OnEmpty_RaisesSkipSignal(self):

    @resolver_function.resolver_function
    def f(root):
      items = unnest_op.Unnest(root, key='x')
      return skip_if_empty_op.SkipIfEmpty(items)

    with self.assertRaises(exceptions.SkipSignal):
      test_utils.run_resolver_function(f, {'x': []})

  def testSkipIfEmpty_OnNonEmpty_ReturnsAsIs(self):

    @resolver_function.resolver_function
    def f(root):
      items = unnest_op.Unnest(root, key='x')
      return skip_if_empty_op.SkipIfEmpty(items)

    x1, x2, x3 = self.create_artifacts(uri_prefix='x/', n=3)

    result = test_utils.run_resolver_function(f, {
        'x': [x1, x2, x3],
    })
    self.assertEqual(result, [{'x': [x1]}, {'x': [x2]}, {'x': [x3]}])


if __name__ == '__main__':
  tf.test.main()
