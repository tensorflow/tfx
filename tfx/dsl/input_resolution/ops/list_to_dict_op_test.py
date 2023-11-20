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
"""Tests for tfx.dsl.input_resolution.ops.list_to_dict_op."""

import tensorflow as tf
from tfx.dsl.input_resolution.ops import ops
from tfx.dsl.input_resolution.ops import test_utils


class ListToDictOpTest(tf.test.TestCase):

  def _list_to_dict(self, *args, **kwargs):
    return test_utils.strict_run_resolver_op(
        ops.ListToDict, args=args, kwargs=kwargs
    )

  def testListToDict_Empty(self):
    actual = self._list_to_dict([])
    self.assertEqual(actual, {})

  def testListToDict_SingleEntry(self):
    a0 = test_utils.DummyArtifact(id=0, create_time_since_epoch=1)
    actual = self._list_to_dict([a0])
    self.assertEqual(actual, {'0': [a0]})

  def testListToDict_MultipleEntries(self):
    a0 = test_utils.DummyArtifact(id=0, create_time_since_epoch=5)
    a1 = test_utils.DummyArtifact(id=1, create_time_since_epoch=10)
    a2 = test_utils.DummyArtifact(id=2, create_time_since_epoch=15)

    with self.subTest('Artifacts in order'):
      actual = self._list_to_dict([a0, a1, a2])
      self.assertEqual(actual, {'0': [a0], '1': [a1], '2': [a2]})

    with self.subTest('Artifacts out of order'):
      actual = self._list_to_dict([a2, a0, a1])
      self.assertEqual(actual, {'0': [a2], '1': [a0], '2': [a1]})

    with self.subTest('n < 3'):
      actual = self._list_to_dict([a0, a1, a2], n=3)
      self.assertEqual(actual, {'0': [a0], '1': [a1], '2': [a2]})

    with self.subTest('n == 3'):
      actual = self._list_to_dict([a0, a1, a2], n=2)
      self.assertEqual(actual, {'0': [a0], '1': [a1]})

    with self.subTest('n > 3'):
      actual = self._list_to_dict([a0, a1, a2], n=4)
      self.assertEqual(actual, {'0': [a0], '1': [a1], '2': [a2], '3': []})

    with self.subTest('n == 0'):
      actual = self._list_to_dict([a0, a1, a2], n=0)
      self.assertEqual(actual, {})


if __name__ == '__main__':
  tf.test.main()
