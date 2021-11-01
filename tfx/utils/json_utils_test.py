# Copyright 2019 Google LLC. All Rights Reserved.
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
"""Tests for tfx.utils.json_utils."""

import tensorflow as tf
from tfx.proto import trainer_pb2
from tfx.utils import deprecation_utils
from tfx.utils import json_utils


class _DefaultJsonableObject(json_utils.Jsonable):

  def __init__(self, a, b, c):
    self.a = a
    self.b = b
    self.c = c


_DeprecatedAlias = deprecation_utils.deprecated_alias(
    deprecated_name='_DeprecatedAlias',
    name='_DefaultJsonableObject',
    func_or_class=_DefaultJsonableObject)


class JsonUtilsTest(tf.test.TestCase):

  def testDumpsJsonableObjectRoundtrip(self):
    obj = _DefaultJsonableObject(1, {'a': 'b'}, [True])

    json_text = json_utils.dumps(obj)

    actual_obj = json_utils.loads(json_text)
    self.assertEqual(1, actual_obj.a)
    self.assertDictEqual({'a': 'b'}, actual_obj.b)
    self.assertCountEqual([True], actual_obj.c)

  def testDumpsNestedJsonableObject(self):
    nested_obj = _DefaultJsonableObject(1, 2,
                                        trainer_pb2.TrainArgs(num_steps=100))
    obj = _DefaultJsonableObject(nested_obj, None, None)

    json_text = json_utils.dumps(obj)

    actual_obj = json_utils.loads(json_text)
    self.assertEqual(1, actual_obj.a.a)
    self.assertEqual(2, actual_obj.a.b)
    self.assertProtoEquals(trainer_pb2.TrainArgs(num_steps=100), actual_obj.a.c)
    self.assertIsNone(actual_obj.b)
    self.assertIsNone(actual_obj.c)

  def testDumpsNestedClass(self):
    obj = _DefaultJsonableObject(_DefaultJsonableObject, None, None)

    json_text = json_utils.dumps(obj)

    actual_obj = json_utils.loads(json_text)
    self.assertEqual(_DefaultJsonableObject, actual_obj.a)
    self.assertIsNone(actual_obj.b)
    self.assertIsNone(actual_obj.c)

  def testDumpsClass(self):
    json_text = json_utils.dumps(_DefaultJsonableObject)

    actual_obj = json_utils.loads(json_text)
    self.assertEqual(_DefaultJsonableObject, actual_obj)

  def testDumpsDeprecatedClass(self):
    json_text = json_utils.dumps(_DeprecatedAlias)

    actual_obj = json_utils.loads(json_text)
    self.assertEqual(_DefaultJsonableObject, actual_obj)


if __name__ == '__main__':
  tf.test.main()
