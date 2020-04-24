# Lint as: python2, python3
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from google.protobuf import struct_pb2
from tfx.proto import trainer_pb2
from tfx.utils import json_utils


class _DefaultJsonableObject(json_utils.Jsonable):

  def __init__(self, a, b, c):
    self.a = a
    self.b = b
    self.c = c


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


class StructUtilsTest(tf.test.TestCase):

  def testStruct_WithDict(self):
    s = json_utils.Struct({'hello': 'world'})

    self.assertEqual(s, struct_pb2.Struct(
        fields={
            'hello': struct_pb2.Value(
                string_value='world'
            )
        }
    ))

  def testStruct_WithKwargs(self):
    s = json_utils.Struct(hello='world')

    self.assertEqual(s, struct_pb2.Struct(
        fields={
            'hello': struct_pb2.Value(
                string_value='world'
            )
        }
    ))

  def testStruct_NestedStruct(self):
    s = json_utils.Struct(
        nested=json_utils.Struct(
            hello='world'
        )
    )

    self.assertEqual(s, struct_pb2.Struct(
        fields={
            'nested': struct_pb2.Value(
                struct_value=struct_pb2.Struct(
                    fields={
                        'hello': struct_pb2.Value(
                            string_value='world'
                        )
                    }
                )
            )
        }
    ))

  def testStructToDict_FloatToInteger(self):
    s = json_utils.Struct(
        int=123,
        float_int=123.0,
        float=123.4)

    d = json_utils.struct_to_dict(s)
    self.assertEqual(d, {
        'int': 123,        # Integer.
        'float_int': 123,  # Also integer.
        'float': 123.4     # Float is float.
    })

  def testRoundTrip_StructDictStruct(self):
    s1 = json_utils.Struct(
        number=123,
        text='abc',
        boolean=True,
        nullable=None,
        list=[1, 'x', False],
        object={
            'hello': 'world',
            'empty_object': {}
        }
    )
    s2 = json_utils.Struct(json_utils.struct_to_dict(s1))

    self.assertEqual(s1, s2)

  def testRoundTrip_DictStructDict(self):
    d1 = {
        'number': 123,
        'text': 'abc',
        'boolean': True,
        'nullable': None,
        'list': [1, 'x', False],
        'object': {
            'hello': 'world',
            'empty_object': {}
        }
    }
    d2 = json_utils.struct_to_dict(json_utils.Struct(d1))

    self.assertEqual(d1, d2)


if __name__ == '__main__':
  tf.test.main()
