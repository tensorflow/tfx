# Copyright 2020 Google LLC. All Rights Reserved.
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
"""Tests for standard TFX Artifact types."""

import math
from typing import Any, Dict
from unittest import mock

import absl
import tensorflow as tf
from tfx.types import standard_artifacts
from tfx.utils import json_utils

# Define constant value for tests.
_TEST_BYTE_RAW = b'hello world'
_TEST_BYTE_DECODED = b'hello world'

_TEST_STRING_RAW = b'hello world'
_TEST_STRING_DECODED = u'hello world'

_TEST_BOOL_RAW = b'1'
_TEST_BOOL_DECODED = True

_TEST_INT_RAW = b'19260817'
_TEST_INT_DECODED = 19260817

_TEST_FLOAT_RAW = b'3.1415926535'
_TEST_FLOAT_DECODED = 3.1415926535

_TEST_FLOAT128_RAW = b'3.14159265358979323846264338327950288'
_TEST_FLOAT128 = 3.14159265358979323846264338327950288  # Too precise

_TEST_JSONVALUE_LIST_RAW = '[42, 42.0]'
_TEST_JSONVALUE_LIST_DECODED = [42, 42.0]

_TEST_JSONVALUE_DICT_RAW = '{\"x\": 42}'
_TEST_JSONVALUE_DICT_DECODED = {'x': 42}


class TestJsonableCls(json_utils.Jsonable):
  """A test class that implements the Jsonable interface."""

  def __init__(self, x):
    self._x = x

  def to_json_dict(self) -> Dict[str, Any]:
    return {'x': self._x}

  @classmethod
  def from_json_dict(cls, dict_data: Dict[str, Any]) -> 'TestJsonableCls':
    return TestJsonableCls(dict_data['x'])

  def __eq__(self, other):
    return isinstance(other, TestJsonableCls) and other._x == self._x


_TEST_JSONVALUE_OBJ_RAW = (
    '{\"__class__\": \"TestJsonableCls\", \"__module__\":'
    ' \"__main__\", \"__tfx_object_type__\": '
    '\"jsonable\", \"x\": 42}')
_TEST_JSONVALUE_OBJ_DECODED = TestJsonableCls(42)


class StandardArtifactsTest(tf.test.TestCase):

  def testUseTfxType(self):
    instance = standard_artifacts.ExampleStatistics()
    self.assertIsInstance(instance, standard_artifacts.ExampleStatistics)

  def testBytesType(self):
    instance = standard_artifacts.Bytes()
    self.assertEqual(_TEST_BYTE_RAW, instance.encode(_TEST_BYTE_DECODED))
    self.assertEqual(_TEST_BYTE_DECODED, instance.decode(_TEST_BYTE_RAW))

  def testStringType(self):
    instance = standard_artifacts.String()
    self.assertEqual(_TEST_STRING_RAW, instance.encode(_TEST_STRING_DECODED))
    self.assertEqual(_TEST_STRING_DECODED, instance.decode(_TEST_STRING_RAW))

  def testBoolType(self):
    instance = standard_artifacts.Boolean()
    self.assertEqual(_TEST_BOOL_RAW, instance.encode(_TEST_BOOL_DECODED))
    self.assertEqual(_TEST_BOOL_DECODED, instance.decode(_TEST_BOOL_RAW))

  def testIntegerType(self):
    instance = standard_artifacts.Integer()
    self.assertEqual(_TEST_INT_RAW, instance.encode(_TEST_INT_DECODED))
    self.assertEqual(_TEST_INT_DECODED, instance.decode(_TEST_INT_RAW))

  def testFloatType(self):
    instance = standard_artifacts.Float()
    self.assertEqual(_TEST_FLOAT_RAW, instance.encode(_TEST_FLOAT_DECODED))
    self.assertAlmostEqual(_TEST_FLOAT_DECODED,
                           instance.decode(_TEST_FLOAT_RAW))

  def testJsonValueList(self):
    instance = standard_artifacts.JsonValue()
    self.assertEqual(_TEST_JSONVALUE_LIST_RAW,
                     instance.encode(_TEST_JSONVALUE_LIST_DECODED))
    self.assertEqual(_TEST_JSONVALUE_LIST_DECODED,
                     instance.decode(_TEST_JSONVALUE_LIST_RAW))

  def testJsonValueDict(self):
    instance = standard_artifacts.JsonValue()
    self.assertEqual(_TEST_JSONVALUE_DICT_RAW,
                     instance.encode(_TEST_JSONVALUE_DICT_DECODED))
    self.assertEqual(_TEST_JSONVALUE_DICT_DECODED,
                     instance.decode(_TEST_JSONVALUE_DICT_RAW))

  def testJsonValueObj(self):
    instance = standard_artifacts.JsonValue()
    self.assertEqual(_TEST_JSONVALUE_OBJ_RAW,
                     instance.encode(_TEST_JSONVALUE_OBJ_DECODED))
    self.assertEqual(_TEST_JSONVALUE_OBJ_DECODED,
                     instance.decode(_TEST_JSONVALUE_OBJ_RAW))

  @mock.patch('absl.logging.warning')
  def testFloatTypePrecisionLossWarning(self, *unused_mocks):
    instance = standard_artifacts.Float()
    # TODO(b/156776413): with self.assertWarnsRegex('lost precision'):
    self.assertAlmostEqual(
        instance.decode(_TEST_FLOAT128_RAW), _TEST_FLOAT128)
    # Lost precision warning
    absl.logging.warning.assert_called_once()

  @mock.patch('absl.logging.warning')
  def testFloatInfNanEncodingWarning(self, *unused_mocks):
    instance = standard_artifacts.Float()
    instance.encode(float('inf'))
    # Non-portable encoding warning
    absl.logging.warning.assert_called_once()

  def testSpecialFloatValues(self):
    coder = standard_artifacts.Float()
    positive_infinity_float = float('inf')
    negative_infinity_float = float('-inf')
    nan_float = float('nan')

    encoded_positive_infinity = coder.encode(positive_infinity_float)
    encoded_negative_infinity = coder.encode(negative_infinity_float)
    encoded_nan = coder.encode(nan_float)

    decoded_positive_infinity = coder.decode(encoded_positive_infinity)
    decoded_negative_infinity = coder.decode(encoded_negative_infinity)
    decoded_nan = coder.decode(encoded_nan)

    self.assertEqual(encoded_positive_infinity, b'Infinity')
    self.assertEqual(encoded_negative_infinity, b'-Infinity')
    self.assertEqual(encoded_nan, b'NaN')

    self.assertEqual(decoded_positive_infinity, positive_infinity_float)
    self.assertEqual(decoded_negative_infinity, negative_infinity_float)

    self.assertTrue(math.isinf(decoded_positive_infinity))
    self.assertTrue(math.isinf(decoded_negative_infinity))
    self.assertTrue(math.isnan(decoded_nan))


if __name__ == '__main__':
  tf.test.main()
