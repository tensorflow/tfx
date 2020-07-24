# Lint as: python2, python3
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import absl
import mock
import tensorflow as tf

from tfx.types import standard_artifacts

# Define constant value for tests.
_TEST_BYTE_RAW = b'hello world'
_TEST_BYTE_DECODED = b'hello world'

_TEST_STRING_RAW = b'hello world'
_TEST_STRING_DECODED = u'hello world'

_TEST_INT_RAW = b'19260817'
_TEST_INT_DECODED = 19260817

_TEST_FLOAT_RAW = b'3.1415926535'
_TEST_FLOAT_DECODED = 3.1415926535

_TEST_FLOAT128_RAW = b'3.14159265358979323846264338327950288'
_TEST_FLOAT128 = 3.14159265358979323846264338327950288  # Too precise


class StandardArtifactsTest(tf.test.TestCase):

  def testBytesType(self):
    instance = standard_artifacts.Bytes()
    self.assertEqual(_TEST_BYTE_RAW, instance.encode(_TEST_BYTE_DECODED))
    self.assertEqual(_TEST_BYTE_DECODED, instance.decode(_TEST_BYTE_RAW))

  def testStringType(self):
    instance = standard_artifacts.String()
    self.assertEqual(_TEST_STRING_RAW, instance.encode(_TEST_STRING_DECODED))
    self.assertEqual(_TEST_STRING_DECODED, instance.decode(_TEST_STRING_RAW))

  def testIntegerType(self):
    instance = standard_artifacts.Integer()
    self.assertEqual(_TEST_INT_RAW, instance.encode(_TEST_INT_DECODED))
    self.assertEqual(_TEST_INT_DECODED, instance.decode(_TEST_INT_RAW))

  def testFloatType(self):
    instance = standard_artifacts.Float()
    self.assertEqual(_TEST_FLOAT_RAW, instance.encode(_TEST_FLOAT_DECODED))
    self.assertAlmostEqual(_TEST_FLOAT_DECODED,
                           instance.decode(_TEST_FLOAT_RAW))

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
