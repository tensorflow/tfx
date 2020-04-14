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

import tensorflow as tf

from tfx.types import standard_artifacts

# Define constant value for tests.
_TEST_BYTE_RAW = b'hello world'
_TEST_BYTE_DECODED = b'hello world'

_TEST_STRING_RAW = b'hello world'
_TEST_STRING_DECODED = u'hello world'

_TEST_INT_RAW = b'\x01%\xe5\x91'
_TEST_INT_DECODED = 19260817

_TEST_FLOAT_RAW = b'@\t!\xfbTA\x17D'
_TEST_FLOAT_DECODED = 3.1415926535


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


if __name__ == '__main__':
  tf.test.main()
