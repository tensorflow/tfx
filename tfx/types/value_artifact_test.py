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
"""Tests for tfx.types.artifact."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Text

# Standard Imports

import mock
import tensorflow as tf
from tfx.types import value_artifact


class _MyValueArtifact(value_artifact.ValueArtifact):
  TYPE_NAME = 'MyValueTypeName'

  def encode(self, value: Text):
    assert isinstance(value, Text), value
    return value.encode('utf-8')

  def decode(self, value: bytes):
    return value.decode('utf-8')


# Mock values for string artifact.
_STRING_VALUE = u'This is a string'
_BYTE_VALUE = b'This is a string'

# Mock paths for string artifact.
_VALID_URI = '/tmp/uri/value'
_VALID_FILE_URI = _VALID_URI

# Mock invalid paths. _BAD_URI points to a valid dir but there's no file within.
_BAD_URI = '/tmp/to/a/bad/dir'


def fake_exist(path: Text) -> bool:
  """Mock behavior of tf.io.gfile.exists."""
  return path in [_VALID_URI, _VALID_FILE_URI]


def fake_isdir(path: Text) -> bool:
  """Mock behavior of tf.io.gfile.isdir."""
  return path in [_VALID_URI]


class ValueArtifactTest(tf.test.TestCase):
  """Tests for ValueArtifact."""

  def setUp(self):
    super(ValueArtifactTest, self).setUp()
    self.addCleanup(mock.patch.stopall)

    self._mock_gfile_readfn = mock.patch.object(
        tf.io.gfile.GFile,
        'read',
        autospec=True,
        return_value=_BYTE_VALUE,
    ).start()

  @mock.patch.object(tf.io.gfile, 'exists', fake_exist)
  @mock.patch.object(tf.io.gfile, 'isdir', fake_isdir)
  def testValueArtifact(self):
    instance = _MyValueArtifact()
    # Test property setters.
    instance.uri = _VALID_URI
    self.assertEqual(_VALID_URI, instance.uri)

    with self.assertRaisesRegexp(
        ValueError, 'The artifact value has not yet been read from storage.'):
      instance.value  # pylint: disable=pointless-statement

    instance.read()
    self.assertEqual(_STRING_VALUE, instance.value)

  @mock.patch.object(tf.io.gfile, 'exists', fake_exist)
  @mock.patch.object(tf.io.gfile, 'isdir', fake_isdir)
  def testValueArtifactWithBadUri(self):
    instance = _MyValueArtifact()
    instance.uri = _BAD_URI

    with self.assertRaisesRegexp(
        RuntimeError, 'Given path does not exist or is not a valid file'):
      instance.read()


if __name__ == '__main__':
  tf.test.main()
