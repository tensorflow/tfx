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

import json
from unittest import mock

import tensorflow as tf
from tfx.dsl.io import fileio
from tfx.types import standard_artifacts
from tfx.types import system_artifacts
from tfx.types import value_artifact

from google.protobuf import json_format
from ml_metadata.proto import metadata_store_pb2


_IS_NULL_KEY = '__is_null__'


class _MyValueArtifact(value_artifact.ValueArtifact):
  TYPE_NAME = 'MyValueTypeName'

  def encode(self, value: str):
    assert isinstance(value, str), value
    return value.encode('utf-8')

  def decode(self, value: bytes):
    return value.decode('utf-8')


class MyDataset(system_artifacts.SystemArtifact):

  MLMD_SYSTEM_BASE_TYPE = 1


_mlmd_artifact_type = metadata_store_pb2.ArtifactType()
json_format.Parse(
    json.dumps({
        'name': 'String_MODEL',
        'base_type': 'MODEL',
    }), _mlmd_artifact_type)
_MyValueArtifact1 = value_artifact._ValueArtifactType(
    mlmd_artifact_type=_mlmd_artifact_type, base=standard_artifacts.String)  # pylint: disable=invalid-name


# Mock values for string artifact.
_STRING_VALUE = u'This is a string'
_BYTE_VALUE = b'This is a string'

# Mock paths for string artifact.
_VALID_URI = '/tmp/uri/value'
_VALID_FILE_URI = _VALID_URI

# Mock invalid paths. _BAD_URI points to a valid dir but there's no file within.
_BAD_URI = '/tmp/to/a/bad/dir'


def fake_exist(path: str) -> bool:
  """Mock behavior of fileio.exists."""
  return path in [_VALID_URI, _VALID_FILE_URI]


def fake_isdir(path: str) -> bool:
  """Mock behavior of fileio.isdir."""
  return path in [_VALID_URI]


def fake_open(unused_path: str, unused_mode: str = 'r') -> bool:
  """Mock behavior of fileio.open."""
  mock_open = mock.Mock()
  mock_open.contents = b''
  mock_open.read.side_effect = lambda: mock_open.contents
  def write_func(value):
    mock_open.contents = value
  mock_write = mock.Mock()
  mock_write.write.side_effect = write_func
  mock_open.__enter__ = mock_write
  mock_open.__exit__ = mock.Mock()
  return mock_open


class ValueArtifactTest(tf.test.TestCase):
  """Tests for ValueArtifact."""

  def setUp(self):
    super().setUp()
    self.addCleanup(mock.patch.stopall)

  @mock.patch.object(fileio, 'exists', fake_exist)
  @mock.patch.object(fileio, 'isdir', fake_isdir)
  @mock.patch.object(fileio, 'open', fake_open)
  def testValueArtifact(self):
    instance = _MyValueArtifact()
    # Test property setters.
    instance.uri = _VALID_URI
    self.assertEqual(_VALID_URI, instance.uri)

    with self.assertRaisesRegex(
        ValueError, 'The artifact value has not yet been read from storage.'):
      instance.value  # pylint: disable=pointless-statement

    instance.read()
    instance.value = _STRING_VALUE
    self.assertEqual(_STRING_VALUE, instance.value)
    self.assertEqual(
        0, instance.get_int_custom_property(_IS_NULL_KEY))
    instance.value = None
    self.assertIsNone(instance.value)
    self.assertEqual(
        1, instance.get_int_custom_property(_IS_NULL_KEY))

  @mock.patch.object(fileio, 'exists', fake_exist)
  @mock.patch.object(fileio, 'isdir', fake_isdir)
  @mock.patch.object(fileio, 'open', fake_open)
  def testValueArtifactWithBadUri(self):
    instance = _MyValueArtifact()
    instance.uri = _BAD_URI

    with self.assertRaisesRegex(
        RuntimeError, 'Given path does not exist or is not a valid file'):
      instance.read()

  def testTypeAnnotation(self):
    annotation_class = _MyValueArtifact.annotate_as(MyDataset)
    self.assertEqual(annotation_class.__name__, '_MyValueArtifact_MyDataset')
    self.assertEqual(annotation_class.TYPE_NAME, 'MyValueTypeName_MyDataset')
    self.assertEqual(annotation_class.TYPE_ANNOTATION.MLMD_SYSTEM_BASE_TYPE,
                     MyDataset.MLMD_SYSTEM_BASE_TYPE)
    self.assertEqual(annotation_class._get_artifact_type().base_type,
                     MyDataset.MLMD_SYSTEM_BASE_TYPE)

    # invalid annotation class
    with self.assertRaisesRegex(ValueError,
                                'is not a subclass of SystemArtifact'):
      _MyValueArtifact.annotate_as(value_artifact.ValueArtifact)

    # no argument
    annotation_class = _MyValueArtifact.annotate_as()
    self.assertEqual(annotation_class.__name__, '_MyValueArtifact')

  @mock.patch.object(fileio, 'exists', fake_exist)
  @mock.patch.object(fileio, 'isdir', fake_isdir)
  @mock.patch.object(fileio, 'open', fake_open)
  def testValueArtifactTypeConstructor(self):
    instance = _MyValueArtifact1()
    self.assertEqual(_MyValueArtifact1.__name__, 'String_MODEL')
    self.assertEqual(_MyValueArtifact1.TYPE_NAME, 'String_MODEL')
    self.assertEqual(_MyValueArtifact1.TYPE_ANNOTATION.MLMD_SYSTEM_BASE_TYPE,
                     metadata_store_pb2.ArtifactType.MODEL)

    self.assertIsInstance(instance, value_artifact.ValueArtifact)
    self.assertIsInstance(instance, standard_artifacts.String)

    # Test property setters.
    instance.uri = _VALID_URI
    self.assertEqual(_VALID_URI, instance.uri)

    # Test functions are inherited from base class standard_artifacts.String.
    instance.read()
    instance.value = _STRING_VALUE
    self.assertEqual(_STRING_VALUE, instance.value)


if __name__ == '__main__':
  tf.test.main()
