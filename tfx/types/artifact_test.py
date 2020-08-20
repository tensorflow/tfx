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

import json
from typing import Text

# Standard Imports

import absl
import mock
import tensorflow as tf
from tfx.types import artifact
from tfx.utils import json_utils

from google.protobuf import json_format
from ml_metadata.proto import metadata_store_pb2


class _MyArtifact(artifact.Artifact):
  TYPE_NAME = 'MyTypeName'
  PROPERTIES = {
      'int1': artifact.Property(type=artifact.PropertyType.INT),
      'int2': artifact.Property(type=artifact.PropertyType.INT),
      'float1': artifact.Property(type=artifact.PropertyType.FLOAT),
      'float2': artifact.Property(type=artifact.PropertyType.FLOAT),
      'string1': artifact.Property(type=artifact.PropertyType.STRING),
      'string2': artifact.Property(type=artifact.PropertyType.STRING),
  }

_MyArtifact2 = artifact._ArtifactType(  # pylint: disable=invalid-name
    name='MyTypeName2',
    properties={
        'int1': artifact.Property(type=artifact.PropertyType.INT),
        'int2': artifact.Property(type=artifact.PropertyType.INT),
        'float1': artifact.Property(type=artifact.PropertyType.FLOAT),
        'float2': artifact.Property(type=artifact.PropertyType.FLOAT),
        'string1': artifact.Property(type=artifact.PropertyType.STRING),
        'string2': artifact.Property(type=artifact.PropertyType.STRING),
    })

_mlmd_artifact_type = metadata_store_pb2.ArtifactType()
json_format.Parse(
    json.dumps({
        'name': 'MyTypeName3',
        'properties': {
            'int1': 'INT',
            'int2': 'INT',
            'float1': 'DOUBLE',
            'float2': 'DOUBLE',
            'string1': 'STRING',
            'string2': 'STRING'
        }
    }), _mlmd_artifact_type)
_MyArtifact3 = artifact._ArtifactType(mlmd_artifact_type=_mlmd_artifact_type)  # pylint: disable=invalid-name


class _MyValueArtifact(artifact.ValueArtifact):
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
_VALID_URI = '/tmp/uri'
_VALID_FILE_URI = _VALID_URI

# Mock invalid paths. _BAD_URI points to a valid dir but there's no file within.
_BAD_URI = '/tmp/to/a/bad/dir'


def fake_exist(path: Text) -> bool:
  """Mock behavior of tf.io.gfile.exists."""
  return path in [_VALID_URI, _VALID_FILE_URI]


def fake_isdir(path: Text) -> bool:
  """Mock behavior of tf.io.gfile.isdir."""
  return path in [_VALID_URI]


class ArtifactTest(tf.test.TestCase):

  def testArtifact(self):
    instance = _MyArtifact()

    # Test property getters.
    self.assertEqual('', instance.uri)
    self.assertEqual(0, instance.id)
    self.assertEqual(0, instance.type_id)
    self.assertEqual('MyTypeName', instance.type_name)
    self.assertEqual('', instance.state)

    # Default property does not have span or split_names.
    with self.assertRaisesRegexp(AttributeError, "has no property 'span'"):
      instance.span  # pylint: disable=pointless-statement
    with self.assertRaisesRegexp(AttributeError,
                                 "has no property 'split_names'"):
      instance.split_names  # pylint: disable=pointless-statement

    # Test property setters.
    instance.uri = '/tmp/uri2'
    self.assertEqual('/tmp/uri2', instance.uri)

    instance.id = 1
    self.assertEqual(1, instance.id)

    instance.type_id = 2
    self.assertEqual(2, instance.type_id)

    instance.state = artifact.ArtifactState.DELETED
    self.assertEqual(artifact.ArtifactState.DELETED, instance.state)

    # Default artifact does not have span.
    with self.assertRaisesRegexp(AttributeError, "unknown property 'span'"):
      instance.span = 20190101
    # Default artifact does not have span.
    with self.assertRaisesRegexp(AttributeError,
                                 "unknown property 'split_names'"):
      instance.split_names = ''

    instance.set_int_custom_property('int_key', 20)
    self.assertEqual(
        20, instance.mlmd_artifact.custom_properties['int_key'].int_value)

    instance.set_string_custom_property('string_key', 'string_value')
    self.assertEqual(
        'string_value',
        instance.mlmd_artifact.custom_properties['string_key'].string_value)

    self.assertEqual(
        'Artifact(artifact: id: 1\n'
        'type_id: 2\n'
        'uri: "/tmp/uri2"\n'
        'custom_properties {\n'
        '  key: "int_key"\n'
        '  value {\n'
        '    int_value: 20\n'
        '  }\n'
        '}\n'
        'custom_properties {\n'
        '  key: "state"\n'
        '  value {\n'
        '    string_value: "deleted"\n'
        '  }\n'
        '}\n'
        'custom_properties {\n'
        '  key: "string_key"\n'
        '  value {\n'
        '    string_value: "string_value"\n'
        '  }\n'
        '}\n'
        ', artifact_type: name: "MyTypeName"\n'
        'properties {\n'
        '  key: "float1"\n'
        '  value: DOUBLE\n'
        '}\n'
        'properties {\n'
        '  key: "float2"\n'
        '  value: DOUBLE\n'
        '}\n'
        'properties {\n'
        '  key: "int1"\n'
        '  value: INT\n'
        '}\n'
        'properties {\n'
        '  key: "int2"\n'
        '  value: INT\n'
        '}\n'
        'properties {\n'
        '  key: "string1"\n'
        '  value: STRING\n'
        '}\n'
        'properties {\n'
        '  key: "string2"\n'
        '  value: STRING\n'
        '}\n'
        ')', str(instance))

    # Test json serialization.
    json_dict = json_utils.dumps(instance)
    other_instance = json_utils.loads(json_dict)
    self.assertEqual(instance.mlmd_artifact, other_instance.mlmd_artifact)
    self.assertEqual(instance.artifact_type, other_instance.artifact_type)

  def testArtifactTypeFunctionAndProto(self):
    # Test usage of _MyArtifact2 and _MyArtifact3, which were defined using the
    # _ArtifactType function.
    types_and_names = [
        (_MyArtifact2, 'MyTypeName2'),
        (_MyArtifact3, 'MyTypeName3'),
    ]
    for type_cls, name in types_and_names:
      self.assertEqual(type_cls.TYPE_NAME, name)
      my_artifact = type_cls()
      self.assertEqual(0, my_artifact.int1)
      self.assertEqual(0, my_artifact.int2)
      my_artifact.int1 = 111
      my_artifact.int2 = 222
      self.assertEqual(0.0, my_artifact.float1)
      self.assertEqual(0.0, my_artifact.float2)
      my_artifact.float1 = 111.1
      my_artifact.float2 = 222.2
      self.assertEqual('', my_artifact.string1)
      self.assertEqual('', my_artifact.string2)
      my_artifact.string1 = '111'
      my_artifact.string2 = '222'
      self.assertEqual(my_artifact.int1, 111)
      self.assertEqual(my_artifact.int2, 222)
      self.assertEqual(my_artifact.float1, 111.1)
      self.assertEqual(my_artifact.float2, 222.2)
      self.assertEqual(my_artifact.string1, '111')
      self.assertEqual(my_artifact.string2, '222')

  def testInvalidArtifact(self):
    with self.assertRaisesRegexp(
        ValueError, 'The "mlmd_artifact_type" argument must be passed'):
      artifact.Artifact()

    class MyBadArtifact(artifact.Artifact):
      # No TYPE_NAME
      pass

    with self.assertRaisesRegexp(
        ValueError,
        'The Artifact subclass .* must override the TYPE_NAME attribute '):
      MyBadArtifact()

    class MyNewArtifact(artifact.Artifact):
      TYPE_NAME = 'MyType'

    # Okay without additional type_name argument.
    MyNewArtifact()

    # Not okay to pass type_name on subclass.
    with self.assertRaisesRegexp(
        ValueError,
        'The "mlmd_artifact_type" argument must not be passed for Artifact '
        'subclass'):
      MyNewArtifact(mlmd_artifact_type=metadata_store_pb2.ArtifactType())

  def testArtifactProperties(self):
    my_artifact = _MyArtifact()
    self.assertEqual(0, my_artifact.int1)
    self.assertEqual(0, my_artifact.int2)
    my_artifact.int1 = 111
    my_artifact.int2 = 222
    self.assertEqual('', my_artifact.string1)
    self.assertEqual('', my_artifact.string2)
    my_artifact.string1 = '111'
    my_artifact.string2 = '222'
    self.assertEqual(my_artifact.int1, 111)
    self.assertEqual(my_artifact.int2, 222)
    self.assertEqual(my_artifact.string1, '111')
    self.assertEqual(my_artifact.string2, '222')

    with self.assertRaisesRegexp(
        AttributeError, "Cannot set unknown property 'invalid' on artifact"):
      my_artifact.invalid = 1

    with self.assertRaisesRegexp(
        AttributeError, "Cannot set unknown property 'invalid' on artifact"):
      my_artifact.invalid = 'x'

    with self.assertRaisesRegexp(AttributeError,
                                 "Artifact has no property 'invalid'"):
      my_artifact.invalid  # pylint: disable=pointless-statement

  def testStringTypeNameNotAllowed(self):
    with self.assertRaisesRegexp(
        ValueError,
        'The "mlmd_artifact_type" argument must be an instance of the proto '
        'message'):
      artifact.Artifact('StringTypeName')

  @mock.patch('absl.logging.warning')
  def testDeserialize(self, *unused_mocks):
    original = _MyArtifact()
    original.uri = '/my/path'
    original.int1 = 111
    original.int2 = 222
    original.string1 = '111'
    original.string2 = '222'

    serialized = original.to_json_dict()

    rehydrated = artifact.Artifact.from_json_dict(serialized)
    absl.logging.warning.assert_not_called()
    self.assertIs(rehydrated.__class__, _MyArtifact)
    self.assertEqual(rehydrated.int1, 111)
    self.assertEqual(rehydrated.int2, 222)
    self.assertEqual(rehydrated.string1, '111')
    self.assertEqual(rehydrated.string2, '222')

  @mock.patch('absl.logging.warning')
  def testDeserializeUnknownArtifactClass(self, *unused_mocks):
    original = _MyArtifact()
    original.uri = '/my/path'
    original.int1 = 111
    original.int2 = 222
    original.string1 = '111'
    original.string2 = '222'

    serialized = original.to_json_dict()
    serialized['__artifact_class_name__'] = 'MissingClassName'

    rehydrated = artifact.Artifact.from_json_dict(serialized)
    absl.logging.warning.assert_called_once()
    self.assertIs(rehydrated.__class__, artifact.Artifact)
    self.assertEqual(rehydrated.int1, 111)
    self.assertEqual(rehydrated.int2, 222)
    self.assertEqual(rehydrated.string1, '111')
    self.assertEqual(rehydrated.string2, '222')

    serialized2 = rehydrated.to_json_dict()
    rehydrated = artifact.Artifact.from_json_dict(serialized2)
    self.assertIs(rehydrated.__class__, artifact.Artifact)
    self.assertEqual(rehydrated.int1, 111)
    self.assertEqual(rehydrated.int2, 222)
    self.assertEqual(rehydrated.string1, '111')
    self.assertEqual(rehydrated.string2, '222')

  def testCopyFrom(self):
    original = _MyArtifact()
    original.id = 1
    original.uri = '/my/path'
    original.int1 = 111
    original.string1 = '111'
    original.set_string_custom_property('my_custom_property', 'aaa')

    copied = _MyArtifact()
    copied.id = 2
    copied.uri = '/some/other/path'
    copied.int1 = 333
    original.set_string_custom_property('my_custom_property', 'bbb')
    copied.copy_from(original)

    # id should not be overridden.
    self.assertEqual(copied.id, 2)
    self.assertEqual(original.uri, copied.uri)
    self.assertEqual(original.int1, copied.int1)
    self.assertEqual(original.string1, copied.string1)
    self.assertEqual(original.get_string_custom_property('my_custom_property'),
                     copied.get_string_custom_property('my_custom_property'))

  def testCopyFromDifferentArtifactType(self):
    artifact1 = _MyArtifact()
    artifact2 = _MyArtifact2()
    with self.assertRaises(AssertionError):
      artifact2.copy_from(artifact1)


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
