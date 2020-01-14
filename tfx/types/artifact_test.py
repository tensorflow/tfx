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
from __future__ import unicode_literals

# Standard Imports

import absl
import mock

import tensorflow as tf
from ml_metadata.proto import metadata_store_pb2
from tfx.types import artifact
from tfx.utils import json_utils


class _MyArtifact(artifact.Artifact):
  TYPE_NAME = 'MyTypeName'
  PROPERTIES = {
      'int1': artifact.Property(type=artifact.PropertyType.INT),
      'int2': artifact.Property(type=artifact.PropertyType.INT),
      'string1': artifact.Property(type=artifact.PropertyType.STRING),
      'string2': artifact.Property(type=artifact.PropertyType.STRING),
  }


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

    self.assertEqual('Artifact(type_name: MyTypeName, uri: /tmp/uri2, id: 1)',
                     str(instance))

    # Test json serialization.
    json_dict = json_utils.dumps(instance)
    other_instance = json_utils.loads(json_dict)
    self.assertEqual(instance.mlmd_artifact, other_instance.mlmd_artifact)
    self.assertEqual(instance.artifact_type, other_instance.artifact_type)

  def testArtifactSpecificProperties(self):
    pass

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


if __name__ == '__main__':
  tf.test.main()
