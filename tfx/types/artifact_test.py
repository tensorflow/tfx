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

import json

# Standard Imports

import tensorflow as tf
from tfx.types import artifact


class ArtifactTest(tf.test.TestCase):

  def testArtifact(self):
    instance = artifact.Artifact('MyTypeName', split='eval')

    # Test property getters.
    self.assertEqual('', instance.uri)
    self.assertEqual(0, instance.id)
    self.assertEqual(0, instance.type_id)
    self.assertEqual('MyTypeName', instance.type_name)
    self.assertEqual('', instance.state)
    self.assertEqual('eval', instance.split)
    self.assertEqual(0, instance.span)

    # Test property setters.
    instance.uri = '/tmp/uri2'
    self.assertEqual('/tmp/uri2', instance.uri)

    instance.id = 1
    self.assertEqual(1, instance.id)

    instance.type_id = 2
    self.assertEqual(2, instance.type_id)

    instance.state = artifact.ArtifactState.DELETED
    self.assertEqual(artifact.ArtifactState.DELETED, instance.state)

    instance.split = ''
    self.assertEqual('', instance.split)

    instance.span = 20190101
    self.assertEqual(20190101, instance.span)

    instance.set_int_custom_property('int_key', 20)
    self.assertEqual(20,
                     instance.artifact.custom_properties['int_key'].int_value)

    instance.set_string_custom_property('string_key', 'string_value')
    self.assertEqual(
        'string_value',
        instance.artifact.custom_properties['string_key'].string_value)

    self.assertEqual(
        'Artifact(type_name: MyTypeName, uri: /tmp/uri2, split: , id: 1)',
        str(instance))

    # Test json serialization.
    json_dict = instance.json_dict()
    s = json.dumps(json_dict)
    other_instance = artifact.Artifact.parse_from_json_dict(json.loads(s))
    self.assertEqual(instance.artifact, other_instance.artifact)
    self.assertEqual(instance.artifact_type, other_instance.artifact_type)

  def testInvalidArtifact(self):
    with self.assertRaisesRegexp(ValueError,
                                 'The "type_name" field must be passed'):
      artifact.Artifact()

    class MyBadArtifact(artifact.Artifact):
      # No TYPE_NAME
      pass

    with self.assertRaisesRegexp(
        ValueError,
        'The Artifact subclass .* must override the TYPE_NAME attribute '):
      MyBadArtifact()

    class MyArtifact(artifact.Artifact):
      TYPE_NAME = 'MyType'

    # Okay without additional type_name argument.
    MyArtifact()

    # Not okay to pass type_name on subclass.
    with self.assertRaisesRegexp(
        ValueError,
        'The "type_name" field must not be passed for Artifact subclass'):
      MyArtifact(type_name='OtherType')


if __name__ == '__main__':
  tf.test.main()
