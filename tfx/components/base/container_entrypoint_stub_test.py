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
"""Tests for tfx.components.base.container_entrypoint_stub."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Standard Imports

import tensorflow as tf

from tfx import types
from tfx.components.base import container_entrypoint_stub as entrypoint_stub
from tfx.components.base import function_parser
from tfx.types.artifact import Property
from tfx.types.artifact import PropertyType


class MyArtifact(types.Artifact):
  TYPE_NAME = 'MyArtifact'
  PROPERTIES = {
      'string_property_1': Property(type=PropertyType.STRING),
      'string_property_2': Property(type=PropertyType.STRING),
      'int_property_1': Property(type=PropertyType.INT),
      'unused_string': Property(type=PropertyType.STRING),
      'unused_int': Property(type=PropertyType.INT),
  }


class ContainerComponentEntrypointStubTest(tf.test.TestCase):

  def testArgFormatsEnumsConsistent(self):
    # Test that `function_parser.ArgFormats` and `entrypoint_stub.ArgFormats`
    # are identical declarations.
    self.assertSequenceEqual(
        list((f.name, f.value) for f in function_parser.ArgFormats),
        list((f.name, f.value) for f in entrypoint_stub.ArgFormats))

  def testArtifactStub(self):
    artifact = MyArtifact()
    artifact.uri = '/my/path'
    artifact.string_property_1 = 'value_a'
    artifact.string_property_2 = 'value_b'
    artifact.int_property_1 = 3

    serialized = artifact.to_json_dict()

    rehydrated = entrypoint_stub.Artifact.from_json_dict(serialized)
    self.assertEqual(rehydrated._type_name, 'MyArtifact')
    self.assertEqual(rehydrated.uri, '/my/path')
    self.assertEqual(rehydrated.string_property_1, 'value_a')
    self.assertEqual(rehydrated.string_property_2, 'value_b')
    self.assertEqual(rehydrated.int_property_1, 3)
    self.assertEqual(rehydrated.unused_string, '')
    self.assertEqual(rehydrated.unused_int, 0)


if __name__ == '__main__':
  tf.test.main()
