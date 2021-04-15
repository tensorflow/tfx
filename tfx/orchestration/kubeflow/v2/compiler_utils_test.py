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
"""Tests for tfx.orchestration.kubeflow.v2.compiler_utils."""

import os
import tensorflow as tf

from tfx.dsl.io import fileio
from tfx.orchestration.kubeflow.v2 import compiler_utils
from tfx.types import artifact
from tfx.types import channel_utils
from tfx.types import standard_artifacts
from tfx.types.experimental import simple_artifacts
import yaml

_EXPECTED_MY_ARTIFACT_SCHEMA = """
title: __main__._MyArtifact
type: object
properties:
"""

_EXPECTED_MY_BAD_ARTIFACT_SCHEMA = """
title: __main__._MyBadArtifact
type: object
properties:
"""

_MY_BAD_ARTIFACT_SCHEMA_WITH_PROPERTIES = """
title: __main__._MyBadArtifact
type: object
properties:
  int1:
    type: string
"""


class _MyArtifact(artifact.Artifact):
  TYPE_NAME = 'TestType'


# _MyBadArtifact should fail the compilation by specifying custom property
# schema, which is not supported yet.
class _MyBadArtifact(artifact.Artifact):
  TYPE_NAME = 'TestBadType'
  PROPERTIES = {
      'int1': artifact.Property(type=artifact.PropertyType.INT),
  }


class CompilerUtilsTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._schema_base_dir = os.path.join(
        os.path.dirname(__file__), 'artifact_types')

  def testArtifactSchemaMapping(self):
    # Test first party standard artifact.
    example_artifact = standard_artifacts.Examples()
    example_schema = compiler_utils.get_artifact_schema(example_artifact)
    expected_example_schema = fileio.open(
        os.path.join(self._schema_base_dir, 'Examples.yaml'), 'rb').read()
    self.assertEqual(expected_example_schema, example_schema)

    # Test Kubeflow simple artifact.
    file_artifact = simple_artifacts.File()
    file_schema = compiler_utils.get_artifact_schema(file_artifact)
    expected_file_schema = fileio.open(
        os.path.join(self._schema_base_dir, 'File.yaml'), 'rb').read()
    self.assertEqual(expected_file_schema, file_schema)

    # Test custom artifact type.
    my_artifact = _MyArtifact()
    my_artifact_schema = compiler_utils.get_artifact_schema(my_artifact)
    self.assertDictEqual(
        yaml.safe_load(my_artifact_schema),
        yaml.safe_load(_EXPECTED_MY_ARTIFACT_SCHEMA))

  def testCustomArtifactMappingFails(self):
    my_bad_artifact = _MyBadArtifact()
    my_bad_artifact_schema = compiler_utils.get_artifact_schema(my_bad_artifact)
    self.assertDictEqual(
        yaml.safe_load(my_bad_artifact_schema),
        yaml.safe_load(_EXPECTED_MY_BAD_ARTIFACT_SCHEMA))

    my_bad_artifact.int1 = 42
    with self.assertRaisesRegexp(KeyError, 'Actual property:'):
      _ = compiler_utils.build_output_artifact_spec(
          channel_utils.as_channel([my_bad_artifact]))

  def testCustomArtifactSchemaMismatchFails(self):
    with self.assertRaisesRegexp(TypeError, 'Property type mismatched at'):
      compiler_utils._validate_properties_schema(
          _MY_BAD_ARTIFACT_SCHEMA_WITH_PROPERTIES, _MyBadArtifact.PROPERTIES)


if __name__ == '__main__':
  tf.test.main()
