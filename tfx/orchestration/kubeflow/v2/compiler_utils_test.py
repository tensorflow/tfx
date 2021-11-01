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

from absl.testing import parameterized
from kfp.pipeline_spec import pipeline_spec_pb2 as pipeline_pb2
import tensorflow as tf
from tfx.dsl.io import fileio
from tfx.orchestration import data_types
from tfx.orchestration.kubeflow.v2 import compiler_utils
from tfx.types import artifact
from tfx.types import channel
from tfx.types import channel_utils
from tfx.types import standard_artifacts
from tfx.types.experimental import simple_artifacts
import yaml

from google.protobuf import text_format

_EXPECTED_MY_ARTIFACT_SCHEMA = """
title: __main__._MyArtifact
type: object
"""

_EXPECTED_MY_BAD_ARTIFACT_SCHEMA = """
title: __main__._MyArtifactWithProperty
type: object
"""

_MY_BAD_ARTIFACT_SCHEMA_WITH_PROPERTIES = """
title: __main__._MyArtifactWithProperty
type: object
properties:
  int1:
    type: string
"""


class _MyArtifact(artifact.Artifact):
  TYPE_NAME = 'TestType'


# _MyArtifactWithProperty should fail the compilation by specifying
# custom property schema, which is not supported yet.
class _MyArtifactWithProperty(artifact.Artifact):
  TYPE_NAME = 'TestBadType'
  PROPERTIES = {
      'int1': artifact.Property(type=artifact.PropertyType.INT),
  }


_TEST_CHANNEL = channel.Channel(type=_MyArtifactWithProperty)


class CompilerUtilsTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._schema_base_dir = os.path.join(
        os.path.dirname(__file__), 'artifact_types')

  def testTitleToClassMapping(self):
    self.assertEqual(compiler_utils.TITLE_TO_CLASS_PATH['tfx.Examples'],
                     'tfx.types.standard_artifacts.Examples')
    self.assertEqual(compiler_utils.TITLE_TO_CLASS_PATH['tfx.String'],
                     'tfx.types.standard_artifacts.String')
    self.assertEqual(compiler_utils.TITLE_TO_CLASS_PATH['tfx.Metrics'],
                     'tfx.types.experimental.simple_artifacts.Metrics')

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
    my_artifact_with_property = _MyArtifactWithProperty()
    my_artifact_with_property_schema = compiler_utils.get_artifact_schema(
        my_artifact_with_property)
    self.assertDictEqual(
        yaml.safe_load(my_artifact_with_property_schema),
        yaml.safe_load(_EXPECTED_MY_BAD_ARTIFACT_SCHEMA))

    my_artifact_with_property.int1 = 42
    with self.assertRaisesRegex(KeyError, 'Actual property:'):
      _ = compiler_utils.build_output_artifact_spec(
          channel_utils.as_channel([my_artifact_with_property]))

  def testCustomArtifactSchemaMismatchFails(self):
    with self.assertRaisesRegex(TypeError, 'Property type mismatched at'):
      compiler_utils._validate_properties_schema(
          _MY_BAD_ARTIFACT_SCHEMA_WITH_PROPERTIES,
          _MyArtifactWithProperty.PROPERTIES)

  def testBuildParameterTypeSpec(self):
    type_enum = pipeline_pb2.PrimitiveType.PrimitiveTypeEnum
    testdata = {
        42: type_enum.INT,
        42.1: type_enum.DOUBLE,
        '42': type_enum.STRING,
        data_types.RuntimeParameter(name='_', ptype=int): type_enum.INT,
        data_types.RuntimeParameter(name='_', ptype=float): type_enum.DOUBLE,
        data_types.RuntimeParameter(name='_', ptype=str): type_enum.STRING,
    }
    for value, expected_type_enum in testdata.items():
      self.assertEqual(
          compiler_utils.build_parameter_type_spec(value).type,
          expected_type_enum)

  def testBuildInputArtifactSpec(self):
    spec = compiler_utils.build_input_artifact_spec(
        channel.Channel(type=standard_artifacts.Model))
    expected_spec = text_format.Parse(
        r'artifact_type { instance_schema: "title: tfx.Model\ntype: object\n" }',
        pipeline_pb2.ComponentInputsSpec.ArtifactSpec())
    self.assertProtoEquals(spec, expected_spec)

    # Test artifact type with properties
    spec = compiler_utils.build_input_artifact_spec(
        channel.Channel(type=standard_artifacts.Examples))
    expected_spec = text_format.Parse(
        """
        artifact_type {
          instance_schema: "title: tfx.Examples\\ntype: object\\nproperties:\\n  span:\\n    type: integer\\n    description: Span for an artifact.\\n  version:\\n    type: integer\\n    description: Version for an artifact.\\n  split_names:\\n    type: string\\n    description: JSON-encoded list of splits for an artifact. Empty string means artifact has no split.\\n"
        }
        """, pipeline_pb2.ComponentInputsSpec.ArtifactSpec())
    self.assertProtoEquals(spec, expected_spec)

  def testBuildOutputArtifactSpec(self):
    examples = standard_artifacts.Examples()
    examples.span = 1
    examples.set_int_custom_property(key='int_param', value=42)
    examples.set_string_custom_property(key='str_param', value='42')
    example_channel = channel.Channel(
        type=standard_artifacts.Examples).set_artifacts([examples])
    spec = compiler_utils.build_output_artifact_spec(example_channel)
    expected_spec = text_format.Parse(
        """
        artifact_type {
          instance_schema: "title: tfx.Examples\\ntype: object\\nproperties:\\n  span:\\n    type: integer\\n    description: Span for an artifact.\\n  version:\\n    type: integer\\n    description: Version for an artifact.\\n  split_names:\\n    type: string\\n    description: JSON-encoded list of splits for an artifact. Empty string means artifact has no split.\\n"
        }
        metadata {
          fields {
            key: "int_param"
            value {
              number_value: 42.0
            }
          }
          fields {
            key: "span"
            value {
              number_value: 1.0
            }
          }
          fields {
            key: "str_param"
            value {
              string_value: "42"
            }
          }
        }
        """, pipeline_pb2.ComponentOutputsSpec.ArtifactSpec())
    self.assertProtoEquals(spec, expected_spec)

    # Empty output channel with only type info.
    model_channel = channel.Channel(type=standard_artifacts.Model)
    spec = compiler_utils.build_output_artifact_spec(model_channel)
    expected_spec = text_format.Parse(
        """
        artifact_type {
          instance_schema: "title: tfx.Model\\ntype: object\\n"
        }
        """, pipeline_pb2.ComponentOutputsSpec.ArtifactSpec())
    self.assertProtoEquals(spec, expected_spec)


class PlaceholderToCELTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._test_channel = channel.Channel(type=_MyArtifactWithProperty)

  @parameterized.named_parameters(
      {
          'testcase_name':
              'two_sides_placeholder',
          'predicate':
              _TEST_CHANNEL.future()[0].property('int1') <
              _TEST_CHANNEL.future()[0].property('int2'),
          'expected_cel':
              '(inputs.artifacts[\'key\'].artifacts[0].metadata[\'int1\'] < '
              'inputs.artifacts[\'key\'].artifacts[0].metadata[\'int2\'])',
      },
      {
          'testcase_name':
              'left_side_placeholder_right_side_int',
          'predicate':
              _TEST_CHANNEL.future()[0].property('int') < 1,
          'expected_cel':
              '(inputs.artifacts[\'key\'].artifacts[0].metadata[\'int\'] < 1.0)',
      },
      {
          'testcase_name':
              'left_side_placeholder_right_side_float',
          'predicate':
              _TEST_CHANNEL.future()[0].property('float') < 1.1,
          'expected_cel':
              '(inputs.artifacts[\'key\'].artifacts[0].metadata[\'float\'] < '
              '1.1)',
      },
      {
          'testcase_name': 'left_side_placeholder_right_side_string',
          'predicate': _TEST_CHANNEL.future()[0].property('str') == 'test_str',
          'expected_cel':
              '(inputs.artifacts[\'key\'].artifacts[0].metadata[\'str\'] == '
              '\'test_str\')',
      },
  )
  def testComparison(self, predicate, expected_cel):
    channel_to_key_map = {
        _TEST_CHANNEL: 'key',
    }
    placeholder_pb = predicate.encode_with_keys(lambda c: channel_to_key_map[c])
    self.assertEqual(
        compiler_utils.placeholder_to_cel(placeholder_pb), expected_cel)

  def testArtifactUri(self):
    predicate = _TEST_CHANNEL.future()[0].uri == 'test_str'
    expected_cel = ('(inputs.artifacts[\'key\'].artifacts[0].uri == '
                    '\'test_str\')')
    channel_to_key_map = {
        _TEST_CHANNEL: 'key',
    }
    placeholder_pb = predicate.encode_with_keys(lambda c: channel_to_key_map[c])
    self.assertEqual(
        compiler_utils.placeholder_to_cel(placeholder_pb), expected_cel)

  def testNegation(self):
    predicate = _TEST_CHANNEL.future()[0].property('int') != 1
    expected_cel = ('!((inputs.artifacts[\'key\'].artifacts[0]'
                    '.metadata[\'int\'] == 1.0))')
    channel_to_key_map = {
        _TEST_CHANNEL: 'key',
    }
    placeholder_pb = predicate.encode_with_keys(lambda c: channel_to_key_map[c])
    self.assertEqual(
        compiler_utils.placeholder_to_cel(placeholder_pb), expected_cel)

  def testConcat(self):
    predicate = _TEST_CHANNEL.future()[0].uri + 'something' == 'test_str'
    expected_cel = (
        '((inputs.artifacts[\'key\'].artifacts[0].uri + \'something\') == '
        '\'test_str\')')
    channel_to_key_map = {
        _TEST_CHANNEL: 'key',
    }
    placeholder_pb = predicate.encode_with_keys(lambda c: channel_to_key_map[c])
    self.assertEqual(
        compiler_utils.placeholder_to_cel(placeholder_pb), expected_cel)

  def testUnsupportedOperator(self):
    predicate = _TEST_CHANNEL.future()[0].b64encode() == 'test_str'
    channel_to_key_map = {
        _TEST_CHANNEL: 'key',
    }
    placeholder_pb = predicate.encode_with_keys(lambda c: channel_to_key_map[c])
    with self.assertRaisesRegex(
        ValueError, 'Got unsupported placeholder operator base64_encode_op.'):
      compiler_utils.placeholder_to_cel(placeholder_pb)

  def testPlaceholderWithoutKey(self):
    predicate = _TEST_CHANNEL.future()[0].uri == 'test_str'
    placeholder_pb = predicate.encode()
    with self.assertRaisesRegex(
        ValueError,
        'Only supports accessing placeholders with a key on KFPv2.'):
      compiler_utils.placeholder_to_cel(placeholder_pb)


if __name__ == '__main__':
  tf.test.main()
