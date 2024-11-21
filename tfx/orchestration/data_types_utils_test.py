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
"""Tests for tfx.orchestration.data_types_utils."""


import importlib
import pytest
from absl.testing import parameterized
from tfx import types
from tfx.orchestration import data_types_utils
from tfx.proto.orchestration import execution_result_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import artifact_utils
from tfx.utils import proto_utils
from tfx.utils import test_case_utils

from google.protobuf import struct_pb2
from google.protobuf import text_format

from ml_metadata.proto import metadata_store_pb2
from ml_metadata.proto import metadata_store_service_pb2

_DEFAULT_ARTIFACT_TYPE_NAME = 'Examples'


@pytest.fixture(scope="module", autouse=True)
def cleanup():
  yield
  importlib.reload(struct_pb2)


def _create_artifact(uri: str) -> types.Artifact:
  artifact = types.Artifact(
      metadata_store_pb2.ArtifactType(name=_DEFAULT_ARTIFACT_TYPE_NAME))
  artifact.uri = uri
  return artifact


class DataTypesUtilsTest(test_case_utils.TfxTest, parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.artifact_struct_dict = {
        'a1':
            text_format.Parse(
                """
                elements {
                  artifact {
                    artifact {
                      id: 123
                    }
                    type {
                      name: 't1'
                    }
                  }
                }
                """, metadata_store_service_pb2.ArtifactStructList()),
        'a2':
            text_format.Parse(
                """
                elements {
                  artifact {
                    artifact {
                      id: 456
                    }
                    type {
                      name: 't2'
                    }
                  }
                }
                """, metadata_store_service_pb2.ArtifactStructList())
    }

    self.artifact_dict = {
        'a1': [
            artifact_utils.deserialize_artifact(
                metadata_store_pb2.ArtifactType(name='t1'),
                metadata_store_pb2.Artifact(id=123))
        ],
        'a2': [
            artifact_utils.deserialize_artifact(
                metadata_store_pb2.ArtifactType(name='t2'),
                metadata_store_pb2.Artifact(id=456))
        ]
    }

    self.metadata_value_dict = {
        'p0': metadata_store_pb2.Value(int_value=0),
        'p1': metadata_store_pb2.Value(int_value=1),
        'p2': metadata_store_pb2.Value(string_value='hello'),
        'p3': metadata_store_pb2.Value(string_value='')
    }
    self.value_dict = {'p0': 0, 'p1': 1, 'p2': 'hello', 'p3': ''}

  def testBuildArtifactDict(self):
    actual_artifact_dict = data_types_utils.build_artifact_dict(
        self.artifact_struct_dict)
    for k, v in actual_artifact_dict.items():
      self.assertLen(self.artifact_dict[k], len(v))
      self.assertEqual(self.artifact_dict[k][0].id, v[0].id)
      self.assertEqual(self.artifact_dict[k][0].type_name, v[0].type_name)

  def testUnpackExecutorOutput(self):
    artifact0 = _create_artifact('uri0').mlmd_artifact
    artifact1 = _create_artifact('uri1').mlmd_artifact
    artifact2 = _create_artifact('uri2').mlmd_artifact
    executor_output_artifacts = {
        'artifact_key0':
            execution_result_pb2.ExecutorOutput.ArtifactList(artifacts=[]),
        'artifact_key1':
            execution_result_pb2.ExecutorOutput.ArtifactList(artifacts=[
                artifact0,
            ]),
        'artifact_key2':
            execution_result_pb2.ExecutorOutput.ArtifactList(artifacts=[
                artifact1,
                artifact2,
            ])
    }
    expected_output = {
        'artifact_key0': [],
        'artifact_key1': [artifact0],
        'artifact_key2': [artifact1, artifact2],
    }
    actual_output = data_types_utils.unpack_executor_output_artifacts(
        executor_output_artifacts)
    self.assertEqual(expected_output, actual_output)

  def testBuildArtifactStructDict(self):
    actual_artifact_struct_dict = data_types_utils.build_artifact_struct_dict(
        self.artifact_dict)
    self.assertEqual(self.artifact_struct_dict, actual_artifact_struct_dict)

  def testBuildValueDict(self):
    actual_value_dict = data_types_utils.build_value_dict(
        self.metadata_value_dict)
    self.assertEqual(self.value_dict, actual_value_dict)

  def testBuildMetadataValueDict(self):
    actual_metadata_value_dict = (
        data_types_utils.build_metadata_value_dict(self.value_dict))
    self.assertEqual(self.metadata_value_dict, actual_metadata_value_dict)

  def testBuildParsedValueDict(self):
    int_value = text_format.Parse(
        """
          field_value {
            int_value: 1
          }
        """, pipeline_pb2.Value())
    string_value = text_format.Parse(
        """
          field_value {
            string_value: 'random str'
          }
        """, pipeline_pb2.Value())
    bool_value = text_format.Parse(
        """
          field_value {
            string_value: 'false'
          }
          schema {
            value_type {
              boolean_type {}
            }
          }
        """, pipeline_pb2.Value())
    proto_value = text_format.Parse(
        """
          field_value {
            string_value: '{"string_value":"hello"}'
          }
          schema {
            value_type {
              proto_type {
                message_type: 'ml_metadata.Value'
              }
            }
          }
        """, pipeline_pb2.Value())
    list_boolean_value = text_format.Parse(
        """
          field_value {
            string_value: '[false, true]'
          }
          schema {
            value_type {
              list_type {
                boolean_type {}
              }
            }
          }
        """, pipeline_pb2.Value())
    list_str_value = text_format.Parse(
        """
          field_value {
            string_value: '["true", "false", "random"]'
          }
          schema {
            value_type {
              list_type {}
            }
          }
        """, pipeline_pb2.Value())
    value_dict = {
        'int_val': int_value,
        'string_val': string_value,
        'bool_val': bool_value,
        'proto_val': proto_value,
        'list_boolean_value': list_boolean_value,
        'list_str_value': list_str_value,
    }
    expected_parsed_dict = {
        'int_val': 1,
        'string_val': 'random str',
        'bool_val': False,
        'list_boolean_value': [False, True],
        'list_str_value': ['true', 'false', 'random'],
        'proto_val': metadata_store_pb2.Value(string_value='hello')
    }
    self.assertEqual(expected_parsed_dict,
                     data_types_utils.build_parsed_value_dict(value_dict))

  def testGetMetadataValueType(self):
    tfx_value = pipeline_pb2.Value()
    text_format.Parse(
        """
        field_value {
          int_value: 1
        }""", tfx_value)
    self.assertEqual(
        data_types_utils.get_metadata_value_type(tfx_value),
        metadata_store_pb2.INT)

    tfx_value = pipeline_pb2.Value()
    text_format.Parse(
        """
        field_value {
          proto_value: {
            type_url: "type.googleapis.com/ml_metadata.Artifact"
            value: ":\rartifact_name"
          }
        }""", tfx_value)
    self.assertEqual(
        data_types_utils.get_metadata_value_type(tfx_value),
        metadata_store_pb2.PROTO)

  def testGetMetadataValue(self):
    # Wrap an arbitrary proto message in an MLMD Value.
    original_proto_value = struct_pb2.Value(string_value='message in a proto')
    mlmd_value = metadata_store_pb2.Value()
    mlmd_value.proto_value.Pack(original_proto_value)
    # Get the raw metadata value, which should be a google.protobuf.Any type
    # since the property has a proto_value.
    raw_property_value = data_types_utils.get_metadata_value(mlmd_value)
    # Unpack the Any protobuf and compare against the original proto message.
    unpacked_value = proto_utils.unpack_proto_any(raw_property_value)
    self.assertEqual(unpacked_value.string_value, 'message in a proto')

  def testGetMetadataValueTypePrimitiveValue(self):
    self.assertEqual(
        data_types_utils.get_metadata_value_type(1), metadata_store_pb2.INT)

  def testGetMetadataValueTypeFailed(self):
    tfx_value = pipeline_pb2.Value()
    text_format.Parse(
        """
        runtime_parameter {
          name: 'rp'
        }""", tfx_value)
    with self.assertRaisesRegex(RuntimeError, 'Expecting field_value but got'):
      data_types_utils.get_metadata_value_type(tfx_value)

  def testGetValue(self):
    tfx_value = pipeline_pb2.Value()
    text_format.Parse(
        """
        field_value {
          int_value: 1
        }""", tfx_value)
    self.assertEqual(data_types_utils.get_value(tfx_value), 1)

  def testGetValueFailed(self):
    tfx_value = pipeline_pb2.Value()
    text_format.Parse(
        """
        runtime_parameter {
          name: 'rp'
        }""", tfx_value)
    with self.assertRaisesRegex(RuntimeError, 'Expecting field_value but got'):
      data_types_utils.get_value(tfx_value)

  def testSetMetadataValueWithTfxValue(self):
    tfx_value = pipeline_pb2.Value()
    metadata_property = metadata_store_pb2.Value()
    text_format.Parse(
        """
        field_value {
            int_value: 1
        }""", tfx_value)
    data_types_utils.set_metadata_value(
        metadata_value=metadata_property, value=tfx_value)
    self.assertProtoEquals('int_value: 1', metadata_property)

  def testSetMetadataValueWithTfxValueFailed(self):
    tfx_value = pipeline_pb2.Value()
    metadata_property = metadata_store_pb2.Value()
    text_format.Parse(
        """
        runtime_parameter {
          name: 'rp'
        }""", tfx_value)
    with self.assertRaisesRegex(ValueError, 'Expecting field_value but got'):
      data_types_utils.set_metadata_value(
          metadata_value=metadata_property, value=tfx_value)

  @parameterized.named_parameters(
      ('IntValue', 42, metadata_store_pb2.Value(int_value=42)),
      ('FloatValue', 42.0, metadata_store_pb2.Value(double_value=42.0)),
      ('StrValue', '42', metadata_store_pb2.Value(string_value='42')),
      ('BooleanValue', True, metadata_store_pb2.Value(string_value='true')),
      ('ListValue', [1, 2], metadata_store_pb2.Value(string_value='[1, 2]')))
  def testSetMetadataValueWithPrimitiveValue(self, value, expected_pb):
    pb = metadata_store_pb2.Value()
    data_types_utils.set_metadata_value(pb, value)
    self.assertEqual(pb, expected_pb)

  def testSetParameterValue(self):
    actual_int = pipeline_pb2.Value()
    expected_int = text_format.Parse(
        """
          field_value {
            int_value: 1
          }
        """, pipeline_pb2.Value())
    self.assertEqual(expected_int,
                     data_types_utils.set_parameter_value(actual_int, 1))

    actual_str = pipeline_pb2.Value()
    expected_str = text_format.Parse(
        """
          field_value {
            string_value: 'hello'
          }
        """, pipeline_pb2.Value())
    self.assertEqual(expected_str,
                     data_types_utils.set_parameter_value(actual_str, 'hello'))

    actual_bool = pipeline_pb2.Value()
    expected_bool = text_format.Parse(
        """
          field_value {
            string_value: 'true'
          }
          schema {
            value_type {
              boolean_type {}
            }
          }
        """, pipeline_pb2.Value())
    self.assertEqual(expected_bool,
                     data_types_utils.set_parameter_value(actual_bool, True))

    actual_proto = pipeline_pb2.Value()
    expected_proto = text_format.Parse(
        """
          field_value {
            string_value: '{\\n  "string_value": "hello"\\n}'
          }
          schema {
            value_type {
              proto_type {
                message_type: 'ml_metadata.Value'
              }
            }
          }
        """, pipeline_pb2.Value())
    data_types_utils.set_parameter_value(
        actual_proto, metadata_store_pb2.Value(string_value='hello'))
    actual_proto.schema.value_type.proto_type.ClearField('file_descriptors')
    self.assertProtoPartiallyEquals(expected_proto, actual_proto)

    actual_list = pipeline_pb2.Value()
    expected_list = text_format.Parse(
        """
          field_value {
            string_value: '[false, true]'
          }
          schema {
            value_type {
              list_type {
                boolean_type {}
              }
            }
          }
        """, pipeline_pb2.Value())
    self.assertEqual(
        expected_list,
        data_types_utils.set_parameter_value(actual_list, [False, True]))

    actual_list = pipeline_pb2.Value()
    expected_list = text_format.Parse(
        """
          field_value {
            string_value: '["true", "false"]'
          }
          schema {
            value_type {
              list_type {}
            }
          }
        """, pipeline_pb2.Value())
    self.assertEqual(
        expected_list,
        data_types_utils.set_parameter_value(actual_list, ['true', 'false']))

  @parameterized.named_parameters(
      dict(
          testcase_name='_dict[str,int]',
          value={
              'a': 1,
              'b': 2
          },
          expected=r"""field_value {
                        string_value: '{\"a\": 1, \"b\": 2}'
                      }
                      schema {
                        value_type {
                          dict_type {
                          }
                        }
                      }"""),
      dict(
          testcase_name='_dict[str,float]',
          value={
              'a': 1.,
              'b': 2.
          },
          expected=r"""field_value {
                        string_value: '{\"a\": 1.0, \"b\": 2.0}'
                      }
                      schema {
                        value_type {
                          dict_type {
                          }
                        }
                      }"""),
      dict(
          testcase_name='_dict[str,list[bool]]',
          value={
              'a': [True, False],
              'b': [False]
          },
          expected=r"""field_value {
                        string_value: '{\"a\": \"[true, false]\", \"b\": \"[false]\"}'
                      }
                      schema {
                        value_type {
                          dict_type {
                            list_type {
                              boolean_type {
                              }
                            }
                          }
                        }
                      }"""),
      dict(
          testcase_name='_dict[str,dict[str,str]]',
          value={'a': {
              'a': 1,
              'b': 2
          }},
          expected=r"""field_value {
                        string_value: '{\"a\": \"{\\"a\\": 1, \\"b\\": 2}\"}'
                      }
                      schema {
                        value_type {
                          dict_type {
                            dict_type {
                            }
                          }
                        }
                      }"""),
      dict(
          testcase_name='_list[float]',
          value=[1., 2.],
          expected=r"""field_value {
                        string_value: '[1.0, 2.0]'
                      }
                      schema {
                        value_type {
                          list_type {
                          }
                        }
                      }"""),
      dict(
          testcase_name='_list[dict[str,bool]]',
          value=[{
              'a': True
          }, {
              'b': False
          }],
          expected=r"""field_value {
                        string_value: '[\"{\\"a\\": true}\", \"{\\"b\\": false}\"]'
                      }
                      schema {
                        value_type {
                          list_type {
                            dict_type {
                              boolean_type {
                              }
                            }
                          }
                        }
                      }"""),
      dict(
          testcase_name='_list[dict[str,list[int]]]',
          value=[{
              'a': [1, 2]
          }],
          expected=r"""field_value {
                        string_value: '[\"{\\"a\\": \\"[1, 2]\\"}\"]'
                      }
                      schema {
                        value_type {
                          list_type {
                            dict_type {
                              list_type {
                              }
                            }
                          }
                        }
                      }"""),
  )
  def testSetParameterValueJson(self, value, expected):
    actual_list = pipeline_pb2.Value()
    expected_list = pipeline_pb2.Value()
    text_format.Parse(expected, expected_list)
    self.assertEqual(expected_list,
                     data_types_utils.set_parameter_value(actual_list, value))
