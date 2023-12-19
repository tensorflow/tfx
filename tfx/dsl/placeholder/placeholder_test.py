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
"""Tests for tfx.dsl.placeholder.placeholder."""

import copy
import os
from typing import Type, TypeVar

import tensorflow as tf
from tfx.dsl.placeholder import placeholder as ph
from tfx.dsl.placeholder import placeholder_base
from tfx.dsl.placeholder import proto_placeholder
from tfx.proto import transform_pb2
from tfx.proto.orchestration import execution_invocation_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.proto.orchestration import placeholder_pb2
from tfx.types import standard_component_specs

from google.protobuf import message
from google.protobuf import text_format
from ml_metadata.proto import metadata_store_pb2


_ExecutionInvocation = ph.create_proto(
    execution_invocation_pb2.ExecutionInvocation
)
_PipelineInfo = ph.create_proto(pipeline_pb2.PipelineInfo)
_PipelineNode = ph.create_proto(pipeline_pb2.PipelineNode)
_MetadataStoreValue = ph.create_proto(metadata_store_pb2.Value)
_StructuralRuntimeParameter = ph.create_proto(
    pipeline_pb2.StructuralRuntimeParameter
)
_StringOrRuntimeParameter = ph.create_proto(
    pipeline_pb2.StructuralRuntimeParameter.StringOrRuntimeParameter
)
_UpdateOptions = ph.create_proto(pipeline_pb2.UpdateOptions)


_P = TypeVar('_P', bound=message.Message)


def load_testdata(
    filename: str, proto_class: Type[_P] = placeholder_pb2.PlaceholderExpression
) -> _P:
  test_pb_filepath = os.path.join(
      os.path.dirname(__file__), 'testdata', filename
  )
  with open(test_pb_filepath) as text_pb_file:
    return text_format.ParseLines(text_pb_file, proto_class())


class PlaceholderTest(tf.test.TestCase):

  def _assert_placeholder_pb_equal_and_deepcopyable(
      self,
      placeholder: ph.Placeholder,
      expected_pb_str: str,
  ):
    """This function will delete the original copy of placeholder."""
    # Due to inclusion in types like ExecutableSpec, placeholders need to by
    # deepcopy()-able.
    placeholder_copy = copy.deepcopy(placeholder)
    expected_pb = text_format.Parse(expected_pb_str,
                                    placeholder_pb2.PlaceholderExpression())
    # The original placeholder is deleted to verify deepcopy works. If caller
    # needs to use an instance of placeholder after calling to this function,
    # we can consider returning placeholder_copy.
    del placeholder

    encoded_placeholder = placeholder_copy.encode()

    # Clear out the descriptors, which we don't want to assert (too verbose).
    create_proto_op = encoded_placeholder.operator.create_proto_op
    if create_proto_op.HasField('file_descriptors'):
      create_proto_op.ClearField('file_descriptors')

    self.assertProtoEquals(expected_pb, encoded_placeholder)

  def testArtifactUriWithDefault0Index(self):
    self._assert_placeholder_pb_equal_and_deepcopyable(
        ph.input('model').uri, """
        operator {
          artifact_uri_op {
            expression {
              operator {
                index_op {
                  expression {
                    placeholder {
                      type: INPUT_ARTIFACT
                      key: "model"
                    }
                  }
                  index: 0
                }
              }
            }
          }
        }
    """)

  def testArtifactPropertyWithIndex(self):
    self._assert_placeholder_pb_equal_and_deepcopyable(
        ph.input('model')[11].property('blessed'), """
        operator {
          artifact_property_op {
            expression {
              operator {
                index_op {
                  expression {
                    placeholder {
                      type: INPUT_ARTIFACT
                      key: "model"
                    }
                  }
                  index: 11
                }
              }
            }
            key: "blessed"
          }
        }
    """)

  def testArtifactPropertyDefault0Index(self):
    self._assert_placeholder_pb_equal_and_deepcopyable(
        ph.input('model').property('blessed'), """
        operator {
          artifact_property_op {
            expression {
              operator {
                index_op {
                  expression {
                    placeholder {
                      type: INPUT_ARTIFACT
                      key: "model"
                    }
                  }
                  index: 0
                }
              }
            }
            key: "blessed"
          }
        }
    """)

  def testArtifactCustomPropertyWithIndex(self):
    self._assert_placeholder_pb_equal_and_deepcopyable(
        ph.input('model')[11].custom_property('blessed'), """
        operator {
          artifact_property_op {
            expression {
              operator {
                index_op {
                  expression {
                    placeholder {
                      type: INPUT_ARTIFACT
                      key: "model"
                    }
                  }
                  index: 11
                }
              }
            }
            key: "blessed"
            is_custom_property: True
          }
        }
    """)

  def testArtifactCustomPropertyDefault0Index(self):
    self._assert_placeholder_pb_equal_and_deepcopyable(
        ph.input('model').custom_property('blessed'), """
        operator {
          artifact_property_op {
            expression {
              operator {
                index_op {
                  expression {
                    placeholder {
                      type: INPUT_ARTIFACT
                      key: "model"
                    }
                  }
                  index: 0
                }
              }
            }
            key: "blessed"
            is_custom_property: True
          }
        }
    """)

  def testArtifactUriWithIndex(self):
    self._assert_placeholder_pb_equal_and_deepcopyable(
        ph.input('model')[0].uri, """
        operator {
          artifact_uri_op {
            expression {
              operator {
                index_op {
                  expression {
                    placeholder {
                      type: INPUT_ARTIFACT
                      key: "model"
                    }
                  }
                  index: 0
                }
              }
            }
          }
        }
    """)

  def testArtifactSplitUriWithIndex(self):
    self._assert_placeholder_pb_equal_and_deepcopyable(
        ph.input('model')[0].split_uri('train'), """
        operator {
          artifact_uri_op {
            expression {
              operator {
                index_op {
                  expression {
                    placeholder {
                      type: INPUT_ARTIFACT
                      key: "model"
                    }
                  }
                  index: 0
                }
              }
            }
            split: "train"
          }
        }
    """)

  def testPrimitiveArtifactValue(self):
    self._assert_placeholder_pb_equal_and_deepcopyable(
        ph.input('primitive').value, """
        operator {
          artifact_value_op {
            expression {
              operator {
                index_op {
                  expression {
                    placeholder {
                      type: INPUT_ARTIFACT
                      key: "primitive"
                    }
                  }
                  index: 0
                }
              }
            }
          }
        }
    """)

  def testPrimitiveArtifactValueWithIndexAccess(self):
    # If the value represented by a primitive artifact is intended to be a
    # JSON value, users can use index operator [] to access fields in the
    # deserialized JSON value.
    # In this example, the placeholder expression represents accessing the
    # following JSON value: { 'key': [...] }
    self._assert_placeholder_pb_equal_and_deepcopyable(
        ph.input('primitive').value['key'][0], """
        operator {
          index_op {
            expression {
              operator {
                index_op {
                  expression {
                    operator {
                      artifact_value_op {
                        expression {
                          operator {
                            index_op {
                              expression {
                                placeholder {
                                  type: INPUT_ARTIFACT
                                  key: "primitive"
                                }
                              }
                              index: 0
                            }
                          }
                        }
                      }
                    }
                  }
                  key: 'key'
                }
              }
            }
            index: 0
          }
        }
    """)

  def testRejectsValueOfOutputArtifact(self):
    with self.assertRaises(ValueError):
      _ = ph.output('primitive').value

  def testConcatUriWithString(self):
    self._assert_placeholder_pb_equal_and_deepcopyable(
        ph.output('model').uri + '/model', """
        operator {
          concat_op {
            expressions {
              operator {
                artifact_uri_op {
                  expression {
                    operator {
                      index_op {
                        expression {
                          placeholder {
                            type: OUTPUT_ARTIFACT
                            key: "model"
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
            expressions {
              value {
                string_value: "/model"
              }
            }
          }
        }
    """)

  def testExecPropertySimple(self):
    self._assert_placeholder_pb_equal_and_deepcopyable(
        ph.exec_property('num_train_steps'), """
        placeholder {
          type: EXEC_PROPERTY
          key: "num_train_steps"
        }
    """)

  def testExecPropertyProtoField(self):
    self._assert_placeholder_pb_equal_and_deepcopyable(
        ph.exec_property('proto')[0].a.b['c'], """
        operator {
          proto_op {
            expression {
              operator {
                index_op {
                  expression {
                    placeholder {
                      type: EXEC_PROPERTY
                      key: "proto"
                    }
                  }
                  index: 0
                }
              }
            }
            proto_field_path: ".a"
            proto_field_path: ".b"
            proto_field_path: "['c']"
          }
        }
    """)
    self._assert_placeholder_pb_equal_and_deepcopyable(
        ph.exec_property('proto').a['b'].c[1], """
        operator {
          index_op {
            expression {
              operator {
                proto_op {
                  expression {
                    placeholder {
                      type: EXEC_PROPERTY
                      key: "proto"
                    }
                  }
                  proto_field_path: ".a"
                  proto_field_path: "['b']"
                  proto_field_path: ".c"
                }
              }
            }
            index: 1
          }
        }
    """)

  def testExecPropertyListProtoSerialize(self):
    self._assert_placeholder_pb_equal_and_deepcopyable(
        ph.exec_property('list_proto').serialize_list(
            ph.ListSerializationFormat.JSON), """
        operator {
          list_serialization_op {
            expression {
              placeholder {
                type: EXEC_PROPERTY
                key: "list_proto"
              }
            }
            serialization_format: JSON
          }
        }
    """)

  def testExecPropertyListProtoIndex(self):
    self._assert_placeholder_pb_equal_and_deepcopyable(
        ph.exec_property('list_proto')[0].serialize(
            ph.ProtoSerializationFormat.JSON), """
        operator {
          proto_op {
            expression {
              operator {
                index_op {
                  expression {
                    placeholder {
                      type: EXEC_PROPERTY
                      key: "list_proto"
                    }
                  }
                  index: 0
                }
              }
            }
            serialization_format: JSON
          }
        }
    """)

  def testListConcat(self):
    self._assert_placeholder_pb_equal_and_deepcopyable(
        ph.to_list(
            [ph.input('model').uri, 'foo', ph.exec_property('random_str')]
        )
        + ph.to_list([ph.input('another_model').uri]),
        """
        operator {
          list_concat_op {
            expressions {
              operator {
                artifact_uri_op {
                  expression {
                    operator {
                      index_op {
                        expression {
                          placeholder {
                            type: INPUT_ARTIFACT
                            key: "model"
                          }
                        }
                        index: 0
                      }
                    }
                  }
                }
              }
            }
            expressions {
              value {
                string_value: "foo"
              }
            }
            expressions {
              placeholder {
                type: EXEC_PROPERTY
                key: "random_str"
              }
            }
            expressions {
              operator {
                artifact_uri_op {
                  expression {
                    operator {
                      index_op {
                        expression {
                          placeholder {
                            type: INPUT_ARTIFACT
                            key: "another_model"
                          }
                        }
                        index: 0
                      }
                    }
                  }
                }
              }
            }
          }
        }
    """,
    )

  def testListConcatAndSerialize(self):
    self._assert_placeholder_pb_equal_and_deepcopyable(
        ph.to_list([ph.input('model').uri,
                    ph.exec_property('random_str')
                   ]).serialize_list(ph.ListSerializationFormat.JSON), """
        operator {
          list_serialization_op {
            expression {
              operator {
                list_concat_op {
                  expressions {
                    operator {
                      artifact_uri_op {
                        expression {
                          operator {
                            index_op {
                              expression {
                                placeholder {
                                  type: INPUT_ARTIFACT
                                  key: "model"
                                }
                              }
                              index: 0
                            }
                          }
                        }
                      }
                    }
                  }
                  expressions {
                    placeholder {
                      type: EXEC_PROPERTY
                      key: "random_str"
                    }
                  }
                }
              }
            }
            serialization_format: JSON
          }
        }
    """)

  def testListEmpty(self):
    self._assert_placeholder_pb_equal_and_deepcopyable(
        ph.to_list([]), """
        operator {
          list_concat_op {}
        }
    """)
    self._assert_placeholder_pb_equal_and_deepcopyable(
        ph.to_list([]) + ph.to_list([ph.exec_property('random_str')]), """
        operator {
          list_concat_op {
            expressions {
              placeholder {
                type: EXEC_PROPERTY
                key: "random_str"
              }
            }
          }
        }
    """)

  def testCreateDictWithConcat(self):
    self._assert_placeholder_pb_equal_and_deepcopyable(
        ph.to_dict([
            ('fookey', ph.input('model').uri),
            (ph.exec_property('random_str'), 'barvalue'),
        ])
        + ph.to_dict({
            'anotherkey': ph.input('another_model').uri,
            'droppedkey': None,
        }),
        """
        operator {
          create_dict_op {
            entries {
              key {
                value {
                  string_value: "fookey"
                }
              }
              value {
                operator {
                  artifact_uri_op {
                    expression {
                      operator {
                        index_op {
                          expression {
                            placeholder {
                              type: INPUT_ARTIFACT
                              key: "model"
                            }
                          }
                          index: 0
                        }
                      }
                    }
                  }
                }
              }
            }
            entries {
              key {
                placeholder {
                  type: EXEC_PROPERTY
                  key: "random_str"
                }
              }
              value {
                value {
                  string_value: "barvalue"
                }
              }
            }
            entries {
              key {
                value {
                  string_value: "anotherkey"
                }
              }
              value {
                operator {
                  artifact_uri_op {
                    expression {
                      operator {
                        index_op {
                          expression {
                            placeholder {
                              type: INPUT_ARTIFACT
                              key: "another_model"
                            }
                          }
                          index: 0
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
    """,
    )

  def testDictEmpty(self):
    self._assert_placeholder_pb_equal_and_deepcopyable(
        ph.to_dict({}),
        """
        operator {
          create_dict_op {}
        }
    """,
    )

  def testProtoOperatorDescriptor(self):
    placeholder = ph.exec_property('splits_config').analyze[0]
    component_spec = standard_component_specs.TransformSpec
    self.assertProtoEquals(
        load_testdata('proto_placeholder_operator.pbtxt'),
        placeholder.encode(component_spec),
    )

  def testConcatWithSelfReferences(self):
    # Tests that Placeholder operators should not mutate the placeholder.
    a = ph.output('model')
    b = a.property('bar')
    self._assert_placeholder_pb_equal_and_deepcopyable(
        '1' + a.uri + '2' + a.property('foo') + '3' + b, """
        operator {
          concat_op {
            expressions {
              value {
                string_value: "1"
              }
            }
            expressions {
              operator {
                artifact_uri_op {
                  expression {
                    operator {
                      index_op {
                        expression {
                          placeholder {
                            type: OUTPUT_ARTIFACT
                            key: "model"
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
            expressions {
              value {
                string_value: "2"
              }
            }
            expressions {
              operator {
                artifact_property_op {
                  expression {
                    operator {
                      index_op {
                        expression {
                          placeholder {
                            type: OUTPUT_ARTIFACT
                            key: "model"
                          }
                        }
                      }
                    }
                  }
                  key: "foo"
                }
              }
            }
            expressions {
              value {
                string_value: "3"
              }
            }
            expressions {
              operator {
                artifact_property_op {
                  expression {
                    operator {
                      index_op {
                        expression {
                          placeholder {
                            type: OUTPUT_ARTIFACT
                            key: "model"
                          }
                        }
                      }
                    }
                  }
                  key: "bar"
                }
              }
            }
          }
        }""")

  def testJoinPlaceholders(self):
    a = ph.output('model').uri
    self._assert_placeholder_pb_equal_and_deepcopyable(
        ph.join([a, '-', a, '+', a]), """
        operator {
          concat_op {
            expressions {
              operator {
                artifact_uri_op {
                  expression {
                    operator {
                      index_op {
                        expression {
                          placeholder {
                            type: OUTPUT_ARTIFACT
                            key: "model"
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
            expressions {
              value {
                string_value: "-"
              }
            }
            expressions {
              operator {
                artifact_uri_op {
                  expression {
                    operator {
                      index_op {
                        expression {
                          placeholder {
                            type: OUTPUT_ARTIFACT
                            key: "model"
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
            expressions {
              value {
                string_value: "+"
              }
            }
            expressions {
              operator {
                artifact_uri_op {
                  expression {
                    operator {
                      index_op {
                        expression {
                          placeholder {
                            type: OUTPUT_ARTIFACT
                            key: "model"
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }""")

  def testJoinPlaceholdersWithSeparator(self):
    a = ph.output('model').uri
    self._assert_placeholder_pb_equal_and_deepcopyable(
        ph.join([a, '-', a, a], separator=','), """
          operator {
            concat_op {
              expressions {
                operator {
                  artifact_uri_op {
                    expression {
                      operator {
                        index_op {
                          expression {
                            placeholder {
                              type: OUTPUT_ARTIFACT
                              key: "model"
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
              expressions {
                value {
                  string_value: ","
                }
              }
              expressions {
                value {
                  string_value: "-"
                }
              }
              expressions {
                value {
                  string_value: ","
                }
              }
              expressions {
                operator {
                  artifact_uri_op {
                    expression {
                      operator {
                        index_op {
                          expression {
                            placeholder {
                              type: OUTPUT_ARTIFACT
                              key: "model"
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
              expressions {
                value {
                  string_value: ","
                }
              }
              expressions {
                operator {
                  artifact_uri_op {
                    expression {
                      operator {
                        index_op {
                          expression {
                            placeholder {
                              type: OUTPUT_ARTIFACT
                              key: "model"
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }""")

  def testComplicatedConcat(self):
    self._assert_placeholder_pb_equal_and_deepcopyable(
        'google/' + ph.output('model').uri + '/model/' + '0/' +
        ph.exec_property('version'), """
        operator {
          concat_op {
            expressions {
              value {
                string_value: "google/"
              }
            }
            expressions {
              operator {
                artifact_uri_op {
                  expression {
                    operator {
                      index_op {
                        expression {
                          placeholder {
                            type: OUTPUT_ARTIFACT
                            key: "model"
                          }
                        }
                        index: 0
                      }
                    }
                  }
                }
              }
            }
            expressions {
              value {
                string_value: "/model/"
              }
            }
            expressions {
              value {
                string_value: "0/"
              }
            }
            expressions {
              placeholder {
                type: EXEC_PROPERTY
                key: "version"
              }
            }
          }
        }
    """)

  def testRuntimeInfoSimple(self):
    self._assert_placeholder_pb_equal_and_deepcopyable(
        ph.runtime_info('platform_config'), """
        placeholder {
          type: RUNTIME_INFO
          key: "platform_config"
        }
    """)

  def testRuntimeInfoInvalidKey(self):
    with self.assertRaises(ValueError):
      ph.runtime_info('invalid_key')  # pytype: disable=wrong-arg-types

  def testProtoSerializationOperator(self):
    self._assert_placeholder_pb_equal_and_deepcopyable(
        ph.exec_property('proto').serialize(ph.ProtoSerializationFormat.JSON),
        """
        operator {
          proto_op {
            expression {
              placeholder {
                type: EXEC_PROPERTY
                key: "proto"
              }
            }
            serialization_format: JSON
          }
        }
        """)

  def testProtoSerializationOperatorWithFieldAccess(self):
    self._assert_placeholder_pb_equal_and_deepcopyable(
        ph.exec_property('proto').a.b.serialize(
            ph.ProtoSerializationFormat.JSON), """
        operator {
          proto_op {
            expression {
              operator {
                proto_op {
                  expression {
                    placeholder {
                      type: EXEC_PROPERTY
                      key: "proto"
                    }
                  }
                  proto_field_path: ".a"
                  proto_field_path: ".b"
                }
              }
            }
            serialization_format: JSON
          }
        }
        """)

  def testProtoSerializationWithDescriptor(self):
    placeholder = ph.exec_property('splits_config').serialize(
        ph.ProtoSerializationFormat.JSON)
    component_spec = standard_component_specs.TransformSpec
    self.assertProtoEquals(
        load_testdata('proto_placeholder_serialization_operator.pbtxt'),
        placeholder.encode(component_spec),
    )

  def testExecInvocation(self):
    self._assert_placeholder_pb_equal_and_deepcopyable(
        ph.execution_invocation().stateful_working_dir, """
        operator {
          proto_op {
            expression {
              placeholder {
                type: EXEC_INVOCATION
              }
            }
            proto_field_path: ".stateful_working_dir"
          }
        }
    """)

  def testEnvironmentVariable(self):
    self._assert_placeholder_pb_equal_and_deepcopyable(
        ph.environment_variable('FOO'), """
          placeholder {
            type: ENVIRONMENT_VARIABLE
            key: "FOO"
          }
    """)

  def testBase64EncodeOperator(self):
    self._assert_placeholder_pb_equal_and_deepcopyable(
        ph.exec_property('str_value').b64encode(), """
        operator {
          base64_encode_op {
            expression {
              placeholder {
                type: EXEC_PROPERTY
                key: "str_value"
              }
            }
          }
        }
    """)

  def testCreateProtoPlaceholder_Empty(self):
    self._assert_placeholder_pb_equal_and_deepcopyable(
        _ExecutionInvocation(),
        """
        operator {
          create_proto_op {
            base {
              [type.googleapis.com/tfx.orchestration.ExecutionInvocation] {}
            }
          }
        }
        """,
    )

  def testCreateProtoPlaceholder_BaseOnly(self):
    self._assert_placeholder_pb_equal_and_deepcopyable(
        _ExecutionInvocation(
            execution_invocation_pb2.ExecutionInvocation(tmp_dir='/foo')
        ),
        """
        operator {
          create_proto_op {
            base {
              [type.googleapis.com/tfx.orchestration.ExecutionInvocation] {
                tmp_dir: "/foo"
              }
            }
          }
        }
        """,
    )

  def testCreateProtoPlaceholder_FieldOnly(self):
    self._assert_placeholder_pb_equal_and_deepcopyable(
        _ExecutionInvocation(tmp_dir='/foo'),
        """
        operator {
          create_proto_op {
            base {
              [type.googleapis.com/tfx.orchestration.ExecutionInvocation] {}
            }
            fields {
              key: "tmp_dir"
              value {
                value {
                  string_value: "/foo"
                }
              }
            }
          }
        }
        """,
    )

  def testCreateProtoPlaceholder_FieldPlaceholder(self):
    self._assert_placeholder_pb_equal_and_deepcopyable(
        _ExecutionInvocation(tmp_dir=ph.exec_property('foo')),
        """
        operator {
          create_proto_op {
            base {
              [type.googleapis.com/tfx.orchestration.ExecutionInvocation] {}
            }
            fields {
              key: "tmp_dir"
              value {
                placeholder {
                  type: EXEC_PROPERTY
                  key: "foo"
                }
              }
            }
          }
        }
        """,
    )

  def testCreateProtoPlaceholder_RejectsUndefinedField(self):
    with self.assertRaisesRegex(ValueError, 'Unknown field undefined_field.*'):
      _ExecutionInvocation(undefined_field=ph.exec_property('foo'))

  def testCreateProtoPlaceholder_OtherFieldTypes(self):
    self._assert_placeholder_pb_equal_and_deepcopyable(
        _MetadataStoreValue(
            int_value=42,
            double_value=42.42,
            string_value='foo42',
            bool_value=True,
        ),
        """
        operator {
          create_proto_op {
            base {
              [type.googleapis.com/ml_metadata.Value] {}
            }
            fields {
              key: "int_value"
              value {
                value {
                  int_value: 42
                }
              }
            }
            fields {
              key: "double_value"
              value {
                value {
                  double_value: 42.42
                }
              }
            }
            fields {
              key: "string_value"
              value {
                value {
                  string_value: "foo42"
                }
              }
            }
            fields {
              key: "bool_value"
              value {
                value {
                  bool_value: true
                }
              }
            }
          }
        }
        """,
    )

  def testCreateProtoPlaceholder_EnumField(self):
    self._assert_placeholder_pb_equal_and_deepcopyable(
        _UpdateOptions(reload_policy=pipeline_pb2.UpdateOptions.ALL),
        """
        operator {
          create_proto_op {
            base {
              [type.googleapis.com/tfx.orchestration.UpdateOptions] {}
            }
            fields {
              key: "reload_policy"
              value {
                value {
                  int_value: 0
                }
              }
            }
          }
        }
        """,
    )

  def testCreateProtoPlaceholder_EnumFieldPlaceholder(self):
    self._assert_placeholder_pb_equal_and_deepcopyable(
        _UpdateOptions(reload_policy=ph.exec_property('foo')),
        """
        operator {
          create_proto_op {
            base {
              [type.googleapis.com/tfx.orchestration.UpdateOptions] {}
            }
            fields {
              key: "reload_policy"
              value {
                placeholder {
                  type: EXEC_PROPERTY
                  key: "foo"
                }
              }
            }
          }
        }
        """,
    )

  def testCreateProtoPlaceholder_RejectsSubmessageIntoScalarField(self):
    with self.assertRaisesRegex(
        ValueError, 'Expected scalar value for.*tmp_dir.*'
    ):
      _ExecutionInvocation(tmp_dir=_PipelineInfo())
    with self.assertRaisesRegex(
        ValueError, 'Expected scalar value for.*tmp_dir.*'
    ):
      _ExecutionInvocation(tmp_dir=pipeline_pb2.PipelineInfo())

  def testCreateProtoPlaceholder_RejectsWrongScalarType(self):
    with self.assertRaisesRegex(
        ValueError, 'Expected .*str.* for .*tmp_dir.*got 42'
    ):
      _ExecutionInvocation(tmp_dir=42)

  def testCreateProtoPlaceholder_SubmessagePlaceholder(self):
    self._assert_placeholder_pb_equal_and_deepcopyable(
        _ExecutionInvocation(
            pipeline_info=_PipelineInfo(id=ph.exec_property('foo'))
        ),
        """
        operator {
          create_proto_op {
            base {
              [type.googleapis.com/tfx.orchestration.ExecutionInvocation] {}
            }
            fields {
              key: "pipeline_info"
              value {
                operator {
                  create_proto_op {
                    base {
                      [type.googleapis.com/tfx.orchestration.PipelineInfo] {}
                    }
                    fields {
                      key: "id"
                      value {
                        placeholder {
                          type: EXEC_PROPERTY
                          key: "foo"
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
        """,
    )

  def testCreateProtoPlaceholder_RejectsScalarValueForSubmessageField(self):
    with self.assertRaisesRegex(
        ValueError, 'Expected submessage .*pipeline_info.*'
    ):
      _ExecutionInvocation(pipeline_info='this is not a submessage')

  def testCreateProtoPlaceholder_RejectsWrongTypeForSubmessageField(self):
    with self.assertRaisesRegex(
        ValueError, 'Expected message of type .*PipelineInfo.*pipeline_info.*'
    ):
      _ExecutionInvocation(pipeline_info=_PipelineNode())
    with self.assertRaisesRegex(
        ValueError, 'Expected message of type .*PipelineInfo.*pipeline_info.*'
    ):
      _ExecutionInvocation(pipeline_info=pipeline_pb2.PipelineNode())

  def testCreateProtoPlaceholder_RepeatedStringField(self):
    self._assert_placeholder_pb_equal_and_deepcopyable(
        _PipelineNode(
            upstream_nodes=['A', ph.exec_property('foo'), 'B'],
        ),
        """
        operator {
          create_proto_op {
            base {
              [type.googleapis.com/tfx.orchestration.PipelineNode] {}
            }
            fields {
              key: "upstream_nodes"
              value {
                operator {
                  list_concat_op {
                    expressions {
                      value {
                        string_value: "A"
                      }
                    }
                    expressions {
                      placeholder {
                        type: EXEC_PROPERTY
                        key: "foo"
                      }
                    }
                    expressions {
                      value {
                        string_value: "B"
                      }
                    }
                  }
                }
              }
            }
          }
        }
        """,
    )

  def testCreateProtoPlaceholder_RepeatedSubmessageField(self):
    self._assert_placeholder_pb_equal_and_deepcopyable(
        _StructuralRuntimeParameter(
            parts=[
                _StringOrRuntimeParameter(constant_value='A'),
                _StringOrRuntimeParameter(
                    constant_value=ph.exec_property('foo')
                ),
                _StringOrRuntimeParameter(constant_value='B'),
            ],
        ),
        """
        operator {
          create_proto_op {
            base {
              [type.googleapis.com/tfx.orchestration.StructuralRuntimeParameter] {}
            }
            fields {
              key: "parts"
              value {
                operator {
                  list_concat_op {
                    expressions {
                      operator {
                        create_proto_op {
                          base {
                            [type.googleapis.com/tfx.orchestration.StructuralRuntimeParameter.StringOrRuntimeParameter] {}
                          }
                          fields {
                            key: "constant_value"
                            value {
                              value {
                                string_value: "A"
                              }
                            }
                          }
                        }
                      }
                    }
                    expressions {
                      operator {
                        create_proto_op {
                          base {
                            [type.googleapis.com/tfx.orchestration.StructuralRuntimeParameter.StringOrRuntimeParameter] {}
                          }
                          fields {
                            key: "constant_value"
                            value {
                              placeholder {
                                type: EXEC_PROPERTY
                                key: "foo"
                              }
                            }
                          }
                        }
                      }
                    }
                    expressions {
                      operator {
                        create_proto_op {
                          base {
                            [type.googleapis.com/tfx.orchestration.StructuralRuntimeParameter.StringOrRuntimeParameter] {}
                          }
                          fields {
                            key: "constant_value"
                            value {
                              value {
                                string_value: "B"
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
        """,
    )

  def testCreateProtoPlaceholder_RejectsScalarValueForRepeatedField(self):
    with self.assertRaisesRegex(
        ValueError, 'Expected list input for repeated field .*upstream_nodes.*'
    ):
      _PipelineNode(upstream_nodes='this is not a list')

  def testCreateProtoPlaceholder_MapWithStringValueField(self):
    self._assert_placeholder_pb_equal_and_deepcopyable(
        _ExecutionInvocation(
            extra_flags={'a': 'A', 'foo': ph.exec_property('foo')},
        ),
        """
        operator {
          create_proto_op {
            base {
              [type.googleapis.com/tfx.orchestration.ExecutionInvocation] {}
            }
            fields {
              key: "extra_flags"
              value {
                operator {
                  create_dict_op {
                    entries {
                      key {
                        value {
                          string_value: "a"
                        }
                      }
                      value {
                        value {
                          string_value: "A"
                        }
                      }
                    }
                    entries {
                      key {
                        value {
                          string_value: "foo"
                        }
                      }
                      value {
                        placeholder {
                          type: EXEC_PROPERTY
                          key: "foo"
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
        """,
    )

  def testCreateProtoPlaceholder_MapWithSubmessageValueField(self):
    self._assert_placeholder_pb_equal_and_deepcopyable(
        _ExecutionInvocation(
            execution_properties={
                'fookey': _MetadataStoreValue(
                    string_value=ph.exec_property('fooprop')
                ),
                'dropped': None,
            },
        ),
        """
        operator {
          create_proto_op {
            base {
              [type.googleapis.com/tfx.orchestration.ExecutionInvocation] {}
            }
            fields {
              key: "execution_properties"
              value {
                operator {
                  create_dict_op {
                    entries {
                      key {
                        value {
                          string_value: "fookey"
                        }
                      }
                      value {
                        operator {
                          create_proto_op {
                            base {
                              [type.googleapis.com/ml_metadata.Value] {}
                            }
                            fields {
                              key: "string_value"
                              value {
                                placeholder {
                                  type: EXEC_PROPERTY
                                  key: "fooprop"
                                }
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
        """,
    )

  def testCreateProtoPlaceholder_RejectsScalarValueForMapField(self):
    with self.assertRaisesRegex(
        ValueError, 'Expected dict.*input for map field.*extra_flags.*'
    ):
      _ExecutionInvocation(extra_flags='this is not a dict')

  def testCreateProtoPlaceholder_RejectsNonStringKeyForMapField(self):
    with self.assertRaisesRegex(
        ValueError, 'Expected string.*for dict key.*extra_flags.*'
    ):
      _ExecutionInvocation(extra_flags={42: 43})

  def testCreateProtoPlaceholder_GeneratesSplitConfig(self):
    # This is part one of a two-part test. This generates the
    # PlaceholderExpression including the proto descriptors for SplitConfig.
    # Part two is in proto_placeholder_test, where that descriptor isn't linked
    # into the default descriptor pool, so that the descriptors must be loaded
    # from the PlaceholderExpression.
    # The two parts are connected through the testdata file.
    placeholder = ph.create_proto(transform_pb2.SplitsConfig)(
        analyze=['foo', 'bar']
    ).serialize(ph.ProtoSerializationFormat.TEXT_FORMAT)
    self.assertProtoEquals(
        load_testdata('create_proto_placeholder.pbtxt'),
        placeholder.encode(),
    )

  def testTraverse(self):
    p = ('google/' + ph.runtime_info('platform_config').user + '/' +
         ph.output('model').uri + '/model/' + '0/' +
         ph.exec_property('version'))
    ph_types = [type(x) for x in p.traverse()]
    self.assertIn(ph.ArtifactPlaceholder, ph_types)
    self.assertIn(ph.ExecPropertyPlaceholder, ph_types)
    self.assertIn(ph.RuntimeInfoPlaceholder, ph_types)
    self.assertNotIn(ph.ChannelWrappedPlaceholder, ph_types)

  def testListTraverse(self):
    p = ph.to_list([
        ph.runtime_info('platform_config').user,
        ph.output('model').uri,
        ph.exec_property('version'),
    ])
    ph_types = [type(x) for x in p.traverse()]
    self.assertIn(ph.ArtifactPlaceholder, ph_types)
    self.assertIn(ph.ExecPropertyPlaceholder, ph_types)
    self.assertIn(ph.RuntimeInfoPlaceholder, ph_types)
    self.assertIn(ph.ListPlaceholder, ph_types)
    self.assertNotIn(ph.ChannelWrappedPlaceholder, ph_types)

  def testDictTraverse(self):
    p = ph.to_dict([
        ('key1', ph.runtime_info('platform_config').user),
        (ph.output('model').uri, ph.exec_property('version')),
    ])
    ph_types = [type(x) for x in p.traverse()]
    self.assertIn(ph.ArtifactPlaceholder, ph_types)
    self.assertIn(ph.ExecPropertyPlaceholder, ph_types)
    self.assertIn(ph.RuntimeInfoPlaceholder, ph_types)
    self.assertIn(ph.DictPlaceholder, ph_types)
    self.assertNotIn(ph.ChannelWrappedPlaceholder, ph_types)

  def testCreateProtoTraverse(self):
    p = _ExecutionInvocation(tmp_dir=ph.exec_property('foo'))
    ph_types = [type(x) for x in p.traverse()]
    self.assertIn(proto_placeholder.CreateProtoPlaceholder, ph_types)
    self.assertIn(ph.ExecPropertyPlaceholder, ph_types)
    self.assertNotIn(ph.ArtifactPlaceholder, ph_types)
    self.assertNotIn(ph.ChannelWrappedPlaceholder, ph_types)
    self.assertNotIn(ph.RuntimeInfoPlaceholder, ph_types)

  def testIterate(self):
    p = ph.input('model')
    with self.assertRaisesRegex(
        RuntimeError, 'Iterating over a placeholder is not supported. '
    ):
      # Iterate over a placeholder by mistake.
      for _ in p:
        break


class EncodeValueLikeTest(tf.test.TestCase):

  def testEncodesPlaceholder(self):
    self.assertProtoEquals(
        """
        placeholder {
          type: EXEC_PROPERTY
          key: "foo"
        }
        """,
        placeholder_base.encode_value_like(ph.exec_property('foo')),
    )

  def testEncodesInt(self):
    self.assertProtoEquals(
        """
        value {
          int_value: 42
        }
        """,
        placeholder_base.encode_value_like(42),
    )

  def testEncodesFloat(self):
    self.assertProtoEquals(
        """
        value {
          double_value: 42.42
        }
        """,
        placeholder_base.encode_value_like(42.42),
    )

  def testEncodesString(self):
    self.assertProtoEquals(
        """
        value {
          string_value: "foo"
        }
        """,
        placeholder_base.encode_value_like('foo'),
    )

  def testEncodesBool(self):
    self.assertProtoEquals(
        """
        value {
          bool_value: true
        }
        """,
        placeholder_base.encode_value_like(True),
    )

  def testFailsOnInvalidInput(self):
    with self.assertRaises(ValueError):
      placeholder_base.encode_value_like(self)


if __name__ == '__main__':
  tf.test.main()
