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

from absl.testing import parameterized
import tensorflow as tf
from tfx.dsl.placeholder import placeholder as ph
from tfx.proto.orchestration import placeholder_pb2
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs
from tfx.types.artifact import Artifact
from tfx.types.artifact import Property
from tfx.types.artifact import PropertyType
from tfx.types.channel import Channel
from tfx.utils import json_utils

from google.protobuf import text_format


class _MyType(Artifact):
  TYPE_NAME = 'MyTypeName'
  PROPERTIES = {
      'string_value': Property(PropertyType.STRING),
  }


class PlaceholderTest(tf.test.TestCase):

  def _assert_placeholder_pb_equal_and_deepcopyable(self, placeholder,
                                                    expected_pb_str):
    """This function will delete the original copy of placeholder."""
    placeholder_copy = copy.deepcopy(placeholder)
    expected_pb = text_format.Parse(expected_pb_str,
                                    placeholder_pb2.PlaceholderExpression())
    # The original placeholder is deleted to verify deepcopy works. If caller
    # needs to use an instance of placeholder after calling to this function,
    # we can consider returning placeholder_copy.
    del placeholder
    self.assertProtoEquals(placeholder_copy.encode(), expected_pb)

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
        ph.to_list([ph.input('model').uri,
                    ph.exec_property('random_str')]) +
        ph.to_list([ph.input('another_model').uri]), """
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
    """)

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

  def testProtoOperatorDescriptor(self):
    test_pb_filepath = os.path.join(
        os.path.dirname(__file__), 'testdata',
        'proto_placeholder_operator.pbtxt')
    with open(test_pb_filepath) as text_pb_file:
      expected_pb = text_format.ParseLines(
          text_pb_file, placeholder_pb2.PlaceholderExpression())
    placeholder = ph.exec_property('splits_config').analyze[0]
    component_spec = standard_component_specs.TransformSpec
    self.assertProtoEquals(placeholder.encode(component_spec), expected_pb)

  def testProtoFutureValueOperator(self):
    test_pb_filepath = os.path.join(
        os.path.dirname(__file__), 'testdata',
        'proto_placeholder_future_value_operator.pbtxt')
    with open(test_pb_filepath) as text_pb_file:
      expected_pb = text_format.ParseLines(
          text_pb_file, placeholder_pb2.PlaceholderExpression())
    output_channel = Channel(type=standard_artifacts.Integer)
    placeholder = output_channel.future()[0].value
    placeholder._key = '_component.num'
    self.assertProtoEquals(placeholder.encode(), expected_pb)

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
    print(ph.join([a, '-', a, a], separator=',').encode())
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
      ph.runtime_info('invalid_key')

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
    test_pb_filepath = os.path.join(
        os.path.dirname(__file__), 'testdata',
        'proto_placeholder_serialization_operator.pbtxt')
    with open(test_pb_filepath) as text_pb_file:
      expected_pb = text_format.ParseLines(
          text_pb_file, placeholder_pb2.PlaceholderExpression())
    placeholder = ph.exec_property('splits_config').serialize(
        ph.ProtoSerializationFormat.JSON)
    component_spec = standard_component_specs.TransformSpec
    self.assertProtoEquals(placeholder.encode(component_spec), expected_pb)

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

  def testJsonSerializable(self):
    json_text = json_utils.dumps(ph.input('model').uri)
    python_instance = json_utils.loads(json_text)
    self.assertEqual(ph.input('model').uri.encode(), python_instance.encode())


class ChannelWrappedPlaceholderTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'two_sides_placeholder',
          'left': Channel(type=_MyType).future().value,
          'right': Channel(type=_MyType).future().value,
      },
      {
          'testcase_name': 'left_side_placeholder_right_side_string',
          'left': Channel(type=_MyType).future().value,
          'right': '#',
      },
      {
          'testcase_name': 'left_side_string_right_side_placeholder',
          'left': 'http://',
          'right': Channel(type=_MyType).future().value,
      },
  )
  def testConcat(self, left, right):
    self.assertIsInstance(left + right, ph.ChannelWrappedPlaceholder)

  def testJoinWithSelf(self):
    left = Channel(type=_MyType).future().value
    right = Channel(type=_MyType).future().value
    self.assertIsInstance(ph.join([left, right]), ph.ChannelWrappedPlaceholder)

  def testEncodeWithKeys(self):
    channel = Channel(type=_MyType)
    channel_future = channel.future()[0].value
    actual_pb = channel_future.encode_with_keys(
        lambda channel: channel.type_name)
    expected_pb = text_format.Parse(
        """
      operator {
        artifact_value_op {
          expression {
            operator {
              index_op {
                expression {
                  placeholder {
                    key: "MyTypeName"
                  }
                }
              }
            }
          }
        }
      }
    """, placeholder_pb2.PlaceholderExpression())
    self.assertProtoEquals(actual_pb, expected_pb)
    self.assertIsNone(channel_future._key)


class PredicateTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'two_sides_placeholder',
          'left': Channel(type=_MyType).future().value,
          'right': Channel(type=_MyType).future().value,
          'expected_op': placeholder_pb2.ComparisonOperator.Operation.LESS_THAN,
          'expected_lhs_field': 'operator',
          'expected_rhs_field': 'operator',
      },
      {
          'testcase_name': 'left_side_placeholder_right_side_int',
          'left': Channel(type=_MyType).future().value,
          'right': 1,
          'expected_op': placeholder_pb2.ComparisonOperator.Operation.LESS_THAN,
          'expected_lhs_field': 'operator',
          'expected_rhs_field': 'value',
          'expected_rhs_value_type': 'int_value',
      },
      {
          'testcase_name': 'left_side_placeholder_right_side_float',
          'left': Channel(type=_MyType).future().value,
          'right': 1.1,
          'expected_op': placeholder_pb2.ComparisonOperator.Operation.LESS_THAN,
          'expected_lhs_field': 'operator',
          'expected_rhs_field': 'value',
          'expected_rhs_value_type': 'double_value',
      },
      {
          'testcase_name': 'left_side_placeholder_right_side_string',
          'left': Channel(type=_MyType).future().value,
          'right': 'one',
          'expected_op': placeholder_pb2.ComparisonOperator.Operation.LESS_THAN,
          'expected_lhs_field': 'operator',
          'expected_rhs_field': 'value',
          'expected_rhs_value_type': 'string_value',
      },
      {
          'testcase_name':
              'right_side_placeholder_left_side_int',
          'left':
              1,
          'right':
              Channel(type=_MyType).future().value,
          'expected_op':
              placeholder_pb2.ComparisonOperator.Operation.GREATER_THAN,
          'expected_lhs_field':
              'operator',
          'expected_rhs_field':
              'value',
          'expected_rhs_value_type':
              'int_value',
      },
      {
          'testcase_name':
              'right_side_placeholder_left_side_float',
          'left':
              1.1,
          'right':
              Channel(type=_MyType).future().value,
          'expected_op':
              placeholder_pb2.ComparisonOperator.Operation.GREATER_THAN,
          'expected_lhs_field':
              'operator',
          'expected_rhs_field':
              'value',
          'expected_rhs_value_type':
              'double_value',
      },
  )
  def testComparison(self,
                     left,
                     right,
                     expected_op,
                     expected_lhs_field,
                     expected_rhs_field,
                     expected_rhs_value_type=None):
    pred = left < right
    actual_pb = pred.encode()
    self.assertEqual(actual_pb.operator.compare_op.op, expected_op)
    self.assertTrue(
        actual_pb.operator.compare_op.lhs.HasField(expected_lhs_field))
    self.assertTrue(
        actual_pb.operator.compare_op.rhs.HasField(expected_rhs_field))
    if expected_rhs_value_type:
      self.assertTrue(
          actual_pb.operator.compare_op.rhs.value.HasField(
              expected_rhs_value_type))

  def testEquals(self):
    left = Channel(type=_MyType)
    right = Channel(type=_MyType)
    pred = left.future().value == right.future().value
    actual_pb = pred.encode()
    self.assertEqual(actual_pb.operator.compare_op.op,
                     placeholder_pb2.ComparisonOperator.Operation.EQUAL)

  def testEncode(self):
    channel_1 = Channel(type=_MyType)
    channel_2 = Channel(type=_MyType)
    pred = channel_1.future().value > channel_2.future().value
    actual_pb = pred.encode()
    expected_pb = text_format.Parse(
        """
      operator {
        compare_op {
          lhs {
            operator {
              artifact_value_op {
                expression {
                  operator {
                    index_op {
                      expression {
                        placeholder {}
                      }
                    }
                  }
                }
              }
            }
          }
          rhs {
            operator {
              artifact_value_op {
                expression {
                  operator {
                    index_op {
                      expression {
                        placeholder {}
                      }
                    }
                  }
                }
              }
            }
          }
          op: GREATER_THAN
        }
      }
    """, placeholder_pb2.PlaceholderExpression())
    self.assertProtoEquals(actual_pb, expected_pb)

  def testEncodeWithKeys(self):
    channel_1 = Channel(type=_MyType)
    channel_2 = Channel(type=_MyType)
    pred = channel_1.future().value > channel_2.future().value
    channel_to_key_map = {
        channel_1: 'channel_1_key',
        channel_2: 'channel_2_key',
    }
    actual_pb = pred.encode_with_keys(
        lambda channel: channel_to_key_map[channel])
    expected_pb = text_format.Parse(
        """
      operator {
        compare_op {
          lhs {
            operator {
              artifact_value_op {
                expression {
                  operator {
                    index_op {
                      expression {
                        placeholder {
                          key: "channel_1_key"
                        }
                      }
                    }
                  }
                }
              }
            }
          }
          rhs {
            operator {
              artifact_value_op {
                expression {
                  operator {
                    index_op {
                      expression {
                        placeholder {
                          key: "channel_2_key"
                        }
                      }
                    }
                  }
                }
              }
            }
          }
          op: GREATER_THAN
        }
      }
    """, placeholder_pb2.PlaceholderExpression())
    self.assertProtoEquals(actual_pb, expected_pb)

  def testNegation(self):
    channel_1 = Channel(type=_MyType)
    channel_2 = Channel(type=_MyType)
    pred = channel_1.future().value < channel_2.future().value
    not_pred = ph.logical_not(pred)
    channel_to_key_map = {
        channel_1: 'channel_1_key',
        channel_2: 'channel_2_key',
    }
    actual_pb = not_pred.encode_with_keys(
        lambda channel: channel_to_key_map[channel])
    expected_pb = text_format.Parse(
        """
      operator {
        unary_logical_op {
          expression {
            operator {
              compare_op {
                lhs {
                  operator {
                    artifact_value_op {
                      expression {
                        operator {
                          index_op {
                            expression {
                              placeholder {
                                key: "channel_1_key"
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
                rhs {
                  operator {
                    artifact_value_op {
                      expression {
                        operator {
                          index_op {
                            expression {
                              placeholder {
                                key: "channel_2_key"
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
                op: LESS_THAN
              }
            }
          }
          op: NOT
        }
      }
    """, placeholder_pb2.PlaceholderExpression())
    self.assertProtoEquals(actual_pb, expected_pb)

  def testDoubleNegation(self):
    """Treat `not(not(a))` as `a`."""
    channel_1 = Channel(type=_MyType)
    channel_2 = Channel(type=_MyType)
    pred = channel_1.future().value < channel_2.future().value
    not_not_pred = ph.logical_not(ph.logical_not(pred))
    channel_to_key_map = {
        channel_1: 'channel_1_key',
        channel_2: 'channel_2_key',
    }
    actual_pb = not_not_pred.encode_with_keys(
        lambda channel: channel_to_key_map[channel])
    expected_pb = text_format.Parse(
        """
      operator {
        compare_op {
          lhs {
            operator {
              artifact_value_op {
                expression {
                  operator {
                    index_op {
                      expression {
                        placeholder {
                          key: "channel_1_key"
                        }
                      }
                    }
                  }
                }
              }
            }
          }
          rhs {
            operator {
              artifact_value_op {
                expression {
                  operator {
                    index_op {
                      expression {
                        placeholder {
                          key: "channel_2_key"
                        }
                      }
                    }
                  }
                }
              }
            }
          }
          op: LESS_THAN
        }
      }
    """, placeholder_pb2.PlaceholderExpression())
    self.assertProtoEquals(actual_pb, expected_pb)

  def testComparison_notEqual(self):
    """Treat `a != b` as `not(a == b)`."""
    channel_1 = Channel(type=_MyType)
    channel_2 = Channel(type=_MyType)
    pred = channel_1.future().value != channel_2.future().value
    channel_to_key_map = {
        channel_1: 'channel_1_key',
        channel_2: 'channel_2_key',
    }
    actual_pb = pred.encode_with_keys(
        lambda channel: channel_to_key_map[channel])
    expected_pb = text_format.Parse(
        """
      operator {
        unary_logical_op {
          expression {
            operator {
              compare_op {
                lhs {
                  operator {
                    artifact_value_op {
                      expression {
                        operator {
                          index_op {
                            expression {
                              placeholder {
                                key: "channel_1_key"
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
                rhs {
                  operator {
                    artifact_value_op {
                      expression {
                        operator {
                          index_op {
                            expression {
                              placeholder {
                                key: "channel_2_key"
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
                op: EQUAL
              }
            }
          }
          op: NOT
        }
      }
    """, placeholder_pb2.PlaceholderExpression())
    self.assertProtoEquals(actual_pb, expected_pb)

  def testComparison_lessThanOrEqual(self):
    """Treat `a <= b` as `not(a > b)`."""
    channel_1 = Channel(type=_MyType)
    channel_2 = Channel(type=_MyType)
    pred = channel_1.future().value <= channel_2.future().value
    channel_to_key_map = {
        channel_1: 'channel_1_key',
        channel_2: 'channel_2_key',
    }
    actual_pb = pred.encode_with_keys(
        lambda channel: channel_to_key_map[channel])
    expected_pb = text_format.Parse(
        """
      operator {
        unary_logical_op {
          expression {
            operator {
              compare_op {
                lhs {
                  operator {
                    artifact_value_op {
                      expression {
                        operator {
                          index_op {
                            expression {
                              placeholder {
                                key: "channel_1_key"
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
                rhs {
                  operator {
                    artifact_value_op {
                      expression {
                        operator {
                          index_op {
                            expression {
                              placeholder {
                                key: "channel_2_key"
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
                op: GREATER_THAN
              }
            }
          }
          op: NOT
        }
      }
    """, placeholder_pb2.PlaceholderExpression())
    self.assertProtoEquals(actual_pb, expected_pb)

  def testComparison_greaterThanOrEqual(self):
    """Treat `a >= b` as `not(a < b)`."""
    channel_1 = Channel(type=_MyType)
    channel_2 = Channel(type=_MyType)
    pred = channel_1.future().value >= channel_2.future().value
    channel_to_key_map = {
        channel_1: 'channel_1_key',
        channel_2: 'channel_2_key',
    }
    actual_pb = pred.encode_with_keys(
        lambda channel: channel_to_key_map[channel])
    expected_pb = text_format.Parse(
        """
      operator {
        unary_logical_op {
          expression {
            operator {
              compare_op {
                lhs {
                  operator {
                    artifact_value_op {
                      expression {
                        operator {
                          index_op {
                            expression {
                              placeholder {
                                key: "channel_1_key"
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
                rhs {
                  operator {
                    artifact_value_op {
                      expression {
                        operator {
                          index_op {
                            expression {
                              placeholder {
                                key: "channel_2_key"
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
                op: LESS_THAN
              }
            }
          }
          op: NOT
        }
      }
    """, placeholder_pb2.PlaceholderExpression())
    self.assertProtoEquals(actual_pb, expected_pb)

  def testNestedLogicalOps(self):
    channel_11 = Channel(type=_MyType)
    channel_12 = Channel(type=_MyType)
    channel_21 = Channel(type=_MyType)
    channel_22 = Channel(type=_MyType)
    channel_3 = Channel(type=_MyType)
    pred = ph.logical_or(
        ph.logical_and(channel_11.future().value >= channel_12.future().value,
                       channel_21.future().value < channel_22.future().value),
        ph.logical_not(channel_3.future().value == 'foo'))

    channel_to_key_map = {
        channel_11: 'channel_11_key',
        channel_12: 'channel_12_key',
        channel_21: 'channel_21_key',
        channel_22: 'channel_22_key',
        channel_3: 'channel_3_key',
    }
    actual_pb = pred.encode_with_keys(
        lambda channel: channel_to_key_map[channel])
    expected_pb = text_format.Parse(
        """
      operator {
        binary_logical_op {
          lhs {
            operator {
              binary_logical_op {
                lhs {
                  operator {
                    unary_logical_op {
                      expression {
                        operator {
                          compare_op {
                            lhs {
                              operator {
                                artifact_value_op {
                                  expression {
                                    operator {
                                      index_op {
                                        expression {
                                          placeholder {
                                            key: "channel_11_key"
                                          }
                                        }
                                      }
                                    }
                                  }
                                }
                              }
                            }
                            rhs {
                              operator {
                                artifact_value_op {
                                  expression {
                                    operator {
                                      index_op {
                                        expression {
                                          placeholder {
                                            key: "channel_12_key"
                                          }
                                        }
                                      }
                                    }
                                  }
                                }
                              }
                            }
                            op: LESS_THAN
                          }
                        }
                      }
                      op: NOT
                    }
                  }
                }
                rhs {
                  operator {
                    compare_op {
                      lhs {
                        operator {
                          artifact_value_op {
                            expression {
                              operator {
                                index_op {
                                  expression {
                                    placeholder {
                                      key: "channel_21_key"
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      }
                      rhs {
                        operator {
                          artifact_value_op {
                            expression {
                              operator {
                                index_op {
                                  expression {
                                    placeholder {
                                      key: "channel_22_key"
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      }
                      op: LESS_THAN
                    }
                  }
                }
                op: AND
              }
            }
          }
          rhs {
            operator {
              unary_logical_op {
                expression {
                  operator {
                    compare_op {
                      lhs {
                        operator {
                          artifact_value_op {
                            expression {
                              operator {
                                index_op {
                                  expression {
                                    placeholder {
                                      key: "channel_3_key"
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      }
                      rhs {
                        value {
                          string_value: "foo"
                        }
                      }
                      op: EQUAL
                    }
                  }
                }
                op: NOT
              }
            }
          }
          op: OR
        }
      }
    """, placeholder_pb2.PlaceholderExpression())
    self.assertProtoEquals(actual_pb, expected_pb)

  def testPredicateDependentChannels(self):
    int1 = Channel(type=standard_artifacts.Integer)
    int2 = Channel(type=standard_artifacts.Integer)
    pred1 = int1.future().value == 1
    pred2 = int1.future().value == int2.future().value
    pred3 = ph.logical_not(pred1)
    pred4 = ph.logical_and(pred1, pred2)

    self.assertEqual(set(pred1.dependent_channels()), {int1})
    self.assertEqual(set(pred2.dependent_channels()), {int1, int2})
    self.assertEqual(set(pred3.dependent_channels()), {int1})
    self.assertEqual(set(pred4.dependent_channels()), {int1, int2})

  def testPlaceholdersInvolved(self):
    p = ('google/' + ph.runtime_info('platform_config').user + '/' +
         ph.output('model').uri + '/model/' + '0/' +
         ph.exec_property('version'))
    got = p.placeholders_involved()
    got_dict = {type(x): x for x in got}
    self.assertCountEqual(
        {
            ph.ArtifactPlaceholder, ph.ExecPropertyPlaceholder,
            ph.RuntimeInfoPlaceholder
        }, got_dict.keys())


if __name__ == '__main__':
  tf.test.main()
