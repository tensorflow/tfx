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
from tfx.proto.orchestration import placeholder_pb2
from tfx.types import standard_component_specs
from tfx.utils import json_utils

from google.protobuf import message
from google.protobuf import text_format


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
    placeholder = ph.exec_property('splits_config').analyze[0]
    component_spec = standard_component_specs.TransformSpec
    self.assertProtoEquals(
        placeholder.encode(component_spec),
        load_testdata('proto_placeholder_operator.pbtxt'),
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
    placeholder = ph.exec_property('splits_config').serialize(
        ph.ProtoSerializationFormat.JSON)
    component_spec = standard_component_specs.TransformSpec
    self.assertProtoEquals(
        placeholder.encode(component_spec),
        load_testdata('proto_placeholder_serialization_operator.pbtxt'),
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

  def testJsonSerializable(self):
    json_text = json_utils.dumps(ph.input('model').uri)
    python_instance = json_utils.loads(json_text)
    self.assertEqual(ph.input('model').uri.encode(), python_instance.encode())

  def testTraverse(self):
    p = ('google/' + ph.runtime_info('platform_config').user + '/' +
         ph.output('model').uri + '/model/' + '0/' +
         ph.exec_property('version'))
    got_dict = {type(x): x for x in p.traverse()}
    self.assertCountEqual(
        {
            ph.ArtifactPlaceholder, ph.ExecPropertyPlaceholder,
            ph.RuntimeInfoPlaceholder
        }, got_dict.keys())


if __name__ == '__main__':
  tf.test.main()
