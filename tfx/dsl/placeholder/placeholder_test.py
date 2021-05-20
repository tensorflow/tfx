# Lint as: python3
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
import tensorflow as tf
from tfx.dsl.placeholder import placeholder as ph
from tfx.proto.orchestration import placeholder_pb2
from tfx.types import standard_component_specs

from google.protobuf import text_format


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
              placeholder {
                type: EXEC_PROPERTY
                key: "proto"
              }
            }
            proto_field_path: "[0]"
            proto_field_path: ".a"
            proto_field_path: ".b"
            proto_field_path: "['c']"
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
        ph.execution_invocation(), """
        placeholder {
          type: EXEC_INVOCATION
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

if __name__ == '__main__':
  tf.test.main()
