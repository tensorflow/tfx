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
"""Tests for tfx.types.channel.ChannelWrappedPlaceholder."""

import os
from typing import Type, TypeVar

from absl.testing import parameterized
import tensorflow as tf
from tfx.dsl.components.base.testing import test_node
from tfx.dsl.placeholder import placeholder as ph
from tfx.proto.orchestration import placeholder_pb2
from tfx.types import channel
from tfx.types import channel_utils
from tfx.types.artifact import Artifact
from tfx.types.artifact import Property
from tfx.types.artifact import PropertyType

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


class _MyType(Artifact):
  TYPE_NAME = 'MyTypeName'
  PROPERTIES = {
      'string_value': Property(PropertyType.STRING),
  }


class ChannelWrappedPlaceholderTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'two_sides_placeholder',
          'left': (
              channel.OutputChannel(
                  artifact_type=_MyType,
                  producer_component=test_node.TestNode('left'),
                  output_key='l',
              )
              .future()
              .value
          ),
          'right': (
              channel.OutputChannel(
                  artifact_type=_MyType,
                  producer_component=test_node.TestNode('right'),
                  output_key='r',
              )
              .future()
              .value
          ),
      },
      {
          'testcase_name': 'left_side_placeholder_right_side_string',
          'left': (
              channel.OutputChannel(
                  artifact_type=_MyType,
                  producer_component=test_node.TestNode('left'),
                  output_key='l',
              )
              .future()
              .value
          ),
          'right': '#',
      },
      {
          'testcase_name': 'left_side_string_right_side_placeholder',
          'left': 'http://',
          'right': (
              channel.OutputChannel(
                  artifact_type=_MyType,
                  producer_component=test_node.TestNode('right'),
                  output_key='r',
              )
              .future()
              .value
          ),
      },
  )
  def testConcat(self, left, right):
    self.assertIsInstance(left + right, ph.Placeholder)

  def testJoinWithSelf(self):
    left = (
        channel.OutputChannel(
            artifact_type=_MyType,
            producer_component=test_node.TestNode('producer'),
            output_key='foo',
        )
        .future()
        .value
    )
    right = (
        channel.OutputChannel(
            artifact_type=_MyType,
            producer_component=test_node.TestNode('producer'),
            output_key='foo',
        )
        .future()
        .value
    )
    self.assertIsInstance(ph.join([left, right]), ph.Placeholder)

  def testEncodeWithKeys(self):
    my_channel = channel.OutputChannel(
        artifact_type=_MyType,
        producer_component=test_node.TestNode('producer'),
        output_key='foo',
    )
    channel_future = my_channel.future()[0].value
    actual_pb = channel_utils.encode_placeholder_with_channels(
        channel_future, lambda c: c.type_name
    )
    expected_pb = text_format.Parse(
        """
      operator {
        artifact_value_op {
          expression {
            operator {
              index_op {
                expression {
                  placeholder {
                    key: "_producer.foo"
                  }
                }
              }
            }
          }
        }
      }
    """,
        placeholder_pb2.PlaceholderExpression(),
    )
    self.assertProtoEquals(actual_pb, expected_pb)


class PredicateTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'two_sides_placeholder',
          'left': (
              channel.OutputChannel(
                  artifact_type=_MyType,
                  producer_component=test_node.TestNode('producer'),
                  output_key='foo',
              )
              .future()
              .value
          ),
          'right': (
              channel.OutputChannel(
                  artifact_type=_MyType,
                  producer_component=test_node.TestNode('producer'),
                  output_key='foo',
              )
              .future()
              .value
          ),
          'expected_op': placeholder_pb2.ComparisonOperator.Operation.LESS_THAN,
          'expected_lhs_field': 'operator',
          'expected_rhs_field': 'operator',
      },
      {
          'testcase_name': 'left_side_placeholder_right_side_int',
          'left': (
              channel.OutputChannel(
                  artifact_type=_MyType,
                  producer_component=test_node.TestNode('producer'),
                  output_key='foo',
              )
              .future()
              .value
          ),
          'right': 1,
          'expected_op': placeholder_pb2.ComparisonOperator.Operation.LESS_THAN,
          'expected_lhs_field': 'operator',
          'expected_rhs_field': 'value',
          'expected_rhs_value_type': 'int_value',
      },
      {
          'testcase_name': 'left_side_placeholder_right_side_float',
          'left': (
              channel.OutputChannel(
                  artifact_type=_MyType,
                  producer_component=test_node.TestNode('producer'),
                  output_key='foo',
              )
              .future()
              .value
          ),
          'right': 1.1,
          'expected_op': placeholder_pb2.ComparisonOperator.Operation.LESS_THAN,
          'expected_lhs_field': 'operator',
          'expected_rhs_field': 'value',
          'expected_rhs_value_type': 'double_value',
      },
      {
          'testcase_name': 'left_side_placeholder_right_side_string',
          'left': (
              channel.OutputChannel(
                  artifact_type=_MyType,
                  producer_component=test_node.TestNode('producer'),
                  output_key='foo',
              )
              .future()
              .value
          ),
          'right': 'one',
          'expected_op': placeholder_pb2.ComparisonOperator.Operation.LESS_THAN,
          'expected_lhs_field': 'operator',
          'expected_rhs_field': 'value',
          'expected_rhs_value_type': 'string_value',
      },
      {
          'testcase_name': 'right_side_placeholder_left_side_int',
          'left': 1,
          'right': (
              channel.OutputChannel(
                  artifact_type=_MyType,
                  producer_component=test_node.TestNode('producer'),
                  output_key='foo',
              )
              .future()
              .value
          ),
          'expected_op': (
              placeholder_pb2.ComparisonOperator.Operation.GREATER_THAN
          ),
          'expected_lhs_field': 'operator',
          'expected_rhs_field': 'value',
          'expected_rhs_value_type': 'int_value',
      },
      {
          'testcase_name': 'right_side_placeholder_left_side_float',
          'left': 1.1,
          'right': (
              channel.OutputChannel(
                  artifact_type=_MyType,
                  producer_component=test_node.TestNode('producer'),
                  output_key='foo',
              )
              .future()
              .value
          ),
          'expected_op': (
              placeholder_pb2.ComparisonOperator.Operation.GREATER_THAN
          ),
          'expected_lhs_field': 'operator',
          'expected_rhs_field': 'value',
          'expected_rhs_value_type': 'double_value',
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
    left = channel.OutputChannel(
        artifact_type=_MyType,
        producer_component=test_node.TestNode('producer'),
        output_key='foo',
    )
    right = channel.OutputChannel(
        artifact_type=_MyType,
        producer_component=test_node.TestNode('producer'),
        output_key='foo',
    )
    pred = left.future().value == right.future().value
    actual_pb = pred.encode()
    self.assertEqual(actual_pb.operator.compare_op.op,
                     placeholder_pb2.ComparisonOperator.Operation.EQUAL)

  def testEncode(self):
    channel_1 = channel.OutputChannel(
        artifact_type=_MyType,
        producer_component=test_node.TestNode('a'),
        output_key='foo',
    )
    channel_2 = channel.OutputChannel(
        artifact_type=_MyType,
        producer_component=test_node.TestNode('b'),
        output_key='bar',
    )
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
                        placeholder {
                            key: "_a.foo"
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
                            key: "_b.bar"
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
    """,
        placeholder_pb2.PlaceholderExpression(),
    )
    self.assertProtoEquals(actual_pb, expected_pb)

  def testEncodeWithKeys(self):
    channel_1 = channel.OutputChannel(
        artifact_type=_MyType,
        producer_component=test_node.TestNode('a'),
        output_key='foo',
    )
    channel_2 = channel.OutputChannel(
        artifact_type=_MyType,
        producer_component=test_node.TestNode('b'),
        output_key='bar',
    )
    pred = channel_1.future().value > channel_2.future().value
    channel_to_key_map = {
        channel_1: 'channel_1_key',
        channel_2: 'channel_2_key',
    }
    actual_pb = channel_utils.encode_placeholder_with_channels(
        pred, lambda channel: channel_to_key_map[channel]
    )
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
                          key: "_a.foo"
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
                          key: "_b.bar"
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
    """,
        placeholder_pb2.PlaceholderExpression(),
    )
    self.assertProtoEquals(actual_pb, expected_pb)

  def testNegation(self):
    channel_1 = channel.OutputChannel(
        artifact_type=_MyType,
        producer_component=test_node.TestNode('a'),
        output_key='foo',
    )
    channel_2 = channel.OutputChannel(
        artifact_type=_MyType,
        producer_component=test_node.TestNode('b'),
        output_key='bar',
    )
    pred = channel_1.future().value < channel_2.future().value
    not_pred = ph.logical_not(pred)
    channel_to_key_map = {
        channel_1: 'channel_1_key',
        channel_2: 'channel_2_key',
    }
    actual_pb = channel_utils.encode_placeholder_with_channels(
        not_pred, lambda channel: channel_to_key_map[channel]
    )
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
                                key: "_a.foo"
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
                                key: "_b.bar"
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
    """,
        placeholder_pb2.PlaceholderExpression(),
    )
    self.assertProtoEquals(actual_pb, expected_pb)

  def testDoubleNegation(self):
    """Treat `not(not(a))` as `a`."""
    channel_1 = channel.OutputChannel(
        artifact_type=_MyType,
        producer_component=test_node.TestNode('a'),
        output_key='foo',
    )
    channel_2 = channel.OutputChannel(
        artifact_type=_MyType,
        producer_component=test_node.TestNode('b'),
        output_key='bar',
    )
    pred = channel_1.future().value < channel_2.future().value
    not_not_pred = ph.logical_not(ph.logical_not(pred))
    channel_to_key_map = {
        channel_1: 'channel_1_key',
        channel_2: 'channel_2_key',
    }
    actual_pb = channel_utils.encode_placeholder_with_channels(
        not_not_pred, lambda channel: channel_to_key_map[channel]
    )
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
                          key: "_a.foo"
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
                          key: "_b.bar"
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
    """,
        placeholder_pb2.PlaceholderExpression(),
    )
    self.assertProtoEquals(actual_pb, expected_pb)

  def testComparison_notEqual(self):
    """Treat `a != b` as `not(a == b)`."""
    channel_1 = channel.OutputChannel(
        artifact_type=_MyType,
        producer_component=test_node.TestNode('a'),
        output_key='foo',
    )
    channel_2 = channel.OutputChannel(
        artifact_type=_MyType,
        producer_component=test_node.TestNode('b'),
        output_key='bar',
    )
    pred = channel_1.future().value != channel_2.future().value
    channel_to_key_map = {
        channel_1: 'channel_1_key',
        channel_2: 'channel_2_key',
    }
    actual_pb = channel_utils.encode_placeholder_with_channels(
        pred, lambda channel: channel_to_key_map[channel]
    )
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
                                key: "_a.foo"
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
                                key: "_b.bar"
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
    """,
        placeholder_pb2.PlaceholderExpression(),
    )
    self.assertProtoEquals(actual_pb, expected_pb)

  def testComparison_lessThanOrEqual(self):
    """Treat `a <= b` as `not(a > b)`."""
    channel_1 = channel.OutputChannel(
        artifact_type=_MyType,
        producer_component=test_node.TestNode('a'),
        output_key='foo',
    )
    channel_2 = channel.OutputChannel(
        artifact_type=_MyType,
        producer_component=test_node.TestNode('b'),
        output_key='bar',
    )
    pred = channel_1.future().value <= channel_2.future().value
    channel_to_key_map = {
        channel_1: 'channel_1_key',
        channel_2: 'channel_2_key',
    }
    actual_pb = channel_utils.encode_placeholder_with_channels(
        pred, lambda channel: channel_to_key_map[channel]
    )
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
                                key: "_a.foo"
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
                                key: "_b.bar"
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
    """,
        placeholder_pb2.PlaceholderExpression(),
    )
    self.assertProtoEquals(actual_pb, expected_pb)

  def testComparison_greaterThanOrEqual(self):
    """Treat `a >= b` as `not(a < b)`."""
    channel_1 = channel.OutputChannel(
        artifact_type=_MyType,
        producer_component=test_node.TestNode('a'),
        output_key='foo',
    )
    channel_2 = channel.OutputChannel(
        artifact_type=_MyType,
        producer_component=test_node.TestNode('b'),
        output_key='bar',
    )
    pred = channel_1.future().value >= channel_2.future().value
    channel_to_key_map = {
        channel_1: 'channel_1_key',
        channel_2: 'channel_2_key',
    }
    actual_pb = channel_utils.encode_placeholder_with_channels(
        pred, lambda channel: channel_to_key_map[channel]
    )
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
                                key: "_a.foo"
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
                                key: "_b.bar"
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
    """,
        placeholder_pb2.PlaceholderExpression(),
    )
    self.assertProtoEquals(actual_pb, expected_pb)

  def testNestedLogicalOps(self):
    channel_11 = channel.OutputChannel(
        artifact_type=_MyType,
        producer_component=test_node.TestNode('a'),
        output_key='1',
    )
    channel_12 = channel.OutputChannel(
        artifact_type=_MyType,
        producer_component=test_node.TestNode('b'),
        output_key='2',
    )
    channel_21 = channel.OutputChannel(
        artifact_type=_MyType,
        producer_component=test_node.TestNode('c'),
        output_key='3',
    )
    channel_22 = channel.OutputChannel(
        artifact_type=_MyType,
        producer_component=test_node.TestNode('d'),
        output_key='4',
    )
    channel_3 = channel.OutputChannel(
        artifact_type=_MyType,
        producer_component=test_node.TestNode('e'),
        output_key='5',
    )
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
    actual_pb = channel_utils.encode_placeholder_with_channels(
        pred, lambda channel: channel_to_key_map[channel]
    )
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
                                            key: "_a.1"
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
                                            key: "_b.2"
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
                                      key: "_c.3"
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
                                      key: "_d.4"
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
                                      key: "_e.5"
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
    """,
        placeholder_pb2.PlaceholderExpression(),
    )
    self.assertProtoEquals(actual_pb, expected_pb)
