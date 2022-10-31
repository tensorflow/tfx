# Copyright 2022 Google LLC. All Rights Reserved.
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
"""Tests for tfx.orchestration.portable.input_resolution.input_graph_resolver."""
from unittest import mock

from absl.testing import parameterized
import tensorflow as tf
from tfx.dsl.components.common import resolver
from tfx.dsl.input_resolution import resolver_op
from tfx.dsl.input_resolution.ops import ops
from tfx.orchestration.portable.input_resolution import exceptions
from tfx.orchestration.portable.input_resolution import input_graph_resolver
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import standard_artifacts

from google.protobuf import text_format


class Integer(standard_artifacts.Integer):
  """Integer value artifact with convenient constructor."""

  def __init__(self, value):
    super().__init__()
    self._has_value = True
    self._modified = False
    self._value = value

  def __repr__(self):
    return f'Integer({self._value})'

  def __eq__(self, other):
    return isinstance(other, Integer) and self.value == other.value


@ops.testonly_register
class Add(
    resolver_op.ResolverOp,
    arg_data_types=(
        resolver_op.DataType.ARTIFACT_LIST,
        resolver_op.DataType.ARTIFACT_LIST),
    return_data_type=resolver_op.DataType.ARTIFACT_LIST):

  def apply(self, xs, ys):
    return [Integer(x.value + y.value) for x, y in zip(xs, ys)]


@ops.testonly_register
class Inc(
    resolver_op.ResolverOp,
    arg_data_types=(resolver_op.DataType.ARTIFACT_LIST,),
    return_data_type=resolver_op.DataType.ARTIFACT_LIST):
  offset = resolver_op.Property(type=int, default=1)

  def apply(self, xs):
    return [Integer(x.value + self.offset) for x in xs]


@ops.testonly_register
class Dot(
    resolver_op.ResolverOp,
    arg_data_types=(
        resolver_op.DataType.ARTIFACT_LIST,
        resolver_op.DataType.ARTIFACT_LIST),
    return_data_type=resolver_op.DataType.ARTIFACT_LIST):

  def apply(self, xs, ys):
    result = sum([x.value * y.value for x, y in zip(xs, ys)])
    return [Integer(result)]


@ops.testonly_register
class RenameStrategy(resolver.ResolverStrategy):

  def __init__(self, new_key: str):
    self._new_key = new_key

  def resolve_artifacts(self, store, input_dict):
    only_key = list(input_dict)[0]
    return {
        self._new_key: input_dict[only_key],
    }


class InputGraphResolverTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._mlmd_handle = mock.MagicMock()

  def parse_input_graph(self, input_graph_text):
    return text_format.Parse(input_graph_text, pipeline_pb2.InputGraph())

  def testBuildGraphFn_InputNode(self):
    input_graph = self.parse_input_graph("""
      nodes {
        key: "input_1"
        value {
          input_node {
            input_key: "x"
          }
          output_data_type: ARTIFACT_LIST
        }
      }
      result_node: "input_1"
    """)

    graph_fn, input_keys = input_graph_resolver.build_graph_fn(
        self._mlmd_handle, input_graph)

    self.assertEqual(input_keys, ['x'])

    result = graph_fn({
        'x': [Integer(42)]
    })
    self.assertEqual(result, [Integer(42)])

  def testBuildGraphFn_DictNode(self):
    input_graph = self.parse_input_graph("""
      nodes {
        key: "input_1"
        value {
          input_node {
            input_key: "x"
          }
          output_data_type: ARTIFACT_LIST
        }
      }
      nodes {
        key: "input_2"
        value {
          input_node {
            input_key: "y"
          }
          output_data_type: ARTIFACT_LIST
        }
      }
      nodes {
        key: "dict_3"
        value {
          dict_node {
            node_ids {
              key: "x_out"
              value: "input_1"
            }
            node_ids {
              key: "y_out"
              value: "input_2"
            }
          }
        }
      }
      result_node: "dict_3"
    """)

    graph_fn, input_keys = input_graph_resolver.build_graph_fn(
        self._mlmd_handle, input_graph)

    self.assertCountEqual(input_keys, ['x', 'y'])

    result = graph_fn({
        'x': [Integer(1)],
        'y': [Integer(2)],
    })
    self.assertEqual(result, {
        'x_out': [Integer(1)],
        'y_out': [Integer(2)],
    })

  def testBuildGraphFn_OpNode(self):
    input_graph = self.parse_input_graph("""
      nodes {
        key: "input_1"
        value {
          input_node {
            input_key: "x"
          }
          output_data_type: ARTIFACT_LIST
        }
      }
      nodes {
        key: "input_2"
        value {
          input_node {
            input_key: "y"
          }
          output_data_type: ARTIFACT_LIST
        }
      }
      nodes {
        key: "add_3"
        value {
          op_node {
            op_type: "Add"
            args {
              node_id: "input_1"
            }
            args {
              node_id: "input_2"
            }
          }
          output_data_type: ARTIFACT_LIST
        }
      }
      result_node: "add_3"
    """)

    graph_fn, input_keys = input_graph_resolver.build_graph_fn(
        self._mlmd_handle, input_graph)

    self.assertCountEqual(input_keys, ['x', 'y'])

    result = graph_fn({
        'x': [Integer(1)],
        'y': [Integer(2)],
    })
    self.assertEqual(result, [Integer(3)])

  def testBuildGraphFn_OpNode_StaticProperty(self):
    input_graph = self.parse_input_graph("""
      nodes {
        key: "input_1"
        value {
          input_node {
            input_key: "x"
          }
          output_data_type: ARTIFACT_LIST
        }
      }
      nodes {
        key: "inc_2"
        value {
          op_node {
            op_type: "Inc"
            args {
              node_id: "input_1"
            }
            kwargs {
              key: "offset"
              value: {
                value {
                  field_value {
                    int_value: 42
                  }
                }
              }
            }
          }
          output_data_type: ARTIFACT_LIST
        }
      }
      result_node: "inc_2"
    """)

    graph_fn, input_keys = input_graph_resolver.build_graph_fn(
        self._mlmd_handle, input_graph)

    self.assertEqual(input_keys, ['x'])

    result = graph_fn({'x': [Integer(1)]})
    self.assertEqual(result, [Integer(43)])

  def testBuildGraphFn_NonExistingResultNode(self):
    input_graph = self.parse_input_graph("""
      nodes {
        key: "input_1"
        value {
          input_node {
            input_key: "x"
          }
          output_data_type: ARTIFACT_LIST
        }
      }
      result_node: "meh"
    """)

    with self.assertRaisesRegex(
        exceptions.FailedPreconditionError, 'result_node meh does not exist'):
      input_graph_resolver.build_graph_fn(self._mlmd_handle, input_graph)

  def testBuildGraphFn_CyclicNodes(self):
    input_graph = self.parse_input_graph("""
      nodes {
        key: "inc_1"
        value {
          op_node {
            op_type: "Inc"
            args {
              node_id: "inc_2"
            }
            kwargs {
              key: "offset"
            }
          }
          output_data_type: ARTIFACT_LIST
        }
      }
      nodes {
        key: "inc_2"
        value {
          op_node {
            op_type: "Inc"
            args {
              node_id: "inc_1"
            }
            kwargs {
              key: "offset"
            }
          }
          output_data_type: ARTIFACT_LIST
        }
      }
      result_node: "inc_2"
    """)

    with self.assertRaisesRegex(
        exceptions.FailedPreconditionError, 'InputGraph has a cycle'):
      input_graph_resolver.build_graph_fn(self._mlmd_handle, input_graph)

  @parameterized.parameters(
      # (1 + 2) * (3 + (++4 * 5)) = 3 * 28 = 84
      (dict(a=1, b=2, c=3, d=4, e=5), 84),
      # (5 + 4) * (3 + (++2 * 1)) = 9 * 6 = 54
      (dict(a=5, b=4, c=3, d=2, e=1), 54)
  )
  def testBuildGraphFn_ComplexCase(self, raw_inputs, expected):
    # (a + b) * (c + ((++d) * e))
    input_graph = self.parse_input_graph("""
      nodes {
        key: "a"
        value {
          input_node {
            input_key: "a"
          }
          output_data_type: ARTIFACT_LIST
        }
      }
      nodes {
        key: "b"
        value {
          input_node {
            input_key: "b"
          }
          output_data_type: ARTIFACT_LIST
        }
      }
      nodes {
        key: "c"
        value {
          input_node {
            input_key: "c"
          }
          output_data_type: ARTIFACT_LIST
        }
      }
      nodes {
        key: "d"
        value {
          input_node {
            input_key: "d"
          }
          output_data_type: ARTIFACT_LIST
        }
      }
      nodes {
        key: "e"
        value {
          input_node {
            input_key: "e"
          }
          output_data_type: ARTIFACT_LIST
        }
      }
      nodes {
        key: "(a + b)"
        value {
          op_node {
            op_type: "Add"
            args {
              node_id: "a"
            }
            args {
              node_id: "b"
            }
          }
          output_data_type: ARTIFACT_LIST
        }
      }
      nodes {
        key: "++d"
        value {
          op_node {
            op_type: "Inc"
            args {
              node_id: "d"
            }
          }
          output_data_type: ARTIFACT_LIST
        }
      }
      nodes {
        key: "(++d * e)"
        value {
          op_node {
            op_type: "Dot"
            args {
              node_id: "++d"
            }
            args {
              node_id: "e"
            }
          }
          output_data_type: ARTIFACT_LIST
        }
      }
      nodes {
        key: "(c + (++d * e))"
        value {
          op_node {
            op_type: "Add"
            args {
              node_id: "c"
            }
            args {
              node_id: "(++d * e)"
            }
          }
          output_data_type: ARTIFACT_LIST
        }
      }
      nodes {
        key: "result"
        value {
          op_node {
            op_type: "Dot"
            args {
              node_id: "(a + b)"
            }
            args {
              node_id: "(c + (++d * e))"
            }
          }
          output_data_type: ARTIFACT_LIST
        }
      }
      result_node: "result"
    """)

    graph_fn, input_keys = input_graph_resolver.build_graph_fn(
        self._mlmd_handle, input_graph)

    self.assertCountEqual(input_keys, ['a', 'b', 'c', 'd', 'e'])
    inputs = {k: [Integer(v)] for k, v in raw_inputs.items()}
    result = graph_fn(inputs)
    self.assertEqual(result, [Integer(expected)])

  def testResolverStrategy(self):
    input_graph = self.parse_input_graph("""
      nodes {
        key: "input_1"
        value {
          input_node {
            input_key: "x"
          }
          output_data_type: ARTIFACT_LIST
        }
      }
      nodes {
        key: "dict_1"
        value {
          dict_node {
            node_ids {
              key: "x"
              value: "input_1"
            }
          }
        }
      }
      nodes {
        key: "op_1"
        value {
          op_node {
            op_type: "__main__.RenameStrategy"
            args {
              node_id: "dict_1"
            }
            kwargs {
              key: "new_key"
              value {
                value {
                  field_value {
                    string_value: "y"
                  }
                }
              }
            }
          }
        }
      }
      result_node: "op_1"
    """)

    graph_fn, input_keys = input_graph_resolver.build_graph_fn(
        self._mlmd_handle, input_graph)
    self.assertEqual(input_keys, ['x'])
    result = graph_fn({'x': [Integer(42)]})
    self.assertEqual(result, {'y': [Integer(42)]})


if __name__ == '__main__':
  tf.test.main()
