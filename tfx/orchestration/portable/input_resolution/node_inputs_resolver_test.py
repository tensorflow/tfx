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
"""Tests for tfx.orchestration.portable.input_resolution.node_inputs_resolver."""

from unittest import mock

import tensorflow as tf
from tfx.orchestration.portable.input_resolution import channel_resolver
from tfx.orchestration.portable.input_resolution import exceptions
from tfx.orchestration.portable.input_resolution import input_graph_resolver
from tfx.orchestration.portable.input_resolution import node_inputs_resolver
from tfx.proto.orchestration import pipeline_pb2

from google.protobuf import text_format


def key(**kwargs):
  return node_inputs_resolver._CompositeKey(kwargs)


class ProtectedFunctionTest(tf.test.TestCase):

  def parse_node_inputs(self, node_inputs_text):
    return text_format.Parse(node_inputs_text, pipeline_pb2.NodeInputs())

  def testTopologicallySortedInputKeys(self):
    node_inputs = self.parse_node_inputs("""
      inputs {
        key: "input_1"
        value {
          channels {
            artifact_query {
              type {
                name: "Examples"
              }
            }
          }
        }
      }
      inputs {
        key: "input_2"
        value {
          input_graph_ref {
            graph_id: "graph_1"
          }
        }
      }
      input_graphs {
        key: "graph_1"
        value {
          nodes {
            key: "node_1"
            value {
              input_node {
                input_key: "input_1"
              }
            }
          }
          result_node: "node_1"
        }
      }
    """)

    # input_2 uses graph_1
    # graph_1 depends on input_1
    # therefore input_2 depends on input_1.
    result = node_inputs_resolver._topologically_sorted_input_keys(
        node_inputs.inputs, node_inputs.input_graphs)
    self.assertEqual(result, ['input_1', 'input_2'])

  def testTopologicallySortedInputKeys_Cycle(self):
    node_inputs = self.parse_node_inputs("""
      inputs {
        key: "input_1"
        value {
          input_graph_ref {
            graph_id: "graph_1"
          }
        }
      }
      input_graphs {
        key: "graph_1"
        value {
          nodes {
            key: "node_1"
            value {
              input_node {
                input_key: "input_1"
              }
            }
          }
          result_node: "node_1"
        }
      }
    """)

    # input_1 uses graph_1
    # graph_1 depends on input_1
    # therefore input_1 depends on input_1 (self cycle).
    with self.assertRaisesRegex(
        exceptions.FailedPreconditionError, 'NodeInputs has a cycle'):
      node_inputs_resolver._topologically_sorted_input_keys(
          node_inputs.inputs, node_inputs.input_graphs)

  def testInnerJoin(self):

    def check(lhs, rhs, join_dims, expected, merge_fn=lambda x, y: x + y):
      with self.subTest(
          lhs=lhs, rhs=rhs, join_dims=join_dims, expected=expected):
        result = node_inputs_resolver._inner_join(
            lhs, rhs, join_dims, merge_fn=merge_fn)
        self.assertEqual(result, expected)

    check(
        lhs=[(key(), 'a'), (key(), 'b')],
        rhs=[(key(), '1'), (key(), '2')],
        join_dims=[],
        expected=[
            (key(), 'a1'),
            (key(), 'a2'),
            (key(), 'b1'),
            (key(), 'b2'),
        ]
    )

    check(
        lhs=[(key(), 'a'), (key(), 'b')],
        rhs=[],
        join_dims=[],
        expected=[]
    )

    check(
        lhs=[],
        rhs=[(key(), '1'), (key(), '2')],
        join_dims=[],
        expected=[]
    )

    check(
        lhs=[(key(x=1), 'x1'), (key(x=2), 'x2')],
        rhs=[(key(y=1), 'y1'), (key(y=2), 'y2')],
        join_dims=[],
        expected=[
            (key(x=1, y=1), 'x1y1'),
            (key(x=1, y=2), 'x1y2'),
            (key(x=2, y=1), 'x2y1'),
            (key(x=2, y=2), 'x2y2'),
        ]
    )

    check(
        lhs=[(key(x=1), 'a'), (key(x=2), 'b')],
        rhs=[(key(x=1), 'pple'), (key(x=2), 'anana')],
        join_dims=['x'],
        expected=[
            (key(x=1), 'apple'),
            (key(x=2), 'banana'),
        ]
    )

    check(
        lhs=[(key(x=1, z=1), 'x1'), (key(x=2, z=2), 'x2')],
        rhs=[(key(y=1, z=1), 'y1'), (key(y=2, z=2), 'y2')],
        join_dims=['z'],
        expected=[
            (key(x=1, y=1, z=1), 'x1y1'),
            (key(x=2, y=2, z=2), 'x2y2'),
        ]
    )

    check(
        lhs=[
            (key(x=1, y=1), 'x1y1'),
            (key(x=1, y=2), 'x1y2'),
            (key(x=2, y=1), 'x2y1'),
            (key(x=2, y=2), 'x2y2'),
        ],
        rhs=[
            (key(x=1, z=1), 'z1'),
            (key(x=2, z=2), 'z2'),
        ],
        join_dims=['x'],
        expected=[
            (key(x=1, y=1, z=1), 'x1y1z1'),
            (key(x=1, y=2, z=1), 'x1y2z1'),
            (key(x=2, y=1, z=2), 'x2y1z2'),
            (key(x=2, y=2, z=2), 'x2y2z2'),
        ]
    )

    check(
        lhs=[
            (key(x=1, y=1), 'x1y1'),
            (key(x=1, y=2), 'x1y2'),
            (key(x=2, y=1), 'x2y1'),
            (key(x=2, y=2), 'x2y2'),
        ],
        rhs=[
            (key(x=1, z=1), 'z1'),
            (key(x=1, z=2), 'z2'),
            (key(x=2, z=3), 'z3'),
            (key(x=2, z=4), 'z4'),
        ],
        join_dims=['x'],
        expected=[
            (key(x=1, y=1, z=1), 'x1y1z1'),
            (key(x=1, y=1, z=2), 'x1y1z2'),
            (key(x=1, y=2, z=1), 'x1y2z1'),
            (key(x=1, y=2, z=2), 'x1y2z2'),
            (key(x=2, y=1, z=3), 'x2y1z3'),
            (key(x=2, y=1, z=4), 'x2y1z4'),
            (key(x=2, y=2, z=3), 'x2y2z3'),
            (key(x=2, y=2, z=4), 'x2y2z4'),
        ]
    )

  def testJoinArtifacts(self):

    def check(entries_map, input_keys, expected):
      result = node_inputs_resolver._join_artifacts(entries_map, input_keys)
      self.assertEqual(result, expected)

    a1, a2, b1, b2, c1, c2, d1, d2, d3, d4, e1, e2 = [
        mock.MagicMock() for _ in range(12)]

    entries_map = {
        'a': [
            (key(x=1), [a1]),
            (key(x=2), [a2]),
        ],
        'b': [
            (key(x=1), [b1]),
            (key(x=2), [b2])
        ],
        'c': [(key(), [c1, c2])],
        'd': [
            (key(x=1, y=1), [d1]),
            (key(x=1, y=2), [d2]),
            (key(x=2, y=1), [d3]),
            (key(x=2, y=2), [d4]),
        ],
        'e': [
            (key(z=1), [e1]),
            (key(z=2), [e2]),
        ]
    }

    check(entries_map, input_keys=['a', 'b'], expected=[
        (key(x=1), {'a': [a1], 'b': [b1]}),
        (key(x=2), {'a': [a2], 'b': [b2]}),
    ])

    check(entries_map, input_keys=['a', 'c'], expected=[
        (key(x=1), {'a': [a1], 'c': [c1, c2]}),
        (key(x=2), {'a': [a2], 'c': [c1, c2]}),
    ])

    check(entries_map, input_keys=['a', 'b', 'd'], expected=[
        (key(x=1, y=1), {'a': [a1], 'b': [b1], 'd': [d1]}),
        (key(x=1, y=2), {'a': [a1], 'b': [b1], 'd': [d2]}),
        (key(x=2, y=1), {'a': [a2], 'b': [b2], 'd': [d3]}),
        (key(x=2, y=2), {'a': [a2], 'b': [b2], 'd': [d4]}),
    ])

    check(entries_map, input_keys=['a', 'e'], expected=[
        (key(x=1, z=1), {'a': [a1], 'e': [e1]}),
        (key(x=1, z=2), {'a': [a1], 'e': [e2]}),
        (key(x=2, z=1), {'a': [a2], 'e': [e1]}),
        (key(x=2, z=2), {'a': [a2], 'e': [e2]}),
    ])

    # pylint: disable=line-too-long
    check(entries_map, input_keys=['a', 'b', 'c', 'd', 'e'], expected=[
        (key(x=1, y=1, z=1), {'a': [a1], 'b': [b1], 'c': [c1, c2], 'd': [d1], 'e': [e1]}),
        (key(x=1, y=1, z=2), {'a': [a1], 'b': [b1], 'c': [c1, c2], 'd': [d1], 'e': [e2]}),
        (key(x=1, y=2, z=1), {'a': [a1], 'b': [b1], 'c': [c1, c2], 'd': [d2], 'e': [e1]}),
        (key(x=1, y=2, z=2), {'a': [a1], 'b': [b1], 'c': [c1, c2], 'd': [d2], 'e': [e2]}),
        (key(x=2, y=1, z=1), {'a': [a2], 'b': [b2], 'c': [c1, c2], 'd': [d3], 'e': [e1]}),
        (key(x=2, y=1, z=2), {'a': [a2], 'b': [b2], 'c': [c1, c2], 'd': [d3], 'e': [e2]}),
        (key(x=2, y=2, z=1), {'a': [a2], 'b': [b2], 'c': [c1, c2], 'd': [d4], 'e': [e1]}),
        (key(x=2, y=2, z=2), {'a': [a2], 'b': [b2], 'c': [c1, c2], 'd': [d4], 'e': [e2]}),
    ])
    # pylint: enable=line-too-long


class CompositeKeyTest(tf.test.TestCase):

  def testCompositeKey(self):
    k = node_inputs_resolver._CompositeKey({'x': 1, 'y': 2, 'z': 3})
    self.assertEqual(k.dims, ('x', 'y', 'z'))
    self.assertEqual(k.partial(['y', 'x']), (2, 1))
    self.assertEqual(k.partial([]), ())
    self.assertTrue(k)
    self.assertEqual(
        node_inputs_resolver._CompositeKey({'x': 1, 'y': 2}),
        node_inputs_resolver._CompositeKey({'y': 2, 'x': 1}))

  def testEmpty(self):
    empty = node_inputs_resolver._EMPTY
    self.assertEqual(empty.dims, ())
    self.assertEqual(empty.partial([]), ())
    self.assertFalse(empty)


class NodeInputsResolverTest(tf.test.TestCase):
  maxDiff = None

  def setUp(self):
    super().setUp()
    self.addCleanup(mock.patch.stopall)
    self._store = mock.MagicMock()
    self._channel_resolve_result = {}
    self._graph_fn_result = {}
    mock.patch.object(
        channel_resolver,
        'resolve_union_channels',
        self._resolve_union_channels).start()
    mock.patch.object(
        input_graph_resolver,
        'build_graph_fn',
        self._build_graph_fn).start()

  def mock_channel_resolution(self, input_spec, artifacts):
    for channel in input_spec.channels:
      channel_key = text_format.MessageToString(channel, as_one_line=True)
      self._channel_resolve_result[channel_key] = artifacts

  def mock_graph_fn(self, input_graph, graph_fn, dependent_inputs=()):
    graph_key = text_format.MessageToString(input_graph, as_one_line=True)
    self._graph_fn_result[graph_key] = (
        graph_fn, dependent_inputs)

  def _resolve_union_channels(self, store, channels):
    del store  # Unused.
    result = []
    for channel in channels:
      channel_key = text_format.MessageToString(channel, as_one_line=True)
      result.extend(self._channel_resolve_result[channel_key])
    return result

  def _build_graph_fn(self, store, input_graph):
    del store  # Unused.
    graph_key = text_format.MessageToString(input_graph, as_one_line=True)
    return self._graph_fn_result[graph_key]

  def parse_input_spec(self, input_spec_text):
    return text_format.Parse(input_spec_text, pipeline_pb2.InputSpec())

  def parse_input_graph(self, input_graph_text):
    return text_format.Parse(input_graph_text, pipeline_pb2.InputGraph())

  def create_artifacts(self, n):
    if n == 1:
      return mock.MagicMock(name='Artifact', id=1)
    else:
      return [mock.MagicMock(name=f'Artifact_{i}', id=i)
              for i in range(1, n + 1)]

  def testResolveChannels(self):
    a1 = self.create_artifacts(1)
    x = self.parse_input_spec("""
      channels {
        artifact_query {
          type {
            name: "MyArtifact"
          }
        }
      }
    """)
    self.mock_channel_resolution(x, [a1])
    node_inputs = pipeline_pb2.NodeInputs(inputs={'x': x})

    result = node_inputs_resolver.resolve(self._store, node_inputs)

    self.assertEqual(result, [{'x': [a1]}])

  def testResolveInputGraphRef_ArtifactList(self):
    a1 = self.create_artifacts(1)
    x1 = self.parse_input_spec("""
      channels {
        artifact_query {
          type {
            name: "MyArtifact"
          }
        }
      }
    """)
    self.mock_channel_resolution(x1, [a1])
    x2 = self.parse_input_spec("""
      input_graph_ref {
        graph_id: "graph_1"
      }
    """)
    graph_1 = self.parse_input_graph("""
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
    self.mock_graph_fn(
        graph_1,
        lambda inputs: inputs['x1'],
        ['x1'])

    node_inputs = pipeline_pb2.NodeInputs(
        inputs={'x1': x1, 'x2': x2},
        input_graphs={'graph_1': graph_1})

    result = node_inputs_resolver.resolve(self._store, node_inputs)

    self.assertEqual(result, [{'x1': [a1], 'x2': [a1]}])

  def testResolveInputGraphRef_ArtifactMultiMap(self):
    a1, a2 = self.create_artifacts(2)
    x1 = self.parse_input_spec("""
      input_graph_ref {
        graph_id: "graph_1"
        key: "x1_out"
      }
    """)
    x2 = self.parse_input_spec("""
      input_graph_ref {
        graph_id: "graph_1"
        key: "x2_out"
      }
    """)
    graph_1 = self.parse_input_graph("""
      nodes {
        key: "op_1"
        value {
          op_node {
            op_type: "DummyOp"
          }
          output_data_type: ARTIFACT_MULTIMAP
        }
      }
      result_node: "op_1"
    """)
    self.mock_graph_fn(
        graph_1,
        lambda inputs: {'x1_out': [a1], 'x2_out': [a2]})

    node_inputs = pipeline_pb2.NodeInputs(
        inputs={'x1': x1, 'x2': x2},
        input_graphs={'graph_1': graph_1})

    result = node_inputs_resolver.resolve(self._store, node_inputs)

    self.assertEqual(result, [{'x1': [a1], 'x2': [a2]}])

  def testResolveInputGraphRef_ArtifactMultiMap_MultipleInputs(self):
    a1, a2 = self.create_artifacts(2)
    x1 = self.parse_input_spec("""
      input_graph_ref {
        graph_id: "graph_1"
        key: "x1_out"
      }
    """)
    x2 = self.parse_input_spec("""
      input_graph_ref {
        graph_id: "graph_2"
        key: "x2_out"
      }
    """)
    graph_1 = self.parse_input_graph("""
      nodes {
        key: "op_1"
        value {
          op_node {
            op_type: "DummyOp"
          }
          output_data_type: ARTIFACT_MULTIMAP_LIST
        }
      }
      result_node: "op_1"
    """)
    self.mock_graph_fn(
        graph_1,
        lambda _: [{'x1_out': [a1]}, {'x1_out': [a2]}])

    graph_2 = self.parse_input_graph("""
      nodes {
        key: "input_1"
        value {
          input_node {
            input_key: "x1"
          }
        }
      }
      nodes {
        key: "op_1"
        value {
          op_node {
            op_type: "DummyOp"
            args {
              node_id: "input_1"
            }
          }
          output_data_type: ARTIFACT_MULTIMAP
        }
      }
      result_node: "op_1"
    """)
    self.mock_graph_fn(
        graph_2,
        lambda inputs: {'x2_out': inputs['x1']},
        ['x1'])

    node_inputs = pipeline_pb2.NodeInputs(
        inputs={'x1': x1, 'x2': x2},
        input_graphs={'graph_1': graph_1, 'graph_2': graph_2})

    result = node_inputs_resolver.resolve(self._store, node_inputs)

    self.assertEqual(result, [
        {'x1': [a1], 'x2': [a1]},
        {'x1': [a2], 'x2': [a2]},
    ])

  def testResolveInputGraphRef_ArtifactMultiMapList(self):
    a1, a2, b1, b2 = self.create_artifacts(4)
    x1 = self.parse_input_spec("""
      input_graph_ref {
        graph_id: "graph_1"
        key: "a"
      }
    """)
    x2 = self.parse_input_spec("""
      input_graph_ref {
        graph_id: "graph_1"
        key: "b"
      }
    """)
    graph_1 = self.parse_input_graph("""
      nodes {
        key: "op_1"
        value {
          op_node {
            op_type: "DummyOp"
          }
          output_data_type: ARTIFACT_MULTIMAP_LIST
        }
      }
      result_node: "op_1"
    """)
    self.mock_graph_fn(
        graph_1,
        lambda _: [{'a': [a1], 'b': [b1]}, {'a': [a2], 'b': [b2]}])

    node_inputs = pipeline_pb2.NodeInputs(
        inputs={'x1': x1, 'x2': x2},
        input_graphs={'graph_1': graph_1})

    result = node_inputs_resolver.resolve(self._store, node_inputs)

    self.assertEqual(result, [
        {'x1': [a1], 'x2': [b1]},
        {'x1': [a2], 'x2': [b2]},
    ])

  def testResolveInputGraphRef_InvalidGraphId(self):
    x = self.parse_input_spec("""
      input_graph_ref {
        graph_id: "non_existential_graph"
      }
    """)
    node_inputs = pipeline_pb2.NodeInputs(inputs={'x': x})

    with self.assertRaises(exceptions.FailedPreconditionError):
      node_inputs_resolver.resolve(self._store, node_inputs)

  def testMixedInputs(self):
    a1, a2 = self.create_artifacts(2)
    x1 = self.parse_input_spec("""
      channels {
        artifact_query {
          type {
            name: "MyArtifact"
          }
        }
      }
    """)
    self.mock_channel_resolution(x1, [a1])
    x2 = self.parse_input_spec("""
      input_graph_ref {
        graph_id: "graph_1"
        key: "a"
      }
    """)
    x3 = self.parse_input_spec("""
      mixed_inputs {
        input_keys: "x1"
        input_keys: "x2"
      }
    """)
    graph_1 = self.parse_input_graph("""
      nodes {
        key: "op_1"
        value {
          op_node {
            op_type: "DummyOp"
          }
          output_data_type: ARTIFACT_MULTIMAP_LIST
        }
      }
      result_node: "op_1"
    """)
    self.mock_graph_fn(
        graph_1,
        lambda _: [{'a': [a1]}, {'a': [a2]}])

    with self.subTest('UNION'):
      x3.mixed_inputs.method = pipeline_pb2.InputSpec.Mixed.Method.UNION
      node_inputs = pipeline_pb2.NodeInputs(
          inputs={'x1': x1, 'x2': x2, 'x3': x3},
          input_graphs={'graph_1': graph_1})

      result = node_inputs_resolver.resolve(self._store, node_inputs)

      self.assertEqual(result, [
          {'x1': [a1], 'x2': [a1], 'x3': [a1]},
          {'x1': [a1], 'x2': [a2], 'x3': [a1, a2]},
      ])

    with self.subTest('CONCAT'):
      x3.mixed_inputs.method = pipeline_pb2.InputSpec.Mixed.Method.CONCAT
      node_inputs = pipeline_pb2.NodeInputs(
          inputs={'x1': x1, 'x2': x2, 'x3': x3},
          input_graphs={'graph_1': graph_1})

      result = node_inputs_resolver.resolve(self._store, node_inputs)

      self.assertEqual(result, [
          {'x1': [a1], 'x2': [a1], 'x3': [a1, a1]},
          {'x1': [a1], 'x2': [a2], 'x3': [a1, a2]},
      ])


if __name__ == '__main__':
  tf.test.main()
