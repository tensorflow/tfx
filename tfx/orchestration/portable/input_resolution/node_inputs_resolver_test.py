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
from tfx.orchestration.portable.input_resolution import partition_utils
from tfx.proto.orchestration import pipeline_pb2
import tfx.types

from google.protobuf import text_format


def partition(**kwargs):
  return partition_utils.Partition(kwargs)


class DummyArtifact(tfx.types.Artifact):
  TYPE_NAME = 'DummyArtifact'

  def __init__(self, **kwargs):
    super().__init__()
    self._artifact.id = kwargs.pop('id')

  def __repr__(self):
    return f'<{self.id}>'


class DummyChannel(tfx.types.BaseChannel):

  def __init__(self, name):
    super().__init__(type=DummyArtifact)
    self.name = name


class ProtectedFunctionTest(tf.test.TestCase):

  def parse_node_inputs(self, node_inputs_text):
    return text_format.Parse(node_inputs_text, pipeline_pb2.NodeInputs())

  def testCheckCycle(self):

    def yes(nodes, dependencies):
      with self.assertRaises(exceptions.FailedPreconditionError):
        node_inputs_resolver._check_cycle(nodes, dependencies)

    def no(nodes, dependencies):
      try:
        node_inputs_resolver._check_cycle(nodes, dependencies)
      except exceptions.FailedPreconditionError:
        self.fail('Expected no cycle but has cycle.')

    no('', {})
    yes('a', {'a': 'a'})
    yes('ab', {'a': 'b', 'b': 'a'})
    yes('abc', {'a': 'b', 'b': 'c', 'c': 'a'})
    no('abcd', {'a': 'bcd', 'b': '', 'c': '', 'd': ''})
    no('abcd', {'a': 'bc', 'b': 'd', 'c': 'd', 'd': ''})

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

  def testJoinArtifacts(self):

    def check(entries_map, input_keys, expected):
      result = node_inputs_resolver._join_artifacts(entries_map, input_keys)
      self.assertEqual(result, expected)

    a1, a2, b1, b2, c1, c2, d1, d2, d3, d4, e1, e2 = [
        mock.MagicMock() for _ in range(12)]

    entries_map = {
        'a': [
            (partition(x=1), [a1]),
            (partition(x=2), [a2]),
        ],
        'b': [
            (partition(x=1), [b1]),
            (partition(x=2), [b2])
        ],
        'c': [(partition(), [c1, c2])],
        'd': [
            (partition(x=1, y=1), [d1]),
            (partition(x=1, y=2), [d2]),
            (partition(x=2, y=1), [d3]),
            (partition(x=2, y=2), [d4]),
        ],
        'e': [
            (partition(z=1), [e1]),
            (partition(z=2), [e2]),
        ]
    }

    check(entries_map, input_keys=['a', 'b'], expected=[
        (partition(x=1), {'a': [a1], 'b': [b1]}),
        (partition(x=2), {'a': [a2], 'b': [b2]}),
    ])

    check(entries_map, input_keys=['a', 'c'], expected=[
        (partition(x=1), {'a': [a1], 'c': [c1, c2]}),
        (partition(x=2), {'a': [a2], 'c': [c1, c2]}),
    ])

    check(entries_map, input_keys=['a', 'b', 'd'], expected=[
        (partition(x=1, y=1), {'a': [a1], 'b': [b1], 'd': [d1]}),
        (partition(x=1, y=2), {'a': [a1], 'b': [b1], 'd': [d2]}),
        (partition(x=2, y=1), {'a': [a2], 'b': [b2], 'd': [d3]}),
        (partition(x=2, y=2), {'a': [a2], 'b': [b2], 'd': [d4]}),
    ])

    check(entries_map, input_keys=['a', 'e'], expected=[
        (partition(x=1, z=1), {'a': [a1], 'e': [e1]}),
        (partition(x=1, z=2), {'a': [a1], 'e': [e2]}),
        (partition(x=2, z=1), {'a': [a2], 'e': [e1]}),
        (partition(x=2, z=2), {'a': [a2], 'e': [e2]}),
    ])

    # pylint: disable=line-too-long
    check(entries_map, input_keys=['a', 'b', 'c', 'd', 'e'], expected=[
        (partition(x=1, y=1, z=1), {'a': [a1], 'b': [b1], 'c': [c1, c2], 'd': [d1], 'e': [e1]}),
        (partition(x=1, y=1, z=2), {'a': [a1], 'b': [b1], 'c': [c1, c2], 'd': [d1], 'e': [e2]}),
        (partition(x=1, y=2, z=1), {'a': [a1], 'b': [b1], 'c': [c1, c2], 'd': [d2], 'e': [e1]}),
        (partition(x=1, y=2, z=2), {'a': [a1], 'b': [b1], 'c': [c1, c2], 'd': [d2], 'e': [e2]}),
        (partition(x=2, y=1, z=1), {'a': [a2], 'b': [b2], 'c': [c1, c2], 'd': [d3], 'e': [e1]}),
        (partition(x=2, y=1, z=2), {'a': [a2], 'b': [b2], 'c': [c1, c2], 'd': [d3], 'e': [e2]}),
        (partition(x=2, y=2, z=1), {'a': [a2], 'b': [b2], 'c': [c1, c2], 'd': [d4], 'e': [e1]}),
        (partition(x=2, y=2, z=2), {'a': [a2], 'b': [b2], 'c': [c1, c2], 'd': [d4], 'e': [e2]}),
    ])
    # pylint: enable=line-too-long


class NodeInputsResolverTest(tf.test.TestCase):
  maxDiff = None

  def setUp(self):
    super().setUp()
    self.addCleanup(mock.patch.stopall)
    # Here we mock resolution processes that requires MLMD store access because
    # they require too complex setup that are mostly irrelevent to the
    # node_inputs_resolver functionality. Each resolution submodule
    # (channel_resolver_test and input_graph_resolver_test) already contains
    # tests with more realistic examples.
    self._mlmd_handle = mock.MagicMock()

    # For each specific resolution process to mock, we directly store the side
    # effect mapping (argument and corresponding result). It is difficult to
    # use mock.return_value as we call mock method multiple times with different
    # arguments, expecting different results.

    # _channel_resolve_result stores InputSpec -> List[Artifact] mapping.
    self._channel_resolve_result = {}
    # _graph_fn_result stores InputGraph -> (GraphFn, List[str]) mapping.
    self._graph_fn_result = {}
    mock.patch.object(
        channel_resolver,
        'resolve_union_channels',
        self._mock_resolve_union_channels).start()
    mock.patch.object(
        input_graph_resolver,
        'build_graph_fn',
        self._mock_build_graph_fn).start()

  def mock_channel_resolution_result(self, input_spec, artifacts):
    assert len(input_spec.channels) == 1
    for channel in input_spec.channels:
      channel_key = text_format.MessageToString(channel, as_one_line=True)
      self._channel_resolve_result[channel_key] = artifacts

  def mock_graph_fn_result(self, input_graph, graph_fn, dependent_inputs=()):
    graph_key = text_format.MessageToString(input_graph, as_one_line=True)
    self._graph_fn_result[graph_key] = (graph_fn, dependent_inputs)

  def _mock_resolve_union_channels(self, store, channels):
    del store  # Unused.
    result = []
    for channel in channels:
      channel_key = text_format.MessageToString(channel, as_one_line=True)
      result.extend(self._channel_resolve_result[channel_key])
    return result

  def _mock_build_graph_fn(self, store, input_graph):
    del store  # Unused.
    graph_key = text_format.MessageToString(input_graph, as_one_line=True)
    return self._graph_fn_result[graph_key]

  def parse_input_spec(self, input_spec_text):
    return text_format.Parse(input_spec_text, pipeline_pb2.InputSpec())

  def parse_input_graph(self, input_graph_text):
    return text_format.Parse(input_graph_text, pipeline_pb2.InputGraph())

  def create_artifacts(self, n):
    if n == 1:
      return DummyArtifact(id=1)
    else:
      return [DummyArtifact(id=i) for i in range(1, n + 1)]

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
    self.mock_channel_resolution_result(x, [a1])
    node_inputs = pipeline_pb2.NodeInputs(inputs={'x': x})

    result = node_inputs_resolver.resolve(self._mlmd_handle, node_inputs)

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
    self.mock_channel_resolution_result(x1, [a1])
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
    self.mock_graph_fn_result(
        graph_1,
        lambda inputs: inputs['x1'],
        ['x1'])

    node_inputs = pipeline_pb2.NodeInputs(
        inputs={'x1': x1, 'x2': x2},
        input_graphs={'graph_1': graph_1})

    result = node_inputs_resolver.resolve(self._mlmd_handle, node_inputs)

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
    self.mock_graph_fn_result(
        graph_1,
        lambda inputs: {'x1_out': [a1], 'x2_out': [a2]})

    node_inputs = pipeline_pb2.NodeInputs(
        inputs={'x1': x1, 'x2': x2},
        input_graphs={'graph_1': graph_1})

    result = node_inputs_resolver.resolve(self._mlmd_handle, node_inputs)

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
    self.mock_graph_fn_result(
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
    self.mock_graph_fn_result(
        graph_2,
        lambda inputs: {'x2_out': inputs['x1']},
        ['x1'])

    node_inputs = pipeline_pb2.NodeInputs(
        inputs={'x1': x1, 'x2': x2},
        input_graphs={'graph_1': graph_1, 'graph_2': graph_2})

    result = node_inputs_resolver.resolve(self._mlmd_handle, node_inputs)

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
    self.mock_graph_fn_result(
        graph_1,
        lambda _: [{'a': [a1], 'b': [b1]}, {'a': [a2], 'b': [b2]}])

    node_inputs = pipeline_pb2.NodeInputs(
        inputs={'x1': x1, 'x2': x2},
        input_graphs={'graph_1': graph_1})

    result = node_inputs_resolver.resolve(self._mlmd_handle, node_inputs)

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
      node_inputs_resolver.resolve(self._mlmd_handle, node_inputs)

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
    self.mock_channel_resolution_result(x1, [a1])
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
    self.mock_graph_fn_result(
        graph_1,
        lambda _: [{'a': [a1]}, {'a': [a2]}])

    with self.subTest('UNION'):
      x3.mixed_inputs.method = pipeline_pb2.InputSpec.Mixed.Method.UNION
      node_inputs = pipeline_pb2.NodeInputs(
          inputs={'x1': x1, 'x2': x2, 'x3': x3},
          input_graphs={'graph_1': graph_1})

      result = node_inputs_resolver.resolve(self._mlmd_handle, node_inputs)

      self.assertEqual(result, [
          {'x1': [a1], 'x2': [a1], 'x3': [a1]},
          {'x1': [a1], 'x2': [a2], 'x3': [a1, a2]},
      ])

  def testHidden(self):
    a1, a2 = self.create_artifacts(2)
    x1 = self.parse_input_spec("""
      input_graph_ref {
        graph_id: "graph_1"
        key: "a"
      }
    """)
    x2 = self.parse_input_spec("""
      input_graph_ref {
        graph_id: "graph_2"
        key: "a"
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
    self.mock_graph_fn_result(
        graph_1,
        lambda _: [{'a': [a1]}, {'a': [a2]}])

    with self.subTest('Not hidden'):
      node_inputs = pipeline_pb2.NodeInputs(
          inputs={'x1': x1, 'x2': x2},
          input_graphs={'graph_1': graph_1, 'graph_2': graph_1})

      result = node_inputs_resolver.resolve(self._mlmd_handle, node_inputs)

      self.assertCountEqual(result, [
          {'x1': [a1], 'x2': [a1]},
          {'x1': [a1], 'x2': [a2]},
          {'x1': [a2], 'x2': [a1]},
          {'x1': [a2], 'x2': [a2]},
      ])

    with self.subTest('x2 is hidden'):
      x2.hidden = True
      node_inputs = pipeline_pb2.NodeInputs(
          inputs={'x1': x1, 'x2': x2},
          input_graphs={'graph_1': graph_1, 'graph_2': graph_1})

      result = node_inputs_resolver.resolve(self._mlmd_handle, node_inputs)

      self.assertCountEqual(result, [
          {'x1': [a1]},
          {'x1': [a2]},
      ])

    with self.subTest('x1, x2 are hidden'):
      x1.hidden = True
      node_inputs = pipeline_pb2.NodeInputs(
          inputs={'x1': x1, 'x2': x2},
          input_graphs={'graph_1': graph_1, 'graph_2': graph_1})

      result = node_inputs_resolver.resolve(self._mlmd_handle, node_inputs)

      self.assertEqual(result, [{}])

  def testConditionals(self):
    a1, a2, a3, a4 = self.create_artifacts(4)

    # Set blessed custom property
    a1.set_int_custom_property('blessed', 1)
    a2.set_int_custom_property('blessed', 0)
    a3.set_int_custom_property('blessed', 0)
    a4.set_int_custom_property('blessed', 1)

    # Set tag custom property
    a1.set_string_custom_property('tag', 'foo')
    a2.set_string_custom_property('tag', 'foo')
    a3.set_string_custom_property('tag', 'bar')
    a4.set_string_custom_property('tag', 'bar')

    x = self.parse_input_spec("""
      input_graph_ref {
        graph_id: "graph_1"
        key: "a"
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
    self.mock_graph_fn_result(
        graph_1,
        lambda _: [{'a': [a1]}, {'a': [a2]}, {'a': [a3]}, {'a': [a4]}])

    # Only allows artifact.custom_properties['blessed'] == 1,
    # which is a1 and a4.
    is_blessed = (
        DummyChannel('x').future()[0].custom_property('blessed') == 1
    ).encode_with_keys(lambda channel: channel.name)

    # Only allows artifact.custom_properties['tag'] == 'foo'
    # which is a1 and a2.
    is_foo = (
        (DummyChannel('x').future()[0].custom_property('tag') == 'foo')
    ).encode_with_keys(lambda channel: channel.name)

    cond_1 = pipeline_pb2.NodeInputs.Conditional(
        placeholder_expression=is_blessed)
    cond_2 = pipeline_pb2.NodeInputs.Conditional(
        placeholder_expression=is_foo)

    with self.subTest('blessed == 1'):
      node_inputs = pipeline_pb2.NodeInputs(
          inputs={'x': x},
          input_graphs={'graph_1': graph_1},
          conditionals={'cond_1': cond_1})

      result = node_inputs_resolver.resolve(self._mlmd_handle, node_inputs)
      self.assertEqual(result, [{'x': [a1]}, {'x': [a4]}])

    with self.subTest('blessed == 1 and tag == foo'):
      node_inputs = pipeline_pb2.NodeInputs(
          inputs={'x': x},
          input_graphs={'graph_1': graph_1},
          conditionals={'cond_1': cond_1, 'cond_2': cond_2})

      result = node_inputs_resolver.resolve(self._mlmd_handle, node_inputs)
      self.assertEqual(result, [{'x': [a1]}])

  def testMinCount(self):
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
    self.mock_channel_resolution_result(x1, [])
    node_inputs = pipeline_pb2.NodeInputs(inputs={'x1': x1})

    with self.subTest('min_count = 0'):
      node_inputs.inputs['x1'].min_count = 0
      result = node_inputs_resolver.resolve(self._mlmd_handle, node_inputs)
      self.assertEqual(result, [{'x1': []}])

    with self.subTest('min_count = 1'):
      node_inputs.inputs['x1'].min_count = 1
      with self.assertRaisesRegex(
          exceptions.FailedPreconditionError,
          r'inputs\[x1\] has min_count = 1 but only got 0 artifacts'):
        node_inputs_resolver.resolve(self._mlmd_handle, node_inputs)

    x2 = self.parse_input_spec("""
      input_graph_ref {
        graph_id: "graph_1"
        key: "a"
      }
      min_count: 1
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
    self.mock_graph_fn_result(
        graph_1,
        lambda _: [{'a': [a1]}, {'a': [a2]}, {'a': []}])

    with self.subTest('Multiple results should all satisfy min_count'):
      node_inputs = pipeline_pb2.NodeInputs(
          inputs={'x2': x2},
          input_graphs={'graph_1': graph_1})
      with self.assertRaisesRegex(
          exceptions.FailedPreconditionError,
          r'inputs\[x2\] has min_count = 1 but only got 0 artifacts'):
        node_inputs_resolver.resolve(self._mlmd_handle, node_inputs)


if __name__ == '__main__':
  tf.test.main()
