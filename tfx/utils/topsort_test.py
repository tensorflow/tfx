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
"""Tests for tfx.utils.topsort."""

import attr
import tensorflow as tf
from tfx.utils import topsort


@attr.s
class Node:
  # Some users topsort PipelineNode protos, which are not hashable.
  # To accomodate this, do not make this dataclass hashable.
  name = attr.ib()
  upstream_nodes = attr.ib()
  downstream_nodes = attr.ib()


class TopsortTest(tf.test.TestCase):

  def test_topsorted_layers_DAG(self):
    nodes = [
        Node('A', [], ['B', 'C', 'D']),
        Node('B', ['A'], []),
        Node('C', ['A'], ['D']),
        Node('D', ['A', 'C', 'F'], ['E']),
        Node('E', ['D'], []),
        Node('F', [], ['D'])
    ]
    node_map = {node.name: node for node in nodes}
    layers = topsort.topsorted_layers(
        nodes,
        get_node_id_fn=lambda n: n.name,
        get_parent_nodes=(
            lambda n: [node_map[name] for name in n.upstream_nodes]),
        get_child_nodes=(
            lambda n: [node_map[name] for name in n.downstream_nodes]))
    self.assertEqual([['A', 'F'], ['B', 'C'], ['D'], ['E']],
                     [[node.name for node in layer] for layer in layers])

  def test_topsorted_layers_error_if_cycle(self):
    nodes = [
        Node('A', [], ['B', 'E']),
        Node('B', ['A', 'D'], ['C']),
        Node('C', ['B'], ['D']),
        Node('D', ['C'], ['B']),
        Node('E', ['A'], [])
    ]
    node_map = {node.name: node for node in nodes}
    with self.assertRaisesRegex(topsort.InvalidDAGError, 'Cycle detected.'):
      topsort.topsorted_layers(
          nodes,
          get_node_id_fn=lambda n: n.name,
          get_parent_nodes=(
              lambda n: [node_map[name] for name in n.upstream_nodes]),
          get_child_nodes=(
              lambda n: [node_map[name] for name in n.downstream_nodes]))

  def test_topsorted_layers_ignore_unknown_parent_node(self):
    nodes = [
        Node('A', [], ['B']),
        Node('B', ['A'], ['C']),
        Node('C', ['B'], []),
    ]
    node_map = {node.name: node for node in nodes}
    # Exclude node A. Node B now has a parent node 'A' that should be ignored.
    layers = topsort.topsorted_layers(
        [node_map['B'], node_map['C']],
        get_node_id_fn=lambda n: n.name,
        get_parent_nodes=(
            lambda n: [node_map[name] for name in n.upstream_nodes]),
        get_child_nodes=(
            lambda n: [node_map[name] for name in n.downstream_nodes]))
    self.assertEqual([['B'], ['C']],
                     [[node.name for node in layer] for layer in layers])

  def test_topsorted_layers_ignore_duplicate_parent_node(self):
    nodes = [
        Node('A', [], ['B']),
        Node('B', ['A', 'A'], []),  # Duplicate parent node 'A'
    ]
    node_map = {node.name: node for node in nodes}
    layers = topsort.topsorted_layers(
        nodes,
        get_node_id_fn=lambda n: n.name,
        get_parent_nodes=(
            lambda n: [node_map[name] for name in n.upstream_nodes]),
        get_child_nodes=(
            lambda n: [node_map[name] for name in n.downstream_nodes]))
    self.assertEqual([['A'], ['B']],
                     [[node.name for node in layer] for layer in layers])

  def test_topsorted_layers_ignore_unknown_child_node(self):
    nodes = [
        Node('A', [], ['B']),
        Node('B', ['A'], ['C']),
        Node('C', ['B'], []),
    ]
    node_map = {node.name: node for node in nodes}
    # Exclude node C. Node B now has a child node 'C' that should be ignored.
    layers = topsort.topsorted_layers(
        [node_map['A'], node_map['B']],
        get_node_id_fn=lambda n: n.name,
        get_parent_nodes=(
            lambda n: [node_map[name] for name in n.upstream_nodes]),
        get_child_nodes=(
            lambda n: [node_map[name] for name in n.downstream_nodes]))
    self.assertEqual([['A'], ['B']],
                     [[node.name for node in layer] for layer in layers])

  def test_topsorted_layers_ignore_duplicate_child_node(self):
    nodes = [
        Node('A', [], ['B', 'B']),  # Duplicate child node 'B'
        Node('B', ['A'], []),
    ]
    node_map = {node.name: node for node in nodes}
    layers = topsort.topsorted_layers(
        nodes,
        get_node_id_fn=lambda n: n.name,
        get_parent_nodes=(
            lambda n: [node_map[name] for name in n.upstream_nodes]),
        get_child_nodes=(
            lambda n: [node_map[name] for name in n.downstream_nodes]))
    self.assertEqual([['A'], ['B']],
                     [[node.name for node in layer] for layer in layers])

  def test_topsorted_layers_empty(self):
    layers = topsort.topsorted_layers(
        nodes=[],
        get_node_id_fn=lambda n: n.name,
        get_parent_nodes=lambda n: [],
        get_child_nodes=lambda n: [])
    self.assertEqual([], layers)


if __name__ == '__main__':
  tf.test.main()
