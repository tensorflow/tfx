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
"""Tests for Graph Partitioning."""

import os
import tempfile
import tensorflow as tf

from create_complex_graph import save_examples_as_graphdefs
from graph_partition import _RemoteOpLayers
from graph_partition import get_graph_name_to_graph_def, partition_all_graphs


class RelationTest(tf.test.TestCase):
  """Tests for Relation."""
  def test_layers(self):
    """Construct an example and validate the layers."""
    remote_op_relations = {'a1': [], 'a2': [], 'b1': ['a1'],
                           'b2': ['a1', 'a2'], 'c1': ['b1'],
                           'c2': ['b1', 'a1', 'b2', 'a2']}
    desired_outputs = [{'a1', 'a2'}, {'b1', 'b2'}, {'c1', 'c2'}]

    order = _RemoteOpLayers(remote_op_relations)
    self.assertEqual(desired_outputs, list(order))


class PartitionTest(tf.test.TestCase):
  """Tests for graph partitioning."""

  def setUp(self):
    super().setUp()
    with tempfile.TemporaryDirectory() as temp_dir:
      # Save examples into a temporary directory
      save_examples_as_graphdefs(temp_dir)

      graph_name_to_filepath = {
          'main': os.path.join(temp_dir, 'main_graph.pb'),
          'remote_op_a': os.path.join(temp_dir, 'graph_a.pb'),
          'remote_op_b': os.path.join(temp_dir, 'graph_b.pb')}
      graph_name_to_outputs = {
          'main': ['AddN_1'],
          'remote_op_b': ['Add_1'],
          'remote_op_a': ['embedding_lookup/Identity']}

      graph_name_to_graph_def = get_graph_name_to_graph_def(
          graph_name_to_filepath)
      self.graph_name_to_specs = partition_all_graphs(
          graph_name_to_graph_def, graph_name_to_outputs)


  def test_subgraph_import_validity(self):
    """Try to import subgraphs and see if they're valid."""
    for execution_specs in self.graph_name_to_specs.values():
      for execution_spec in execution_specs:
        if not execution_spec.is_remote_op:
          graph = tf.Graph()
          with graph.as_default():
            tf.import_graph_def(execution_spec.subgraph)


  def test_subgraph_specs(self):
    """Validate a subgraph spec."""
    for execution_specs in self.graph_name_to_specs.values():
      for spec in execution_specs:
        if not spec.is_remote_op:
          all_nodes = self._get_node_names_from_subgraph(spec.subgraph)

          self.assertTrue(spec.output_names.issubset(spec.body_node_names))
          self.assertEqual(all_nodes,
                           spec.body_node_names.union(spec.input_names))

          for input_name in spec.input_names:
            self.assertNotIn(input_name, spec.body_node_names)

          for node_from_other_layer in spec.nodes_from_other_layers:
            self.assertNotIn(node_from_other_layer, spec.body_node_names)


  def _get_node_names_from_subgraph(self, subgraph):
    node_names = {node.name for node in subgraph.node}
    return node_names


  def test_remote_op_specs(self):
    """Validate a remote op spec."""
    for execution_specs in self.graph_name_to_specs.values():
      for spec in execution_specs:
        if spec.is_remote_op:
          self.assertIsNone(spec.subgraph)
          self.assertLen(spec.output_names, 1)
          self.assertLen(spec.body_node_names, 1)


if __name__ == '__main__':
  tf.test.main()
  