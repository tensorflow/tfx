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
import unittest

import tensorflow as tf

from tfx.dsl.io import fileio
from tfx.experimental.distributed_inference.graphdef_experiments.subgraph_partitioning import create_complex_graph
from tfx.experimental.distributed_inference.graphdef_experiments.subgraph_partitioning import graph_partition
from google.protobuf import text_format


@unittest.skipIf(tf.__version__ < '2', 'Incompatible with TF1')
class RemoteOpLayerTest(tf.test.TestCase):
  """A test for the class _RemoteOpLayer."""

  def test_layers(self):
    """Validates the class through an example."""
    remote_op_relations = {
        'a1': [],
        'a2': [],
        'b1': ['a1'],
        'b2': ['a1', 'a2'],
        'c1': ['b1'],
        'c2': ['b1', 'a1', 'b2', 'a2']
    }
    desired_outputs = [{'a1', 'a2'}, {'b1', 'b2'}, {'c1', 'c2'}]

    order = graph_partition._RemoteOpLayers(remote_op_relations)
    self.assertEqual(desired_outputs, list(order))


@unittest.skipIf(tf.__version__ < '2', 'Incompatible with TF1')
class PartitionTest(tf.test.TestCase):
  """A set of tests for the graph partitioning library."""

  def setUp(self):
    """Sets up some example graphs and their partitions."""
    super().setUp()
    with tempfile.TemporaryDirectory() as temp_dir:
      # Save examples into a temporary directory
      create_complex_graph.save_examples_as_graphdefs(temp_dir)

      graph_name_to_filepath = {
          'main': os.path.join(temp_dir, 'main_graph.pb'),
          'remote_op_a': os.path.join(temp_dir, 'graph_a.pb'),
          'remote_op_b': os.path.join(temp_dir, 'graph_b.pb')
      }
      graph_name_to_outputs = {
          'main': ['AddN_1'],
          'remote_op_b': ['Add_1'],
          'remote_op_a': ['embedding_lookup/Identity']
      }

      graph_name_to_graph_def = graph_partition.get_graph_name_to_graph_def(
          graph_name_to_filepath)
      self.graph_name_to_specs = graph_partition.partition_all_graphs(
          graph_name_to_graph_def, graph_name_to_outputs)

  def test_subgraph_import_validity(self):
    """Tests if the partitioned subgraphs can be imported."""
    for execution_specs in self.graph_name_to_specs.values():
      for execution_spec in execution_specs:
        if execution_spec.is_remote_op:
          continue

        graph = tf.Graph()
        with graph.as_default():
          tf.import_graph_def(execution_spec.subgraph)

  def test_remote_op_specs(self):
    """Validates a remote op spec."""
    for execution_specs in self.graph_name_to_specs.values():
      for spec in execution_specs:
        if not spec.is_remote_op:
          continue

        self.assertIsNone(spec.subgraph)
        self.assertLen(spec.output_names, 1)

  def test_subgraphs_with_golden_set(self):
    """Checks if the partitioned subgraphs match the golden set."""
    for graph_name, specs in self.graph_name_to_specs.items():
      for spec in specs:
        if spec.is_remote_op:
          continue
        golden_graph_def = _get_golden_subgraph(graph_name, spec)
        # Compare node names instead of `GraphDef` protos because sets in
        # graph_partition are not ordered. If there are two nodes with the
        # same type in a subgraph layer, for example Add. Sometimes a node
        # may have name "Add" while other times have name "Add_1".
        self.assertEqual(
            _get_node_names(golden_graph_def), _get_node_names(spec.subgraph))


def _get_golden_subgraph(graph_name, spec):
  """Retrieves a corresponding golden subgraph."""
  filename = _generate_unique_filename(spec.input_names)
  filepath = os.path.join(
      os.path.dirname(__file__), 'testdata', graph_name, filename)

  graph_def = tf.compat.v1.GraphDef()
  with fileio.open(filepath, 'r') as f:
    text_format.Parse(f.read(), graph_def)
  return graph_def


def _generate_unique_filename(input_names):
  return 'input_names-%s.pbtxt' % ('-'.join(sorted(input_names)))


def _get_node_names(graph_def):
  return {node.name for node in graph_def.node}


if __name__ == '__main__':
  tf.test.main()
