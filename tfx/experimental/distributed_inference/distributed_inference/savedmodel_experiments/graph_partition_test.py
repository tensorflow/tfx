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

import create_complex_graph
import graph_partition


class RelationTest(tf.test.TestCase):
  """Tests for Relation."""
  def test_layers(self):
    """Construct an example and validate the layers."""
    remote_op_relations = {'a1': [], 'a2': [], 'b1': ['a1'],
                           'b2': ['a1', 'a2'], 'c1': ['b1'],
                           'c2': ['b1', 'a1', 'b2', 'a2']}
    relations = graph_partition.Relations(remote_op_relations)

    self.assertFalse(relations.check_if_finished())
    self.assertFalse(relations.check_if_finished())

    self.assertEqual(relations.get_next_layer(), {'a1', 'a2'})
    self.assertFalse(relations.check_if_finished())
    self.assertEqual(relations.get_next_layer(), {'b1', 'b2'})
    self.assertFalse(relations.check_if_finished())
    self.assertEqual(relations.get_next_layer(), {'c1', 'c2'})

    self.assertTrue(relations.check_if_finished())
    self.assertTrue(relations.check_if_finished())


class PartitionTest(tf.test.TestCase):
  """Tests for graph partitioning."""

  def test_subgraph_import_validity(self):
    """Try to import SavedModels and see if they're valid."""
    with tempfile.TemporaryDirectory() as temp_dir:
      create_complex_graph.save_examples_as_savedmodels(temp_dir)

      op_to_filepath = {
          'remote_op_a': os.path.join(temp_dir, 'graph_a'),
          'remote_op_b': os.path.join(temp_dir, 'graph_b'),
          'main': os.path.join(temp_dir, 'main_graph')}

      op_to_graph_def, op_to_output_names = graph_partition.load_saved_models(
          op_to_filepath)
      op_to_directory_paths = graph_partition.partition_all_graphs(
          op_to_graph_def, op_to_output_names, temp_dir)

      for directory_paths in op_to_directory_paths.values():
        for directory_path in directory_paths:
          with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            meta_graph_def = tf.compat.v1.saved_model.load(
                sess,
                [tf.compat.v1.saved_model.tag_constants.SERVING],
                directory_path)
            meta_graph_def = meta_graph_def
            print(meta_graph_def.signature_def)


if __name__ == '__main__':
  tf.test.main()
  