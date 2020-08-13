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

from tfx.experimental.distributed_inference.savedmodel_experiments import create_complex_graph
from tfx.experimental.distributed_inference.savedmodel_experiments import graph_partition


class PartitionTest(tf.test.TestCase):
  """Tests for graph partitioning."""

  def test_subgraph_import_validity(self):
    """Tests if the partitioned subgraphs can be imported."""
    with tempfile.TemporaryDirectory() as temp_dir:
      create_complex_graph.save_examples_as_saved_models(temp_dir)

      graph_name_to_unpartitioned_paths = {
          'remote_op_a': os.path.join(temp_dir, 'graph_a'),
          'remote_op_b': os.path.join(temp_dir, 'graph_b'),
          'main': os.path.join(temp_dir, 'main_graph')
      }

      out_1, out_2 = graph_partition.get_info_from_paths(
          graph_name_to_unpartitioned_paths)
      graph_name_to_graph_def, graph_name_to_output_names = out_1, out_2
      graph_name_to_partitioned_paths = graph_partition.partition_all_graphs(
          graph_name_to_graph_def, graph_name_to_output_names, temp_dir)

      for partitioned_paths in graph_name_to_partitioned_paths.values():
        for path in partitioned_paths:
          with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            meta_graph_def = tf.compat.v1.saved_model.load(
                sess, [tf.compat.v1.saved_model.tag_constants.SERVING], path)
            self.assertIsNotNone(meta_graph_def)

  def test_with_golden_set(self):
    """Tests if the SavedModels match the golden set.

    Problem: Saved_model.save or saved_model.simple_save doesn't allow us
             to specify the format (binary or text) of the saved_model file
             inside the folder. Not sure if I should use binary files
             for the golden set.
    """


if __name__ == '__main__':
  tf.test.main()
  