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
"""Tests for Beam Pipeline."""

import os
import tempfile
import tensorflow as tf
import apache_beam as beam
from apache_beam.testing import test_pipeline
from apache_beam.testing import util

import create_complex_graph
import graph_partition
import beam_pipeline


class BeamPipelineTest(tf.test.TestCase):
  """A test for the beam pipeline library."""

  def test_validate_outputs(self):
    """Compares the results from the original model and the beam pipeline."""
    with tempfile.TemporaryDirectory() as temp_dir:
      create_complex_graph.save_examples_as_graphdefs(temp_dir)

      # Root graph is the graph to compute. Since it's unique, it's both the
      # graph name extended and the graph name.
      root_graph = 'main'
      parent_graph_to_remote_graph_input_name_mapping = {
          'main': {'remote_op_a': {'ids_a': 'ids1'},
                   'remote_op_b': {'ids_b1': 'ids1',
                                   'ids_b2': 'ids2'},
                   'remote_op_a_1': {'ids_a': 'FloorMod_1'},
                   'remote_op_b_1': {'ids_b1': 'FloorMod_1',
                                     'ids_b2': 'FloorMod'}
                  },
          'remote_op_b': {'remote_op_a': {'ids_a': 'FloorMod'},
                          'remote_op_a_1': {'ids_a': 'ids_b2'}
                         }
      }

      # Create input PColl with this.
      root_graph_inputs = [
          {'main': {'import/ids1:0': 3, 'import/ids2:0': 3}},
          {'main': {'import/ids1:0': 10, 'import/ids2:0': 10}}]

      graph_name_to_filepath = {
          'main': os.path.join(temp_dir, 'main_graph.pb'),
          'remote_op_a': os.path.join(temp_dir, 'graph_a.pb'),
          'remote_op_b': os.path.join(temp_dir, 'graph_b.pb')}
      graph_name_to_outputs = {
          'main': ['AddN_1'],
          'remote_op_b': ['Add_1'],
          'remote_op_a': ['embedding_lookup/Identity']}

      original_model_outputs = _run_original_model(root_graph,
                                                   root_graph_inputs,
                                                   graph_name_to_filepath,
                                                   graph_name_to_outputs)

      graph_name_to_graph_def = graph_partition.get_graph_name_to_graph_def(
          graph_name_to_filepath)
      graph_name_to_specs = graph_partition.partition_all_graphs(
          graph_name_to_graph_def, graph_name_to_outputs)

      with test_pipeline.TestPipeline() as p:

        inputs = p | 'LoadInputs' >> beam.Create(root_graph_inputs)
        outputs = (inputs
                   | 'RunModel' >> beam_pipeline.ExecuteOneGraph(
                       graph_name_to_specs,
                       parent_graph_to_remote_graph_input_name_mapping,
                       root_graph)
                   | 'ExtractOutputs' >> beam.Map(_extract_outputs,
                                                  graph_name_to_outputs,
                                                  root_graph))

        # Problem: The output for the example graph is a scalar, equal_to
        # doesn't work with more complex things like tensors.
        util.assert_that(outputs, util.equal_to(original_model_outputs))


def _run_original_model(root_graph,
                        root_graph_inputs,
                        graph_name_to_filepath,
                        graph_name_to_outputs):
  """Runs the original model."""
  graph_name_to_graph_def = graph_partition.get_graph_name_to_graph_def(
      graph_name_to_filepath)
  graph_def = graph_name_to_graph_def[root_graph]
  output_tensor_names = [_import_tensor_name(node_name)
                         for node_name in graph_name_to_outputs[root_graph]]

  outputs = []
  with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    tf.import_graph_def(graph_def)
    for graph_name_to_feed_dict in root_graph_inputs:
      outputs.append(
          sess.run(output_tensor_names, graph_name_to_feed_dict[root_graph]))

  return outputs


def _import_tensor_name(node_name):
  return 'import/%s:0' % node_name


def _extract_outputs(element, graph_name_to_outputs, root_graph):
  outputs = [element[root_graph][_import_tensor_name(node_name)]
             for node_name in graph_name_to_outputs[root_graph]]
  return outputs


if __name__ == '__main__':
  tf.test.main()
