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
import unittest

import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf

from tfx.experimental.distributed_inference.graphdef_experiments.subgraph_partitioning import beam_pipeline
from tfx.experimental.distributed_inference.graphdef_experiments.subgraph_partitioning import create_complex_graph
from tfx.experimental.distributed_inference.graphdef_experiments.subgraph_partitioning import graph_partition


@unittest.skipIf(tf.__version__ < '2', 'Incompatible with TF1')
class BeamPipelineTest(tf.test.TestCase):
  """A test for the beam pipeline library."""

  def test_compare_outputs(self):
    """Compares the results from the original model and the beam pipeline."""
    with tempfile.TemporaryDirectory() as temp_dir:
      create_complex_graph.save_examples_as_graphdefs(temp_dir)

      # "main" is both a graph name and a remote op name.
      root_graph = 'main'
      graph_to_remote_op_input_name_mapping = {
          'main': {
              'remote_op_a': {
                  'ids_a': 'ids1'
              },
              'remote_op_b': {
                  'ids_b1': 'ids1',
                  'ids_b2': 'ids2'
              },
              'remote_op_a_1': {
                  'ids_a': 'FloorMod_1'
              },
              'remote_op_b_1': {
                  'ids_b1': 'FloorMod_1',
                  'ids_b2': 'FloorMod'
              }
          },
          'graph_b': {
              'remote_op_a': {
                  'ids_a': 'FloorMod'
              },
              'remote_op_a_1': {
                  'ids_a': 'ids_b2'
              }
          }
      }

      # Create input PColl with this.
      root_graph_input_data = [{
          'main': {
              'import/ids1:0': 3,
              'import/ids2:0': 3
          }
      }, {
          'main': {
              'import/ids1:0': 10,
              'import/ids2:0': 10
          }
      }]

      graph_name_to_filepath = {
          'main': os.path.join(temp_dir, 'main_graph.pb'),
          'graph_b': os.path.join(temp_dir, 'graph_b.pb'),
          'graph_a': os.path.join(temp_dir, 'graph_a.pb')
      }
      graph_name_to_output_names = {
          'main': ['AddN_1'],
          'graph_b': ['Add_1'],
          'graph_a': ['embedding_lookup/Identity']
      }
      remote_op_name_to_graph_name = {
          'main': 'main',
          'remote_op_a': 'graph_a',
          'remote_op_a_1': 'graph_a',
          'remote_op_b': 'graph_b',
          'remote_op_b_1': 'graph_b'
      }

      original_model_outputs = _run_original_model(root_graph,
                                                   root_graph_input_data,
                                                   graph_name_to_filepath,
                                                   graph_name_to_output_names)

      graph_name_to_graph_def = graph_partition.get_graph_name_to_graph_def(
          graph_name_to_filepath)
      graph_name_to_specs = graph_partition.partition_all_graphs(
          graph_name_to_graph_def, graph_name_to_output_names)

      with beam.Pipeline() as p:

        beam_inputs = p | 'LoadInputs' >> beam.Create(root_graph_input_data)
        beam_outputs = (
            beam_inputs
            | 'RunModel' >> beam_pipeline.ExecuteGraph(
                root_graph, remote_op_name_to_graph_name, graph_name_to_specs,
                graph_to_remote_op_input_name_mapping)
            | 'ExtractOutputs' >> beam.Map(_extract_outputs, root_graph,
                                           graph_name_to_output_names))

        util.assert_that(beam_outputs, almost_equal_to(original_model_outputs))


def _run_original_model(root_graph, root_graph_input_data,
                        graph_name_to_filepath, graph_name_to_output_names):
  """Runs the original model."""
  graph_name_to_graph_def = graph_partition.get_graph_name_to_graph_def(
      graph_name_to_filepath)
  graph_def = graph_name_to_graph_def[root_graph]
  output_tensor_names = [
      _import_tensor_name(name)
      for name in graph_name_to_output_names[root_graph]
  ]

  outputs = []
  with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    tf.import_graph_def(graph_def)
    for graph_name_to_feed_dict in root_graph_input_data:
      outputs.append(
          sess.run(output_tensor_names, graph_name_to_feed_dict[root_graph]))

  return outputs


def _import_tensor_name(node_name):
  return 'import/%s:0' % node_name


def _extract_outputs(element, root_graph, graph_name_to_output_names):
  outputs = [
      element[root_graph][_import_tensor_name(name)]
      for name in graph_name_to_output_names[root_graph]
  ]
  return outputs


def almost_equal_to(expected):

  def _almost_equal(actual):
    sorted_expected = sorted(expected)
    sorted_actual = sorted(actual)
    if not np.allclose(sorted_expected, sorted_actual):
      raise util.BeamAssertException(
          'Failed assert: {} is not almost equal to {}'.format(
              sorted_expected, sorted_actual))

  return _almost_equal


if __name__ == '__main__':
  tf.test.main()
