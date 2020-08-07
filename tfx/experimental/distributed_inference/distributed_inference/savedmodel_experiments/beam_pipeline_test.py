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

from create_complex_graph import save_examples_as_savedmodels
import graph_partition
import beam_pipeline


op_to_remote_op_name_mapping = {
    'main': {'remote_op_a': {'ids_a': 'ids1'},
             'remote_op_b': {'ids_b1': 'ids1', 'ids_b2': 'ids2'},
             'remote_op_a_1': {'ids_a': 'FloorMod_1'},
             'remote_op_b_1': {'ids_b1': 'FloorMod_1', 'ids_b2': 'FloorMod'}},
    'remote_op_b': {'remote_op_a': {'ids_a': 'FloorMod'},
                    'remote_op_a_1': {'ids_a': 'ids_b2'}}}

feed_dicts_main_graph = [
    {'main': {'import/ids1:0': 3, 'import/ids2:0': 3}},
    {'main': {'import/ids1:0': 10, 'import/ids2:0': 10}}]
op_to_feed_dicts = {
    'main': feed_dicts_main_graph}


class BeamPipelineTest(tf.test.TestCase):
  """Test that compares the results from beam and the original model."""
  def test_results(self):
    """Compute a model in two ways and compare the results."""
    graph_name = 'main'
    
    with tempfile.TemporaryDirectory() as temp_dir:
      # Save examples into a temporary directory
      save_examples_as_savedmodels(temp_dir)

      op_to_filepath = {
          'remote_op_a': os.path.join(temp_dir, 'graph_a'),
          'remote_op_b': os.path.join(temp_dir, 'graph_b'),
          'main': os.path.join(temp_dir, 'main_graph')}
      op_to_graph_def, op_to_output_names = graph_partition.load_saved_models(
          op_to_filepath)
      op_to_directory_paths = graph_partition.partition_all_graphs(
          op_to_graph_def, op_to_output_names, temp_dir)

      # Run the partitioned model in beam
      with test_pipeline.TestPipeline() as p:
        input_pcoll = p | 'LoadData' >> beam.Create(
            op_to_feed_dicts[graph_name])

        output = (input_pcoll
                  | 'RunModel' >> beam_pipeline.ExecuteOneGraph(
                      op_to_directory_paths,
                      op_to_remote_op_name_mapping,
                      graph_name)
                  | 'ExtractOutput' >> beam.Map(_extract_outputs,
                                                op_to_output_names,
                                                graph_name))

      # Run the original model
      result_original_model = _run_original_model(graph_name,
                                                  op_to_graph_def,
                                                  op_to_output_names,
                                                  op_to_feed_dicts)

      # Problem: Doesn't work with more complex things like tensors.
      util.assert_that(output, util.equal_to(result_original_model))
      p.run()


def _run_original_model(graph_name,
                        op_to_graph_def,
                        op_to_outputs,
                        op_to_feed_dicts):
  """Run the original TF model."""
  graph_def = op_to_graph_def[graph_name]
  graph = _get_graph(graph_def)

  output_names = [_import_tensor_name(output_name)
                  for output_name in op_to_outputs[graph_name]]
  feed_dicts = op_to_feed_dicts[graph_name]

  results = []
  with tf.compat.v1.Session(graph=graph) as sess:
    for feed_dict in feed_dicts:
      results.append(sess.run(output_names, feed_dict[graph_name]))

  return results


def _get_graph(graph_def):
  temp = tf.Graph()
  with temp.as_default():
    tf.import_graph_def(graph_def)
    return tf.compat.v1.get_default_graph()


def _import_tensor_name(node_name):
  return 'import/%s:0' % node_name


def _extract_outputs(element, op_to_outputs, graph_name):
  """Extract the outputs within output_names"""
  outputs = [element[graph_name][_import_tensor_name(output)]
             for output in op_to_outputs[graph_name]]
  return outputs


if __name__ == '__main__':
  tf.test.main()
