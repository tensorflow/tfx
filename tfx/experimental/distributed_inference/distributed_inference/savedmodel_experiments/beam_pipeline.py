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
"""Run batch inference in beam on partitioned graphs.

After getting partitioned SavedModels from graph partition, we'd like to
  arrange, load, and execute the subgraphs in beam.
"""

import os
import copy
import tensorflow as tf
import apache_beam as beam

from tensorflow.core.protobuf import saved_model_pb2


@beam.ptransform_fn
def ExecuteOneGraph(pcoll,  # pylint: disable=invalid-name
                    op_to_directory_paths,
                    op_to_remote_op_name_mapping,
                    graph_name):
  """Compute one graph.

  Main assumptions:
    1. op == graph name.
    2. The parent graph has set up the placeholder inputs for the child graph.
       (So we need to setup PColl for the main graph)

  Args:
    pcoll: input PCollection, each unit contains
           {graph_name: {computed tensor name: value}}
    op_to_directory_paths: {op: a list of partitioned SavedModel paths}

    # This information was stored in py_func's feed_dict,
    # so we don't have it.
    op_to_remote_op_name_mapping:
      {graph_name:
        {remote op name: {placeholder name inside subgraph: input name}}}

    graph_name: the name of the graph

  Returns:
    pcoll with the intermediate/end results of this graph.
  """
  ops = set(op_to_directory_paths.keys())
  directory_paths = _get_directory_paths(graph_name,
                                         op_to_directory_paths,
                                         ops)

  for directory_path in directory_paths:
    saved_model = _get_saved_model(directory_path)
    output_tensor_names = _get_output_tensor_names(saved_model)

    if not _is_remote_op_subgraph(saved_model, ops):
      # Execute a regular subgraph
      step_name = "ExecuteOneSubgraph: graph: {}; output names: {}".format(
          graph_name, ' '.join(output_tensor_names))
      pcoll = pcoll | step_name >> beam.ParDo(_ExecuteOneSubgraph(),
                                              directory_path,
                                              graph_name)

    else:
      # Execute a remote op subgraph (which only contains one remote op)
      current_graph_name = graph_name
      remote_graph_name = output_tensor_names[0]
      graph_descriptor = "current graph: {}; remote graph: {}".format(
          current_graph_name, remote_graph_name)
      
      step_name = "LoadRemoteGraphInputs: %s" % graph_descriptor
      pcoll = pcoll | step_name >> _LoadRemoteGraphInputs(
          current_graph_name, remote_graph_name,
          op_to_remote_op_name_mapping, ops)

      # A good place to add beam.Reshuffle()
      step_name = "ExecuteOneGraph: %s" % graph_descriptor
      pcoll = pcoll | step_name >> ExecuteOneGraph(
          op_to_directory_paths,
          op_to_remote_op_name_mapping,
          remote_graph_name)

      step_name = "ExtractRemoteGraphOutput: %s" % graph_descriptor
      pcoll = pcoll | step_name >> _ExtractRemoteGraphOutput(
          current_graph_name, remote_graph_name,
          op_to_directory_paths, ops)

  return pcoll


def _get_directory_paths(graph_name, op_to_directory_paths, ops):
  op_type = _get_op_type(graph_name, ops)
  return op_to_directory_paths[op_type]


def _get_op_type(graph_name, ops):
  """Get the type of the graph, which can be one of the remote graphs."""
  for op_type in ops:
    if op_type in graph_name:
      return op_type

  return None


def _get_saved_model(directory_path):
  saved_model = saved_model_pb2.SavedModel()
  filepath = os.path.join(directory_path, 'saved_model.pb')
  with open(filepath, 'rb') as f:
    saved_model.ParseFromString(f.read())
  return saved_model


def _get_input_tensor_names(saved_model):
  """Get the input tensor names for a SavedModel."""
  signature_def = saved_model.meta_graphs[0].signature_def['serving_default']
  inputs = signature_def.inputs
  input_tensor_names = [info.name for info in dict(inputs).values()]
  return input_tensor_names


def _get_output_tensor_names(saved_model):
  """Get the output tensor names for a SavedModel."""
  signature_def = saved_model.meta_graphs[0].signature_def['serving_default']
  outputs = signature_def.outputs
  output_tensor_names = [info.name for info in dict(outputs).values()]
  return output_tensor_names


def _is_remote_op_subgraph(saved_model, ops):
  """Determine if a saved_model represents a remote op subgraph."""
  output_tensor_names = _get_output_tensor_names(saved_model)

  # A remote op subgraph only contains one output: a remote op.
  if len(output_tensor_names) == 1:
    for op_type in ops:
      if op_type in output_tensor_names[0]:
        return True

  return False


class _ExecuteOneSubgraph(beam.DoFn):
  """Execute a regular subgraph.

  The Pardo takes in a SavedModel directory path, extract information
    such as inputs and outputs, prepare for the feed_dict,
    load and execute the SavedModel, and store the results.
  """

  def process(self, element, directory_path, graph_name):
    """Executes one subgraph.

    Args:
      element: {graph_name: {computed tensor: value}}
      directory_path: a directory path for SavedModel
      graph_name: the name of the graph for this spec

    Returns:
      A dictionary with the computed outputs added.
    """
    element = copy.deepcopy(element)

    saved_model = _get_saved_model(directory_path)
    input_names = _get_input_tensor_names(saved_model)
    output_names = _get_output_tensor_names(saved_model)

    feed_dict = self._get_feed_dict(element,
                                    graph_name,
                                    input_names)

    results = self._load_and_execute(directory_path,
                                     output_names,
                                     feed_dict)

    element = self._store_results(element,
                                  graph_name,
                                  output_names,
                                  results)
    yield element

  def _get_feed_dict(self, element, graph_name, input_names):
    feed_dict = {input_name: element[graph_name][input_name]
                 for input_name in input_names}
    return feed_dict

  def _load_and_execute(self, directory_path, output_names, feed_dict):
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
      tf.compat.v1.saved_model.load(
          sess,
          [tf.compat.v1.saved_model.tag_constants.SERVING],
          directory_path)
         
      return sess.run(output_names, feed_dict=feed_dict)

  def _store_results(self, element, graph_name, output_names, results):
    for index, output_name in enumerate(output_names):
      element[graph_name][output_name] = results[index]
    return element


@beam.ptransform_fn
def _LoadRemoteGraphInputs(pcoll,  # pylint: disable=invalid-name
                           current_graph_name,
                           remote_graph_name,
                           op_to_remote_op_name_mapping,
                           ops):
  """Load the remote op graph's inputs."""
  current_op_type = _get_op_type(current_graph_name, ops)
  remote_op_type = _get_graph_name_without_attributes(remote_graph_name)
  mapping = op_to_remote_op_name_mapping[current_op_type][remote_op_type]

  for placeholder_name, input_name in mapping.items():
    step_name = "Prepare input for %s in %s" % (placeholder_name, 
                                                remote_graph_name)
    pcoll = pcoll | step_name >> beam.Map(
        _copy_tensor,
        current_graph_name,
        _import_tensor_name(input_name),
        remote_graph_name,
        _import_tensor_name(placeholder_name))

  return pcoll


def _copy_tensor(element,
                 old_graph,
                 old_tensor_name,
                 new_graph,
                 new_tensor_name):
  """Modify element: copy tensor from one graph to another."""
  element = copy.deepcopy(element)
  if new_graph not in element:
    element[new_graph] = {}

  element[new_graph][new_tensor_name] = element[old_graph][old_tensor_name]
  return element


def _get_graph_name_without_attributes(graph_name):
  """Remove the 'import/' prefix and the ':0' postfix."""
  return graph_name[7:-2]


def _import_tensor_name(node_name):
  return 'import/%s:0' % node_name


@beam.ptransform_fn
def _ExtractRemoteGraphOutput(pcoll,  # pylint: disable=invalid-name
                              current_graph_name,
                              remote_graph_name,
                              op_to_directory_paths,
                              ops):
  """Extract the remote op graph's output."""
  remote_op_type = _get_op_type(remote_graph_name, ops)
  remote_graph_paths = op_to_directory_paths[remote_op_type]
  remote_graph_last_saved_model = _get_saved_model(remote_graph_paths[-1])
  output_name = _get_output_tensor_names(remote_graph_last_saved_model)[0]
  
  step_name_extract = "Extract output from %s to %s" % (remote_graph_name, 
                                                        current_graph_name)
  step_name_remove = "Remove finished graph info for %s" % (remote_graph_name)

  return (pcoll
          # Extract the output from the remote graph
          | step_name_extract >> beam.Map(
              _copy_tensor,
              remote_graph_name,
              output_name,
              current_graph_name,
              remote_graph_name)
          # Remove the intermediate results of the remote graph
          | step_name_remove >> beam.Map(
              _remove_finished_graph_info,
              remote_graph_name))


def _remove_finished_graph_info(element, finished_graph_name):
  """Remove the intermediate results of a finished remote op graph."""
  element = copy.deepcopy(element)
  del element[finished_graph_name]
  return element
