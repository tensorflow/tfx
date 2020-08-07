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
"""Perform graph partitioning on TensorFlow SavedModels with remote ops.

The current implementation targets two goals:
  1. Maximal subgraphs
  2. Avoid repeated work when running subgraphs in beam

Definition:
  1. "op" refers to a graph name. It can be the name of one of the
     remote graphs or the name of the main graph.

Key Assumptions/Limitations for this implementation:
  1. Inputs should be SavedModels without variables
  2. Only one output per node/op
  3. Each SavedModel only contains one metagraph
  4. Not supporting tf.functions
"""

import os
import tensorflow as tf

from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.core.framework import graph_pb2

from execution_spec import ExecutionSpec


def load_saved_models(op_to_directory_paths):
  """Extract information from SavedModels.

  Args:
    op_to_directory_paths: {graph name: a SavedModel directory path}

  Returns:
    op_to_graph_def: {graph name: a GraphDef proto}
    op_to_output_names: {graph name: a list of output names}
  """
  op_to_graph_def = {}
  op_to_output_names = {}

  for op, directory_path in op_to_directory_paths.items():
    saved_model = _get_saved_model(
        os.path.join(directory_path, 'saved_model.pb'))

    op_to_graph_def[op] = _get_graph_def(saved_model)
    op_to_output_names[op] = _get_output_names(saved_model)

  return op_to_graph_def, op_to_output_names


def _get_saved_model(file_path):
  saved_model = saved_model_pb2.SavedModel()
  with open(file_path, 'rb') as f:
    saved_model.ParseFromString(f.read())
  return saved_model


def _get_graph_def(saved_model):
  return saved_model.meta_graphs[0].graph_def


def _get_output_names(saved_model):
  meta_graph = saved_model.meta_graphs[0]
  signature_def = meta_graph.signature_def['serving_default']
  outputs = signature_def.outputs

  # [:-2] to remove the ':0' tensor postfix
  output_names = [output.name[:-2] for output in dict(outputs).values()]
  return output_names


def partition_all_graphs(op_to_graph_def, op_to_output_names, export_dir):
  """Partition all the graphs.

  Here, we partition each graph given its graph_def and output names.
    Then, we save the partitioned graphs as SavedModels to the export_dir.

  Args:
    op_to_graph_def: {graph name: a GraphDef proto}
    op_to_output_names: {graph name: a list of output names}
    export_dir: the directory to save

  Returns:
    {graph name: a list of SavedModel directory paths}
  """
  op_to_directory_paths = {}

  for op in op_to_graph_def:
    saved_model_directory_paths = _partition_one_graph(op_to_graph_def[op],
                                                       op_to_output_names[op],
                                                       export_dir,
                                                       op)
    op_to_directory_paths[op] = saved_model_directory_paths

  return op_to_directory_paths


def _partition_one_graph(graph_def, output_names, export_dir, graph_name):
  """Partition one graph.

  Args:
    graph_def: a GraphDef proto
    output_names: a list of output names
    export_dir: the directory to save
    graph_name: the name of the current graph

  Returns:
    A list of SavedModel directory paths.
  """
  graph = _get_graph(graph_def)
  node_name_to_node_def = _get_node_name_to_node_def(graph_def)
  node_name_to_input_names = _get_node_name_to_input_names(graph_def)

  remote_op_relations = _get_remote_op_relations(graph_def,
                                                 node_name_to_node_def,
                                                 node_name_to_input_names)

  execution_specs = _get_execution_specs(graph_def,
                                         output_names,
                                         graph,
                                         node_name_to_node_def,
                                         node_name_to_input_names,
                                         remote_op_relations)

  saved_model_directory_paths = _export_as_saved_models(export_dir,
                                                        graph_name,
                                                        execution_specs)

  return saved_model_directory_paths


def _get_graph(graph_def):
  with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    tf.import_graph_def(graph_def)
    return sess.graph


def _get_node_name_to_node_def(graph_def):
  return {node.name: node for node in graph_def.node}


def _get_node_name_to_input_names(graph_def):
  return {node.name: list(node.input) for node in graph_def.node}


def _get_remote_op_relations(graph_def,
                             node_name_to_node_def,
                             node_name_to_input_names):
  """Get the execution dependencies between remote ops.

  The remote op children must be executed before executing a remote op.

  Args:
    graph_def: a GraphDef proto
    node_name_to_node_def: {node name: a NodeDef proto}
    node_name_to_input_names: {node name: a list of node input names}

  Returns:
    {remote op: a list of remote op children}
  """
  remote_op_relations = {}

  for node in graph_def.node:
    if _is_remote_op(node):
      remote_op_relations[node.name] = _get_remote_op_children(
          node.name,
          node_name_to_node_def,
          node_name_to_input_names)

  return remote_op_relations


def _get_remote_op_children(remote_op_name,
                            node_name_to_node_def,
                            node_name_to_input_names):
  """Find the remote op children for a remote op.

  Args:
    remote_op_name: the name of the parent remote op
    node_name_to_node_def: {node name: a NodeDef proto}
    node_name_to_input_names: {node name: a list of node input names}

  Returns:
    A list of remote op children.
  """
  queue = [remote_op_name]
  visited = set([remote_op_name])
  remote_op_children = []

  while queue:
    current_node_name = queue[0]
    del queue[0]

    for input_node_name in node_name_to_input_names[current_node_name]:
      if input_node_name not in visited:
        visited.add(input_node_name)
        input_node = node_name_to_node_def[input_node_name]

        if _is_remote_op(input_node):
          remote_op_children.append(input_node_name)
        else:
          queue.append(input_node_name)

  return remote_op_children


def _is_placeholder_op(node):
  return node.op == "Placeholder"


def _is_remote_op(node):
  return node.op == "PyFunc"


def _get_execution_specs(graph_def,
                         graph_outputs,
                         graph,
                         node_name_to_node_def,
                         node_name_to_input_names,
                         remote_op_relations):
  """Generate the execution_specs for a graph.

  Note that the execution_specs captures the order of execution by the order
    of the list. Execution_spec with smaller index executes first.

  Args:
    graph_def: a GraphDef proto
    graph_outputs: {graph name: a list of output node names}
    graph: a tf.Graph instance
    node_name_to_node_def: {node name: a NodeDef proto}
    node_name_to_input_names: {node name: a list of node input names}
    remote_op_relations: {remote op: a list of remote op children}

  Returns:
    A list of execution spec.
  """
  execution_specs = []
  previous_layers_visited = set([])

  order = Relations(remote_op_relations)
  while not order.check_if_finished():
    remote_ops_one_layer = order.get_next_layer()

    # Handle one subgraph layer
    layer_output_node_names = _get_subgraph_layer_output_node_names(
        remote_ops_one_layer,
        node_name_to_node_def,
        node_name_to_input_names)

    if layer_output_node_names:
      subgraph_spec = _partition_one_subgraph_layer(previous_layers_visited,
                                                    graph_def,
                                                    graph,
                                                    layer_output_node_names,
                                                    node_name_to_node_def,
                                                    node_name_to_input_names)

      subgraph_spec = _handle_nodes_from_other_layers(subgraph_spec,
                                                      execution_specs,
                                                      graph)
      execution_specs.append(subgraph_spec)
      previous_layers_visited = previous_layers_visited.union(
          subgraph_spec.body_node_names)

    # Handle one remote op layer
    remote_op_specs = _partition_one_remote_op_layer(remote_ops_one_layer,
                                                     graph_def,
                                                     graph,
                                                     node_name_to_node_def,
                                                     node_name_to_input_names)
    execution_specs.extend(remote_op_specs)

  # Handle the last subgraph layer
  output_node_names = set(graph_outputs)
  subgraph_spec = _partition_one_subgraph_layer(previous_layers_visited,
                                                graph_def,
                                                graph,
                                                output_node_names,
                                                node_name_to_node_def,
                                                node_name_to_input_names)

  subgraph_spec = _handle_nodes_from_other_layers(subgraph_spec,
                                                  execution_specs,
                                                  graph)
  execution_specs.append(subgraph_spec)
  previous_layers_visited = previous_layers_visited.union(
      subgraph_spec.body_node_names)

  return execution_specs


def _get_subgraph_layer_output_node_names(remote_ops_one_layer,
                                          node_name_to_node_def,
                                          node_name_to_input_names):
  """Get the output names of a subgraph layer.

  Note that they are equivalant to the input names of the succeeding
    remote op layer.
  """
  output_node_names = set([])

  for remote_op in remote_ops_one_layer:
    for input_node_name in node_name_to_input_names[remote_op]:
      input_node = node_name_to_node_def[input_node_name]

      # Assumption: graph inputs (placeholders) are loaded before
      #             executing a graph.
      if not _is_placeholder_op(input_node):
        output_node_names.add(input_node_name)

  return output_node_names


def _partition_one_subgraph_layer(previous_layers_visited,
                                  graph_def,
                                  graph,
                                  output_names,
                                  node_name_to_node_def,
                                  node_name_to_input_names):
  """Perform a modified BFS for graph partitioning.

  Expand from the outputs, until one of the stopping condition: remote op,
    placeholder, visited before in this layer, or visited before in the
    previous layers.

  Args:
    previous_layers_visited: a set of nodes
    graph_def: a GraphDef proto
    graph: a tf.Graph
    output_names: a set of output names
    node_name_to_node_def: {node name: a NodeDef proto}
    node_name_to_input_names: {node name: a list of node input names}

  Returns:
    An execution specs.
  """
  subgraph = graph_pb2.GraphDef()
  subgraph.versions.CopyFrom(graph_def.versions)
  subgraph.library.CopyFrom(graph_def.library)

  queue = list(output_names)
  current_layer_visited = set([])
  nodes_from_other_layers = set([])

  while queue:
    current_node_name = queue[0]
    current_node = node_name_to_node_def[current_node_name]
    del queue[0]

    if _is_remote_op(current_node) or _is_placeholder_op(current_node):
      # Remote op or placeholder input will always be prepared.
      if current_node_name not in current_layer_visited:
        placeholder_node = _create_placeholder_node_from_existing_node_name(
            current_node_name, graph)
        subgraph.node.append(placeholder_node)

        current_layer_visited.add(current_node_name)
    else:
      # Regular op may be an intermediate node from other graphs and
      # not prepared, so we need to find them and do something later.
      if current_node_name in previous_layers_visited:
        nodes_from_other_layers.add(current_node_name)

      elif current_node_name not in current_layer_visited:
        subgraph.node.append(current_node)

        current_layer_visited.add(current_node_name)
        queue.extend(node_name_to_input_names[current_node_name])

  return ExecutionSpec(
      subgraph=subgraph,
      input_names=_get_input_names_from_subgraph(subgraph),
      output_names=set(output_names),
      is_remote_op=False,
      body_node_names=_get_body_node_names_from_subgraph(subgraph),
      nodes_from_other_layers=nodes_from_other_layers)


def _handle_nodes_from_other_layers(current_spec,
                                    execution_specs,
                                    graph):
  """Handle nodes that are from other layers."""
  for node_name in current_spec.nodes_from_other_layers:
    for previous_spec in execution_specs:

      if node_name in previous_spec.body_node_names:
        previous_spec.output_names.add(node_name)
        current_spec.input_names.add(node_name)

        placeholder = _create_placeholder_node_from_existing_node_name(
            node_name, graph)
        current_spec.subgraph.node.append(placeholder)

  return current_spec


def _partition_one_remote_op_layer(remote_op_names,
                                   graph_def,
                                   graph,
                                   node_name_to_node_def,
                                   node_name_to_input_names):
  """Construct execution spec for remote ops.

  Args:
    remote_op_names: a set of node names
    graph_def: a GraphDef proto
    graph: a tf.Graph
    node_name_to_node_def: {node name: a NodeDef proto}
    node_name_to_input_names: {node name: a list of node input names}

  Returns:
    A list of execution specs.
  """
  list_of_specs = []
  for remote_op_name in remote_op_names:
    subgraph = graph_pb2.GraphDef()
    subgraph.versions.CopyFrom(graph_def.versions)
    subgraph.library.CopyFrom(graph_def.library)

    input_names = node_name_to_input_names[remote_op_name]
    for input_name in input_names:
      placeholder_node = _create_placeholder_node_from_existing_node_name(
          input_name, graph)
      subgraph.node.append(placeholder_node)

    subgraph.node.append(node_name_to_node_def[remote_op_name])

    spec = ExecutionSpec(
        subgraph=subgraph,
        input_names=set(input_names),
        output_names=set([remote_op_name]),
        is_remote_op=True,
        body_node_names=set([remote_op_name]),
        nodes_from_other_layers=set([]))
    list_of_specs.append(spec)

  return list_of_specs


def _create_placeholder_node(dtype, shape, name):
  temp = tf.Graph()
  with temp.as_default():
    tf.compat.v1.placeholder(dtype=dtype, shape=shape, name=name)
    return temp.as_graph_def().node[0]  # The first and the only node


def _create_placeholder_node_from_existing_node_name(node_name, graph):
  operation = graph.get_operation_by_name('import/%s' % (node_name))
  dtype = operation.outputs[0].dtype
  return _create_placeholder_node(dtype=dtype,
                                  shape=None,
                                  name=node_name)


def _get_input_names_from_subgraph(subgraph):
  input_names = {node.name for node in subgraph.node
                 if _is_placeholder_op(node)}
  return input_names


def _get_body_node_names_from_subgraph(subgraph):
  body_node_names = {node.name for node in subgraph.node
                     if not _is_placeholder_op(node)}
  return body_node_names


def _export_as_saved_models(export_dir,
                            graph_name,
                            execution_specs):
  """Save the partitioned graphs as SavedModels.

  Args:
    export_dir: the directory to save
    graph_name: the name of the original graph
    execution_specs: a list of execution specs for partitioned graphs

  Returns:
    A list of SavedModel directory paths.
  """
  directory_paths = []

  for counter, execution_spec in enumerate(execution_specs):
    directory_path = _get_saved_model_directory_path(
        export_dir, graph_name, counter, execution_spec.output_names)
    directory_paths.append(directory_path)

    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
      tf.import_graph_def(execution_spec.subgraph)
      inputs = _get_name_to_tensor(sess.graph, execution_spec.input_names)
      outputs = _get_name_to_tensor(sess.graph, execution_spec.output_names)

      tf.compat.v1.saved_model.simple_save(sess,
                                           directory_path,
                                           inputs,
                                           outputs)
  return directory_paths


def _get_saved_model_directory_path(export_dir,
                                    graph_name,
                                    counter,
                                    output_names):
  """Get a unique directory path for partitioned SavedModel."""
  directory_path = os.path.join(
      export_dir,
      'partitioned_saved_models',
      'graph_name_%s_subgraph_%s_output_names_%s' %
      (graph_name, str(counter), '_'.join(output_names)))
  return directory_path


def _get_name_to_tensor(graph, node_names):
  """Get {node name: tf.Tensor} for a set of node names."""
  name_to_tensor = {name: graph.get_tensor_by_name('import/%s:0' % name)
                    for name in node_names}
  return name_to_tensor


class Relations:
  """A class that outputs remote op layers (custom topological sort).

  What is a layer? A layer is a set of remote ops that don't have
    dependencies on each other and are ready to execute."""
  def __init__(self, relations):
    self.relations = relations
    self.processed = set([])
    self.to_be_processed = set(relations.keys())

  def check_if_finished(self):
    return not self.to_be_processed

  def get_next_layer(self):
    """Get the next set of remote ops."""
    layer_nodes = set([])

    for node in self.to_be_processed:
      node_inputs = set(self.relations[node])
      if node_inputs.issubset(self.processed):
        layer_nodes.add(node)

    for node in layer_nodes:
      self.to_be_processed.remove(node)
      self.processed.add(node)

    return layer_nodes
