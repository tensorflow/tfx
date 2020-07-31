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
"""Perform graph partitioning on TensorFlow GraphDefs with remote ops.

The current partitioning strategy aims at maximum subgraphs.

There are two libraries representing two stages: graph_partition and
beam_pipeline. In this library, we take in some GraphDef inputs, partition
the graphs, and store the partitioned subgraphs into lists of ExecutionSpecs.
The order of the list represents the order of execution. Take a look at
beam_pipeline if you want to run the partitioned subgraphs.

The current implementation has some key limitations:
  1. This library only accepts GraphDefs (or their filepaths) as the inputs.
  2. All the node/op should only have one output.
  3. This library isn't supporting tf.variable.
  4. This library isn't supporting tf.function.

  Typical usage example:
  '''
  graph_name_to_filepath = {graph_1_name: filepath_1,
                            graph_2_name: filepath_2}
  graph_name_to_output_names = {graph_1_name: [output node names of graph_1],
                                graph_2_name: [output node names of graph_2]}

  graph_name_to_graph_def = get_graph_name_to_graph_def(
      graph_name_to_filepath)
  graph_name_to_execution_specs = partition_all_graphs(
      graph_name_to_graph_def,
      graph_name_to_output_names)

  # Followed by the beam pipeline.
  '''
"""

from typing import Any, List, Mapping, Set, Text

import tensorflow as tf
from tensorflow.core.framework import graph_pb2

from execution_spec import ExecutionSpec


def get_graph_name_to_graph_def(
    graph_name_to_filepath: Mapping[Text, Text]
) -> Mapping[Text, Any]:
  """Get the 'GraphDef' proto inputs.

  Args:
    graph_name_to_filepath: A mapping from graph names to filepaths. Each
                            filepath points to a 'GraphDef' proto in binary.

  Returns:
    A mapping from graph names to 'GraphDef' protos.
  """
  graph_name_to_graph_def = {
      graph_name: _get_graph_def(filepath)
      for graph_name, filepath in graph_name_to_filepath.items()}
  return graph_name_to_graph_def


def _get_graph_def(filepath: Text) -> Any:
  graph_def = graph_pb2.GraphDef()
  with tf.io.gfile.GFile(filepath, 'rb') as f:
    graph_def.ParseFromString(f.read())
  return graph_def


def partition_all_graphs(
    graph_name_to_graph_def: Mapping[Text, Any],
    graph_name_to_output_names: Mapping[Text, List[Text]]
) -> Mapping[Text, List[ExecutionSpec]]:
  """Partition all the graphs.

  For each graph, the partitioning algorithm takes in the graph's 'GraphDef'
  proto and output names, partitions the graph, and returns a list of
  ExecutionSpecs. Later, the beam_pipeline library can take in the
  ExecutionSpecs and execute the partitioned subgraphs.

  Args:
    graph_name_to_graph_def: A mapping from graph names to 'GraphDef' protos.
    graph_name_to_output_names: A mapping from graph names to lists of their
                                output node names.

  Returns:
    A mapping from graph names to a list of ExecutionSpecs, where the order
    of the list represents the order of execution.
  """
  graph_name_to_specs = {}
  for graph_name in graph_name_to_graph_def:
    specs = _partition_one_graph(graph_name_to_graph_def[graph_name],
                                 graph_name_to_output_names[graph_name])
    graph_name_to_specs[graph_name] = specs
  return graph_name_to_specs


def _partition_one_graph(
    graph_def: Any,
    output_names: List[Text]) -> List[ExecutionSpec]:
  """Partition one graph.

  Args:
    graph_def: A 'GraphDef' proto for that graph.
    output_names: A list of graph's output node names.

  Returns:
    A list of ExecutionSpecs.
  """
  graph = _get_graph(graph_def)
  node_name_to_node_def = _get_node_name_to_node_def(graph_def)

  remote_op_to_parents = _get_remote_op_to_parents(node_name_to_node_def)

  specs = _get_execution_specs(graph_def,
                               output_names,
                               graph,
                               node_name_to_node_def,
                               remote_op_to_parents)
  return specs


def _get_graph(graph_def: Any) -> Any:
  with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    tf.import_graph_def(graph_def)
    return sess.graph


def _get_node_name_to_node_def(graph_def: Any) -> Mapping[Text, Any]:
  return {node.name: node for node in graph_def.node}


def _get_remote_op_to_parents(
    node_name_to_node_def: Mapping[Text, Any]) -> Mapping[Text, List[Text]]:
  """Get the execution dependencies between remote ops.

  The remote op parents must be executed before executing a remote op.

  Args:
    node_name_to_node_def: A mapping from node names to 'NodeDef' protos.

  Returns:
    A mapping from a remote op name to a list of immediate remote op parent
    names.
  """
  remote_op_to_parents = {}

  for node in node_name_to_node_def.values():
    if _is_remote_op(node):
      remote_op_to_parents[node.name] = _get_remote_op_parents(
          node.name, node_name_to_node_def)

  return remote_op_to_parents


def _get_remote_op_parents(
    remote_op_name: Text,
    node_name_to_node_def: Mapping[Text, Any]) -> List[Text]:
  """Find the immediate remote op parents for a remote op.

  Args:
    remote_op_name: The name of the child remote op.
    node_name_to_node_def: A mapping from node names to 'NodeDef' protos.

  Returns:
    A list of immediate remote op parent names.
  """
  queue = [remote_op_name]
  visited = set([remote_op_name])
  remote_op_parents = []

  while queue:
    current_node_name = queue[0]
    del queue[0]

    for input_node_name in node_name_to_node_def[current_node_name].input:
      if input_node_name not in visited:
        visited.add(input_node_name)
        input_node = node_name_to_node_def[input_node_name]

        # Stop traversing when reaching a remote op.
        if _is_remote_op(input_node):
          remote_op_parents.append(input_node_name)
        else:
          queue.append(input_node_name)

  return remote_op_parents


def _is_placeholder_op(node: Any) -> bool:
  return node.op == "Placeholder"


def _is_remote_op(node: Any) -> bool:
  return node.op == "PyFunc"


def _get_execution_specs(
    graph_def: Any,
    graph_output_names: List[Text],
    graph: Any,
    node_name_to_node_def: Mapping[Text, Any],
    remote_op_to_parents: Mapping[Text, List[Text]]
) -> List[ExecutionSpec]:
  """Generate the ExecutionSpecs for a graph.

  A 'layer' contains one or more nodes inside a graph. There are two types of
  layers: subgraph layer and remote op layer. A subgraph layer doesn't
  contain remote ops, whereas a remote op layer only contains remote ops.
  Remote ops inside a remote op layer don't depend on each other's output,
  so the order of execution between those remote ops doesn't matter.

  We first identify the remote op layers of a graph. Then, based on the
  remote op layers, we can derive the subgraph layers. For example, after
  identifying the first remote op layer, we can equate the inputs of the
  remote op layer to the outputs of the previous subgraph layer. We can
  then traverse and construct the previous subgraph layer.

  Each subgraph layer can be captured into one ExecutionSpec, but each
  remote op layer need to be stored into one or more ExecutionSpecs based on
  the number of remote ops. This happens because we assume that the execution
  of ExecutionSpecs is strictly sequential.

  Args:
    graph_def: A 'GraphDef' proto.
    graph_output_names: A list of graph output node names.
    graph: A tf.Graph representing the same graph as graph_def.
    node_name_to_node_def: A mapping from node names to 'NodeDef' protos.
    remote_op_to_parents: A mapping from remote op name to a list of
                          immediate remote op parent names.

  Returns:
    A list of ExecutionSpecs, where the order of the list represents the
    order of execution.
  """
  execution_specs = []  # type: List[ExecutionSpec]
  previous_layers_visited = set([])  # type: Set[Text]

  for remote_op_layer in _RemoteOpLayers(remote_op_to_parents):
    # Get one subgraph layer
    output_node_names = _get_subgraph_layer_output_node_names(
        remote_op_layer, node_name_to_node_def)

    if output_node_names:
      spec = _partition_one_subgraph_layer(graph_def,
                                           graph,
                                           node_name_to_node_def,
                                           previous_layers_visited,
                                           output_node_names)

      spec = _handle_nodes_from_other_layers(spec,
                                             execution_specs,
                                             graph,
                                             node_name_to_node_def)
      execution_specs.append(spec)
      previous_layers_visited |= spec.body_node_names

    # Get one remote op layer
    specs = _partition_one_remote_op_layer(remote_op_layer,
                                           node_name_to_node_def)
    execution_specs.extend(specs)

  # Get the last subgraph layer
  output_node_names = set(graph_output_names)
  spec = _partition_one_subgraph_layer(graph_def,
                                       graph,
                                       node_name_to_node_def,
                                       previous_layers_visited,
                                       output_node_names)

  spec = _handle_nodes_from_other_layers(spec,
                                         execution_specs,
                                         graph,
                                         node_name_to_node_def)
  execution_specs.append(spec)
  previous_layers_visited |= spec.body_node_names

  return execution_specs


def _get_subgraph_layer_output_node_names(
    remote_op_layer: Set[Text],
    node_name_to_node_def: Mapping[Text, Any]) -> Set[Text]:
  """Get the output node names of a subgraph layer.

  We derive the output node names from the input names of the succeeding
  remote op layer.

  Args:
    remote_op_layer: A set of remote op names for a remote op layer.
    node_name_to_node_def: A mapping from node names to 'NodeDef' protos.

  Returns:
    A set of output node names of a subgraph layer.
  """
  output_node_names = set([])

  for remote_op_name in remote_op_layer:
    for input_node_name in node_name_to_node_def[remote_op_name].input:
      input_node = node_name_to_node_def[input_node_name]

      # Assumption: Later in beam, the placeholder/graph inputs are loaded
      #             for us.
      if not _is_placeholder_op(input_node):
        output_node_names.add(input_node_name)

  return output_node_names


def _partition_one_subgraph_layer(
    graph_def: Any,
    graph: Any,
    node_name_to_node_def: Mapping[Text, Any],
    previous_layers_visited: Set[Text],
    output_node_names: Set[Text]
) -> ExecutionSpec:
  """Partition one subgraph layer.

  As discussed in _get_execution_specs(), a subgraph layer contains one or
  more nodes excluding remote ops. Based on a set of output node names, we
  perform a BFS, where it stops when it encounters a placeholder, a remote op,
  a visited node for this layer, or a visited node from other layers.

  We may encounter different types of nodes as we traverse. Since remote ops
  are the outputs of the remote op layers and placeholder ops are the
  pre-loaded graph inputs, they must be computed before this subgraph layer.
  Therefore, we can add them as placeholder inputs. Nodes other than
  the remote ops and the placeholder ops can be considered as 'regular nodes',
  where we can add them directly to the subgraph layer.

  Args:
    graph_def: A GraphDef proto for the original graph.
    graph: A tf.Graph instance for the original graph.
    node_name_to_node_def: A mapping from node names to 'NodeDef' protos.
    previous_layers_visited: A set of node names that have been visited
                             by the previously processed layers.
    output_node_names: A set of output node names for the current layer.

  Returns:
    An ExecutionSpec representing a subgraph layer.
  """
  subgraph = graph_pb2.GraphDef()
  subgraph.versions.CopyFrom(graph_def.versions)
  subgraph.library.CopyFrom(graph_def.library)

  queue = list(output_node_names)

  # Nodes we've visited that belong to the current layer.
  current_layer_visited = set([])
  # Nodes coming from other layers.
  nodes_from_other_layers = set([])

  while queue:
    current_node_name = queue[0]
    current_node = node_name_to_node_def[current_node_name]
    del queue[0]

    if _is_remote_op(current_node) or _is_placeholder_op(current_node):
      # These ops must be computed before this subgraph layer. Hence,
      # we treat them as placeholder inputs.
      if current_node_name not in current_layer_visited:
        placeholder_node = _create_placeholder_node_from_existing_node(
            current_node, graph)
        subgraph.node.append(placeholder_node)

        current_layer_visited.add(current_node_name)
    else:
      # A regular node coming from another subgraph layer may not be an
      # output of that subgraph layer. We need to check for that later.
      if current_node_name in previous_layers_visited:
        nodes_from_other_layers.add(current_node_name)

      elif current_node_name not in current_layer_visited:
        subgraph.node.append(current_node)

        current_layer_visited.add(current_node_name)
        queue.extend(list(node_name_to_node_def[current_node_name].input))

  return ExecutionSpec(
      subgraph=subgraph,
      input_names=_get_input_names(subgraph),
      output_names=set(output_node_names),
      is_remote_op=False,
      body_node_names=_get_body_node_names(subgraph),
      nodes_from_other_layers=nodes_from_other_layers)


def _get_input_names(subgraph: Any) -> Set[Text]:
  input_names = {node.name for node in subgraph.node
                 if _is_placeholder_op(node)}
  return input_names


def _get_body_node_names(subgraph: Any) -> Set[Text]:
  body_node_names = {node.name for node in subgraph.node
                     if not _is_placeholder_op(node)}
  return body_node_names


def _handle_nodes_from_other_layers(
    current_spec: ExecutionSpec,
    execution_specs: List[ExecutionSpec],
    graph: Any,
    node_name_to_node_def: Mapping[Text, Any]
) -> ExecutionSpec:
  """Handle nodes that are from other layers.

  If a node from another layer isn't one of the outputs of that layer, we
  need to add it to the output names. We also need to add it as a placeholder
  input to the current layer.

  Args:
    current_spec: An ExecutionSpec for the current layer.
    execution_specs: A list of ExecutionSpecs for the previously partitioned
                     layers.
    graph: A tf.Graph() instance for the original graph.
    node_name_to_node_def: A mapping from node names to 'NodeDef' protos.

  Returns:
    An ExecutionSpec for the current layer with nodes from other layers
    added as placeholder inputs.
  """
  for node_name in current_spec.nodes_from_other_layers:
    for previous_spec in execution_specs:

      if node_name in previous_spec.body_node_names:
        previous_spec.output_names.add(node_name)
        current_spec.input_names.add(node_name)

        node = node_name_to_node_def[node_name]
        placeholder = _create_placeholder_node_from_existing_node(node, graph)
        current_spec.subgraph.node.append(placeholder)

  return current_spec


def _partition_one_remote_op_layer(
    remote_op_layer: Set[Text],
    node_name_to_node_def: Mapping[Text, Any]
) -> List[ExecutionSpec]:
  """Construct ExecutionSpecs for a remote op layer.

  As discussed in _get_execution_specs(), a remote op layer contains one
  or more remote ops having no dependencies on each other. However, instead of
  having one ExecutionSpec to store a layer (as it is with subgraph layer),
  we use multiple ExecutionSpecs to represent a remote op layer.

  Args:
    remote_op_layer: A set of remote op names for a remote op layer.
    node_name_to_node_def: A mapping from node names to 'NodeDef' protos.

  Returns:
    A list of ExecutionSpecs representing a remote op layer.
  """
  list_of_specs = []

  for remote_op_name in remote_op_layer:
    spec = ExecutionSpec(
        subgraph=None,
        input_names=set(node_name_to_node_def[remote_op_name].input),
        output_names=set([remote_op_name]),
        is_remote_op=True,
        body_node_names=set([remote_op_name]),
        nodes_from_other_layers=set([]))
    list_of_specs.append(spec)

  return list_of_specs


def _create_placeholder_node(
    dtype: Any, shape: Any, name: Text) -> Any:
  with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    tf.compat.v1.placeholder(dtype=dtype, shape=shape, name=name)
    return sess.graph_def.node[0]  # The only node


def _create_placeholder_node_from_existing_node(
    node: Any, graph: Any) -> Any:
  operation = graph.get_operation_by_name('import/%s' % (node.name))
  dtype = operation.outputs[0].dtype
  return _create_placeholder_node(dtype=dtype,
                                  shape=None,
                                  name=node.name)


class _RemoteOpLayers:
  """A class that outputs remote op layers (custom topological sort).

  A remote op layer contains a set of remote op names that don't have
  dependencies on each other. The 'next' remote op layer refers to
  a remote op layer that is ready to execute.
  """
  def __init__(self, remote_op_to_parents: Mapping[Text, List[Text]]):
    """Initializes the class.

    Args:
      remote_op_to_parents: A mapping from a remote op name to a list of
                            immediate remote op parent names.
    """
    self.remote_op_to_parents = remote_op_to_parents

  def __iter__(self):
    """Initialize the iterator."""
    self._processed = set([])
    self._not_processed = set(self.remote_op_to_parents.keys())
    return self

  def __next__(self) -> Set[Text]:
    """Get the remote op names for the next remote op layer.

    Returns:
      A set of remote op names.
    """
    if not self._not_processed:
      raise StopIteration

    layer_node_names = set([])
    for remote_op_name in self._not_processed:
      remote_op_parents = set(self.remote_op_to_parents[remote_op_name])
      if remote_op_parents.issubset(self._processed):
        layer_node_names.add(remote_op_name)

    self._not_processed -= layer_node_names
    self._processed |= layer_node_names

    return layer_node_names
