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
"""Graph partitioning on TensorFlow GraphDefs with remote ops.

The current partitioning strategy aims for maximum subgraphs.

There are two libraries representing two stages: graph_partition and
beam_pipeline. In this library, we take in some GraphDef inputs, partition
the graphs, and store the partitioned subgraphs into lists of ExecutionSpecs.
The order of the list represents the order of execution. Take a look at
beam_pipeline if you want to run the partitioned subgraphs.

The current implementation has some key limitations:
  1. This library only accepts GraphDefs (or their filepaths) as the inputs.
  2. All the node/op should only have one output.
  3. This library doesn't support tf.variable.
  4. This library doesn't support tf.function.

  Typical usage example:
  ```
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
  ```
"""

import collections
from typing import Dict, List, Mapping, Set
import tensorflow as tf
from tfx.dsl.io import fileio
from tfx.experimental.distributed_inference.graphdef_experiments.subgraph_partitioning import execution_spec


def get_graph_name_to_graph_def(
    graph_name_to_filepath: Mapping[str, str]
) -> Dict[str, tf.compat.v1.GraphDef]:
  """Gets the `GraphDef` protos from files.

  Args:
    graph_name_to_filepath: A mapping from graph names to filepaths. Each
      filepath points to a `GraphDef` proto in binary.

  Returns:
    A mapping from graph names to `GraphDef` protos.
  """
  graph_name_to_graph_def = {
      graph_name: _get_graph_def(filepath)
      for graph_name, filepath in graph_name_to_filepath.items()
  }
  return graph_name_to_graph_def


def _get_graph_def(filepath: str) -> tf.compat.v1.GraphDef:
  graph_def = tf.compat.v1.GraphDef()
  with fileio.open(filepath, 'rb') as f:
    graph_def.ParseFromString(f.read())
  return graph_def


def partition_all_graphs(
    graph_name_to_graph_def: Mapping[str, tf.compat.v1.GraphDef],
    graph_name_to_output_names: Mapping[str, List[str]]
) -> Dict[str, List[execution_spec.ExecutionSpec]]:
  """Partitions all the graphs.

  For each graph, the partitioning algorithm takes in the graph's `GraphDef`
  proto and output names, partitions the graph, and returns a list of
  ExecutionSpecs. Later, the beam_pipeline library can take in the
  ExecutionSpecs and execute the partitioned subgraphs.

  Args:
    graph_name_to_graph_def: A mapping from graph names to `GraphDef` protos.
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
    graph_def: tf.compat.v1.GraphDef,
    output_names: List[str]) -> List[execution_spec.ExecutionSpec]:
  """Partitions one graph.

  Args:
    graph_def: A `GraphDef` proto for that graph.
    output_names: A list of graph's output node names.

  Returns:
    A list of ExecutionSpecs.
  """
  graph = _get_graph(graph_def)
  node_name_to_node_def = _get_node_name_to_node_def(graph_def)

  remote_op_to_immediate_dep = _get_remote_op_to_immediate_dep(
      node_name_to_node_def)

  specs = _get_execution_specs(graph_def, output_names, graph,
                               node_name_to_node_def,
                               remote_op_to_immediate_dep)

  _modify_execution_specs_for_input_validity(specs)

  return specs


def _get_graph(graph_def: tf.compat.v1.GraphDef) -> tf.Graph:
  with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    tf.import_graph_def(graph_def)
    return sess.graph


def _get_node_name_to_node_def(
    graph_def: tf.compat.v1.GraphDef) -> Dict[str, tf.compat.v1.NodeDef]:
  return {node.name: node for node in graph_def.node}


def _get_remote_op_to_immediate_dep(
    node_name_to_node_def: Mapping[str, tf.compat.v1.NodeDef]
) -> Dict[str, List[str]]:
  """Gets the execution dependencies between remote ops.

  The remote op immediate dependencies must be executed before executing
  a remote op.

  Args:
    node_name_to_node_def: A mapping from node names to `NodeDef` protos.

  Returns:
    A mapping from a remote op name to a set of remote op immediate
    dependencies' names.
  """
  remote_op_to_immediate_dep = {}

  for node in node_name_to_node_def.values():
    if _is_remote_op(node):
      remote_op_to_immediate_dep[node.name] = _get_remote_op_immediate_dep(
          node.name, node_name_to_node_def)

  return remote_op_to_immediate_dep


def _get_remote_op_immediate_dep(
    remote_op_name: str,
    node_name_to_node_def: Mapping[str, tf.compat.v1.NodeDef]) -> List[str]:
  """Finds the remote op immediate dependencies for a remote op.

  Args:
    remote_op_name: The name of the child remote op.
    node_name_to_node_def: A mapping from node names to `NodeDef` protos.

  Returns:
    A list of remote op immediate dependencies' names.
  """
  queue = collections.deque([remote_op_name])
  visited = set([remote_op_name])
  remote_op_immediate_dep = []

  while queue:
    current_node_name = queue.popleft()

    for input_node_name in node_name_to_node_def[current_node_name].input:
      if input_node_name not in visited:
        visited.add(input_node_name)
        input_node = node_name_to_node_def[input_node_name]

        # Stop traversing when reaching a remote op.
        if _is_remote_op(input_node):
          remote_op_immediate_dep.append(input_node_name)
        else:
          queue.append(input_node_name)

  return remote_op_immediate_dep


def _is_placeholder_op(node: tf.compat.v1.NodeDef) -> bool:
  return node.op == 'Placeholder'


def _is_remote_op(node: tf.compat.v1.NodeDef) -> bool:
  return node.op == 'PyFunc'


def _get_execution_specs(
    graph_def: tf.compat.v1.GraphDef, graph_output_names: List[str],
    graph: tf.Graph, node_name_to_node_def: Mapping[str, tf.compat.v1.NodeDef],
    remote_op_to_immediate_dep: Mapping[str, List[str]]
) -> List[execution_spec.ExecutionSpec]:
  """Generates the ExecutionSpecs for a graph.

  A "layer" contains one or more nodes inside a graph. There are two types of
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
  remote op layer need to be stored into N ExecutionSpecs, where N equals
  to the number of remote ops inside a remote op layer. This happens
  because each remote op essentially represents a graph.

  Args:
    graph_def: A `GraphDef` proto.
    graph_output_names: A list of graph output node names.
    graph: A tf.Graph representing the same graph as graph_def.
    node_name_to_node_def: A mapping from node names to `NodeDef` protos.
    remote_op_to_immediate_dep: A mapping from remote op name to a list of
      remote op immediate dependencies' names.

  Returns:
    A list of ExecutionSpecs, where the order of the list represents the
    order of execution.
  """
  execution_specs = []  # type: List[execution_spec.ExecutionSpec]
  previously_visited = set()  # type: Set[str]

  for remote_op_layer in _RemoteOpLayers(remote_op_to_immediate_dep):
    # Get one subgraph layer
    output_node_names = _get_previous_subgraph_layer_output_node_names(
        remote_op_layer, node_name_to_node_def)

    if output_node_names:
      spec = _get_execution_spec_for_subgraph_layer(graph_def, graph,
                                                    node_name_to_node_def,
                                                    previously_visited,
                                                    output_node_names)
      execution_specs.append(spec)
      previously_visited |= _get_non_input_names(spec.subgraph)

    # Get one remote op layer
    specs = _get_execution_specs_for_remote_op_layer(remote_op_layer,
                                                     node_name_to_node_def)
    execution_specs.extend(specs)

  # Get the last subgraph layer
  output_node_names = set(graph_output_names)
  spec = _get_execution_spec_for_subgraph_layer(graph_def, graph,
                                                node_name_to_node_def,
                                                previously_visited,
                                                output_node_names)
  execution_specs.append(spec)

  return execution_specs


def _get_previous_subgraph_layer_output_node_names(
    remote_op_layer: Set[str],
    node_name_to_node_def: Mapping[str, tf.compat.v1.NodeDef]) -> Set[str]:
  """Gets the output node names of the previous subgraph layer.

  Given a remote op layer, we derive the output node names of the previous
  subgraph layer. Layers tend to have the following order: subgraph layer,
  remote op layer, subgraph layer, remote op layer, ...

  Args:
    remote_op_layer: A set of remote op names for a remote op layer.
    node_name_to_node_def: A mapping from node names to `NodeDef` protos.

  Returns:
    A set of output node names of the previous subgraph layer.
  """
  previous_subgraph_layer_output_node_names = set()

  for remote_op_name in remote_op_layer:
    for input_node_name in node_name_to_node_def[remote_op_name].input:
      input_node = node_name_to_node_def[input_node_name]

      # Assumption: Graph inputs and previous remote op outputs are always
      #             computed and stored.
      if _is_placeholder_op(input_node) or _is_remote_op(input_node):
        continue
      previous_subgraph_layer_output_node_names.add(input_node_name)

  return previous_subgraph_layer_output_node_names


def _get_execution_spec_for_subgraph_layer(
    graph_def: tf.compat.v1.GraphDef, graph: tf.Graph,
    node_name_to_node_def: Mapping[str, tf.compat.v1.NodeDef],
    previously_visited: Set[str],
    output_node_names: Set[str]) -> execution_spec.ExecutionSpec:
  """Constructs one subgraph layer.

  As discussed in _get_execution_specs(), a subgraph layer contains one or
  more nodes excluding remote ops. Based on a set of output node names, we
  traverse toward the ancestors (upward) until encountering a "special" node.

  Here, we traverse upward because each node's node_def contains input names
  but not output names.

  A "special" node could be either a placeholder node, a remote op, or a node
  visited by a previous layer. Since it is computed/stored prior to the
  current subgraph layer, we can treat it as an input of the current subgraph
  layer.

  Args:
    graph_def: A `GraphDef` proto for the original graph.
    graph: A tf.Graph instance for the original graph.
    node_name_to_node_def: A mapping from node names to `NodeDef` protos.
    previously_visited: A set of node names from previous subgraph layers.
    output_node_names: A set of output node names for the current subgraph.

  Returns:
    An ExecutionSpec representing a subgraph layer.
  """
  subgraph = tf.compat.v1.GraphDef()
  subgraph.versions.CopyFrom(graph_def.versions)
  subgraph.library.CopyFrom(graph_def.library)

  queue = collections.deque(output_node_names)
  visited = set()

  while queue:
    current_node_name = queue.popleft()
    current_node = node_name_to_node_def[current_node_name]

    if current_node_name not in visited:
      visited.add(current_node_name)

      if (_is_remote_op(current_node) or _is_placeholder_op(current_node) or
          current_node_name in previously_visited):
        # These ops must be computed before this subgraph layer. Hence,
        # we treat them as placeholder inputs.
        placeholder_node = _create_placeholder_node_from_existing_node(
            current_node, graph)
        subgraph.node.append(placeholder_node)

      else:
        subgraph.node.append(current_node)
        queue.extend(node_name_to_node_def[current_node_name].input)

  return execution_spec.ExecutionSpec(
      subgraph=subgraph,
      input_names=_get_input_names(subgraph),
      output_names=set(output_node_names),
      is_remote_op=False)


def _create_placeholder_node_from_existing_node(
    node: tf.compat.v1.NodeDef, graph: tf.Graph) -> tf.compat.v1.NodeDef:
  """Creates a placeholder node to represent an existing node.

  Some partitioned subgraphs may require inputs that are loaded or computed
  previously. Hence, we replace the input nodes with placeholder nodes that
  share the same name, shape, and dtype. Now the inputs become placeholders
  inside partitioned subgraphs, and can be loaded by feed dicts at the runtime.

  Args:
    node: A `NodeDef` proto for the existing node.
    graph: A tf.Graph instance for the graph that contains the existing node.

  Returns:
    A `NodeDef` proto that stores a placeholder node.
  """
  operation = graph.get_operation_by_name('import/%s' % (node.name))
  output_tensor = operation.outputs[0]

  with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    tf.compat.v1.placeholder(
        dtype=output_tensor.dtype, shape=output_tensor.shape, name=node.name)
    return sess.graph_def.node[0]


def _get_input_names(subgraph: tf.compat.v1.GraphDef) -> Set[str]:
  input_names = {
      node.name for node in subgraph.node if _is_placeholder_op(node)
  }
  return input_names


def _get_non_input_names(subgraph: tf.compat.v1.GraphDef) -> Set[str]:
  non_input_names = {
      node.name for node in subgraph.node if not _is_placeholder_op(node)
  }
  return non_input_names


def _get_execution_specs_for_remote_op_layer(
    remote_op_layer: Set[str],
    node_name_to_node_def: Mapping[str, tf.compat.v1.NodeDef]
) -> List[execution_spec.ExecutionSpec]:
  """Constructs ExecutionSpecs for a remote op layer.

  As discussed in _get_execution_specs(), a remote op layer contains one
  or more remote ops having no dependencies on each other. However, instead of
  having one ExecutionSpec to store a layer (as it is with subgraph layer),
  we use multiple ExecutionSpecs to represent a remote op layer.

  Args:
    remote_op_layer: A set of remote op names for a remote op layer.
    node_name_to_node_def: A mapping from node names to `NodeDef` protos.

  Returns:
    A list of ExecutionSpecs representing a remote op layer.
  """
  list_of_specs = []

  for remote_op_name in remote_op_layer:
    spec = execution_spec.ExecutionSpec(
        subgraph=None,
        input_names=set(node_name_to_node_def[remote_op_name].input),
        output_names=set([remote_op_name]),
        is_remote_op=True)
    list_of_specs.append(spec)

  return list_of_specs


def _modify_execution_specs_for_input_validity(
    specs: List[execution_spec.ExecutionSpec]) -> None:
  """Modifies the execution specs to ensure that all inputs are valid.

  Ensure inputs have been outputted by previous specs. Sometimes an input
  of a spec may be a node from a previous spec but not one of the outputs.
  We'd like to add it to previous spec's outputs.

  Args:
    specs: A list of ExecutionSpecs, where order of the list represents the
      order of the execution.
  """
  for current_spec_index, current_spec in enumerate(specs):
    for previous_spec in specs[:current_spec_index]:
      if previous_spec.is_remote_op:
        continue
      _add_current_spec_input_to_previous_spec_output(current_spec,
                                                      previous_spec)


def _add_current_spec_input_to_previous_spec_output(
    current_spec: execution_spec.ExecutionSpec,
    previous_spec: execution_spec.ExecutionSpec) -> None:
  for input_name in current_spec.input_names:
    if input_name in _get_non_input_names(previous_spec.subgraph):
      # Output names is a set, which doesn't allow duplicates.
      previous_spec.output_names.add(input_name)


class _RemoteOpLayers:
  """A class that outputs remote op layers (custom topological sort).

  A remote op layer contains a set of remote op names that don't have
  dependencies on each other. The remote op layers are returned in execution
  order. In other words, a remote op layer returned earlier will be executed
  earlier.
  """

  def __init__(self, remote_op_to_immediate_dep: Mapping[str, List[str]]):
    """Initializes the class.

    Args:
      remote_op_to_immediate_dep: A mapping from a remote op name to a list of
        remote op immediate dependencies' names.
    """
    self.remote_op_to_immediate_dep = remote_op_to_immediate_dep

  def __iter__(self):
    self._not_processed = set(self.remote_op_to_immediate_dep.keys())
    return self

  def __next__(self) -> Set[str]:
    """Gets the remote op names for the next remote op layer.

    Returns:
      A set of remote op names.
    """
    if not self._not_processed:
      raise StopIteration

    layer_node_names = set()
    for remote_op_name in self._not_processed:
      remote_op_immediate_dep = set(
          self.remote_op_to_immediate_dep[remote_op_name])
      if not remote_op_immediate_dep & self._not_processed:
        layer_node_names.add(remote_op_name)

    self._not_processed -= layer_node_names

    return layer_node_names
