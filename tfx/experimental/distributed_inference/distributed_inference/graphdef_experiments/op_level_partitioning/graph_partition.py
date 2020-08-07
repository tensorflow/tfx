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

This implementation follows the idea of op-level partitioning, where
  we partition each graph into its smallest unit: a node. Here, each
  partitioned subgraph only represents one node from the original graph.

Definition:
  1. "op" refers to a graph name.
      In our example: op = {'main', 'remote_op_b', 'remote_op_a'}.

  2. "execution_specs" refers to a structure passed to the beam pipeline.
     It is a list, where we need to execute spec 1 before later specs.

  <--closer to inputs                   closer to outputs-->
      --------------------------------------------------
      |          |          |          |               |
      |  spec 1  |  spec 2  |  spec 3  |  ......       |
      |          |          |          |               |
      --------------------------------------------------
               /              \
              /                \
             / ---------------- \
               |   Subgraph   |
               ----------------
               |    Inputs    |
               ----------------
               |   Outputs    |
               ----------------
               | is_remote_op |
               ----------------

Key Assumptions/Limitations for this implementation:
  1. Inputs should be GraphDefs
  2. Only one output per node/op
  3. Not supporting tf.functions
  4. Unknown consequences with tf.variable
"""

import tensorflow as tf
from tensorflow.core.framework import graph_pb2


def get_op_to_graph_def(op_to_filepath):
  """Import graph_defs.

  The current implementation loads graph_defs from memory."""
  op_to_graph_def = {op: _get_graph_def(filepath)
                     for op, filepath in op_to_filepath.items()}
  return op_to_graph_def


def _get_graph_def(filepath):
  graph_def = graph_pb2.GraphDef()
  with tf.compat.v1.gfile.FastGFile(filepath, 'rb') as f:
    graph_def.ParseFromString(f.read())
  return graph_def


def partition_all_graphs(op_to_graph_def):
  """Partition all the graph_defs.

  In our example, we have three graphs: 'main', 'remote_op_a', 'remote_op_b'.
  Here, we partition each graph given its graph_def and output names.'

  Args:
    op_to_graph_def: {a graph name: a GraphDef proto}

  Returns:
    op_to_execution_specs: {a graph name: a list of execution specs}
      each execution spec is a dict which comprises:
        {'subgraph': a GraphDef,
         'inputs': a set of input node names,
         'outputs': a set of output node names,
         'is_remote_op': a Boolean}
  """
  op_to_execution_specs = {op: _partition_one_graph(op_to_graph_def[op])
                           for op in op_to_graph_def}
  return op_to_execution_specs


def _partition_one_graph(graph_def):
  """Partition a graph_def.

  Args:
    graph_def: a GraphDef proto

  Returns:
    execution_specs: a list of specs, each spec is a dict that contains:
      {'subgraph': a GraphDef,
       'inputs': a set of input node names,
       'outputs': a set of output node names,
       'is_remote_op': a Boolean}
  """
  graph = _get_graph(graph_def)
  node_name_to_node_def = _get_node_name_to_node_def(graph_def)
  node_name_to_input_names = _get_node_name_to_input_names(graph_def)

  execution_specs = _get_execution_specs(graph_def,
                                         graph,
                                         node_name_to_node_def,
                                         node_name_to_input_names)
  return execution_specs


def _get_graph(graph_def):
  temp = tf.Graph()
  with temp.as_default():
    tf.import_graph_def(graph_def)
    return tf.compat.v1.get_default_graph()


def _get_node_name_to_node_def(graph_def):
  return {node.name: node for node in graph_def.node}


def _get_node_name_to_input_names(graph_def):
  return {node.name: list(node.input) for node in graph_def.node}


def _get_execution_specs(graph_def,
                         graph,
                         node_name_to_node_def,
                         node_name_to_input_names):
  """Generate the execution_specs for a graph.

  Note that the execution_specs captures the order of execution by the order
    of the list. Execution_spec with smaller index executes first.

  Args:
    graph_def: a GraphDef proto
    graph: a tf.Graph instance
    node_name_to_node_def: {node name: a NodeDef proto}
    node_name_to_input_names: {node name: a list of node input names}

  Returns:
    execution_specs: a list of specs, each spec is a dict that includes:
                     {'subgraph': a GraphDef,
                      'inputs': a set of input node names,
                      'outputs': a set of output node names,
                      'is_remote_op': a Boolean}
  """
  execution_specs = []
  relations = Relations(node_name_to_input_names)

  while not relations.check_if_finished():
    next_node_name = relations.get_next_node_name()
    next_node = node_name_to_node_def[next_node_name]

    # We assume that placeholder inputs are given to us.
    if _is_placeholder_op(next_node):
      pass

    elif _is_remote_op(next_node):
      execution_specs.append(
          _get_remote_op_spec(next_node))

    else:
      execution_specs.append(
          _get_regular_op_spec(next_node, graph_def, graph))

  return execution_specs


def _is_placeholder_op(node):
  return node.op == "Placeholder"


def _is_remote_op(node):
  return node.op == "PyFunc"


def _get_remote_op_spec(remote_node):
  """Get the execution spec for a remote op."""
  return {'subgraph': None,
          'inputs': set(remote_node.input),
          'outputs': {remote_node.name},
          'is_remote_op': True}


def _get_regular_op_spec(regular_node, graph_def, graph):
  """Get the execution spec for a regular op."""
  subgraph = graph_pb2.GraphDef()
  subgraph.versions.CopyFrom(graph_def.versions)
  subgraph.library.CopyFrom(graph_def.library)

  regular_node = _remove_colocated_attr(regular_node)

  # Using tf.Graph() to get the dtype of the inputs.
  node = graph.get_operation_by_name('import/%s' % regular_node.name)
  for input_node in node.inputs:
    subgraph.node.append(
        _create_placeholder_node(dtype=input_node.dtype,
                                 shape=None,
                                 name=_get_node_name(input_node.name)))
  subgraph.node.append(regular_node)

  return {'subgraph': subgraph,
          'inputs': set(regular_node.input),
          'outputs': {regular_node.name},
          'is_remote_op': False}


def _remove_colocated_attr(node):
  """Remove the colocated attributes when performing op-level partitioning.

  If not, errors will arise when executing the partitioned subgraphs.
  """
  if '_class' in node.attr:
    del node.attr['_class']
  return node


def _get_node_name(node_name):
  """Remove the prefix "import/" and the postfix ":0" """
  return node_name[7:-2]


def _create_placeholder_node(dtype, shape, name):
  temp = tf.Graph()
  with temp.as_default():
    tf.compat.v1.placeholder(dtype=dtype, shape=shape, name=name)
    return temp.as_graph_def().node[0]  # The first and the only node


class Relations:
  """A class that outputs the order of TF graph execution."""
  def __init__(self, node_name_to_input_names):
    self.relations = node_name_to_input_names
    self.processed = set([])
    self.to_be_processed = set(self.relations.keys())

  def check_if_finished(self):
    return not self.to_be_processed

  def get_next_node_name(self):
    """Get the next node name to execute."""
    for node_name in self.to_be_processed:
      node_input_names = set(self.relations[node_name])

      if node_input_names.issubset(self.processed):
        self.processed.add(node_name)
        self.to_be_processed.remove(node_name)
        return node_name
    return ''
