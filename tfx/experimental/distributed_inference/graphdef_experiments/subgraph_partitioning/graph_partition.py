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

The current implementation targets two goals:
    1. Maximal subgraphs
    2. Avoid repeated work when running subgraphs in beam
    
Definition:
    1. "op" refers to a graph name. 
        In our example: op = {'main', 'remote_op_b', 'remote_op_a'}.

    2. "execution_specs" refers to a structure (list) passed to the beam pipeline.
       We execute spec 1, spec 2, spec 3 in order inside a beam pipeline.
    
    <--closer to inputs                   closer to outputs-->
        --------------------------------------------------
        |          |          |          |               |
        |  spec 1  |  spec 2  |  spec 3  | ......        |
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
    1. Inputs should be GraphDefs.
    2. Only one output per node/op.
    3. Not supporting tf.functions
    4. Unknown consequences with tf.variable
"""

import tensorflow as tf
from tensorflow.core.framework import graph_pb2


def get_op_to_graph_def(op_to_filepath):
    """Import graph_defs.
    
    The current implementation loads graph_defs from memory."""
    op_to_graph_def = {op: _get_graph_def(filepath) for op, filepath
                       in op_to_filepath.items()}
    return op_to_graph_def


def _get_graph_def(filepath):
    graph_def = graph_pb2.GraphDef()
    with tf.compat.v1.gfile.FastGFile(filepath, 'rb') as f:
        graph_def.ParseFromString(f.read())
    return graph_def


def partition_all_graphs(op_to_graph_def, op_to_outputs):
    """Partition all the graph_defs.
    
    In our example, op/graph_name contains {'main', 'remote_op_b', 'remote_op_a'}.
    Here, we partition each graph given its graph_def and output names.
    
    Arguments:
        op_to_graph_def: {a graph name: a GraphDef proto}
        op_to_outputs: {a graph name: a list of output node names}
    
    Returns:
        op_to_execution_specs: {a graph name: a list of execution specs}
            each execution spec is a dict which comprises:
                {'subgraph': a GraphDef,
                 'inputs': a set of input node names,
                 'outputs': a set of output node names,
                 'is_remote_op': a Boolean}
    """
    op_to_execution_specs = {}
    for op in op_to_graph_def:
        execution_specs = _partition_one_graph(op_to_graph_def[op],
                                               op_to_outputs[op])
        op_to_execution_specs[op] = execution_specs
    return op_to_execution_specs


def _partition_one_graph(graph_def, outputs):
    """Partition a graph_def.
    
    Arguments:
        graph_def: a GraphDef proto
        outputs: a set of output node names
    
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
    
    remote_op_relations = _get_remote_op_relations(graph_def,
                                                   node_name_to_node_def,
                                                   node_name_to_input_names)
    
    execution_specs = _get_execution_specs(graph_def,
                                           graph,
                                           node_name_to_node_def,
                                           node_name_to_input_names,
                                           remote_op_relations,
                                           outputs)
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


def _get_remote_op_relations(graph_def, node_name_to_node_def, node_name_to_input_names):
    """Get the execution dependencies between remote ops.
      
    The remote op children must be executed before executing a remote op.
    
    Arguments:
        graph_def: a GraphDef proto
        node_name_to_node_def: {node name: a NodeDef proto}
        node_name_to_input_names: {node name: a list of node input names}
        
    Returns:
        remote_op_relations: {remote op: a list of remote op children}
    """
    remote_op_relations = {}
    
    for node in graph_def.node:
        if _is_remote_op(node):
            remote_op_relations[node.name] = _get_remote_op_children(node.name,
                                                                     node_name_to_node_def,
                                                                     node_name_to_input_names)
    return remote_op_relations


def _get_remote_op_children(remote_op_name, node_name_to_node_def, node_name_to_input_names):
    """Find the remote op children for a remote op."""
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
                         graph,
                         node_name_to_node_def,
                         node_name_to_input_names,
                         remote_op_relations,
                         graph_outputs):
    """Generate the execution_specs for a graph.
    
    Note that the execution_specs captures the order of execution by the order
        of the list. Execution_spec with smaller index executes first.
    
    Arguments:
        graph_def: a GraphDef proto
        graph: a tf.Graph instance
        node_name_to_node_def: {node name: a NodeDef proto}
        node_name_to_input_names: {node name: a list of node input names}
        remote_op_relations: {remote op: a list of remote op children}
        graph_outputs: {graph name: a list of output node names}
        
    Returns:
        execution_specs: a list of specs, each spec is a dict that includes:
                         {'subgraph': a GraphDef,
                          'inputs': a set of input node names,
                          'outputs': a set of output node names,
                          'is_remote_op': a Boolean}
    """
    execution_specs = []
    previous_layers_visited = set([])
    order = Relations(remote_op_relations)
    
    while not order.check_if_finished():
        remote_ops_one_layer = order.get_next_layer()
        
        # Handle one subgraph layer
        layer_output_node_names = _get_subgraph_layer_output_node_names(remote_ops_one_layer,
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
                                                            graph,
                                                            node_name_to_node_def)
            execution_specs.append(subgraph_spec)
            previous_layers_visited = previous_layers_visited.union(subgraph_spec['body_nodes'])
        
        # Handle one remote op layer
        remote_op_specs = _partition_one_remote_op_layer(remote_ops_one_layer, node_name_to_input_names)
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
                                                    graph,
                                                    node_name_to_node_def)
    execution_specs.append(subgraph_spec)
    previous_layers_visited = previous_layers_visited.union(subgraph_spec['body_nodes'])

    return execution_specs


def _get_subgraph_layer_output_node_names(remote_ops_one_layer, 
                                          node_name_to_node_def, 
                                          node_name_to_input_names):
    """Get the output names of a subgraph layer.
    
    Note that they are equivalant to the input names of the succeeding remote op layer."""
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
    
    Arguments:
        previous_layers_visited: a set of nodes
        graph_def: a GraphDef proto
        graph: a tf.Graph
        output_names: a list of output names
        node_name_to_node_def: {node name: a NodeDef proto}
        node_name_to_input_names: {node name: a list of node input names}
        
    Returns:
        An execution spec, which stores:
        {'subgraph': a graph_def, 
         'inputs': a set of node names, 
         'outputs': a set of node names,
         'is_remote_op': a boolean status,
         'body_nodes': a set of node names,
         'nodes_from_other_layers': a set of node names from the previous layers}
        
        Here, body_nodes and nodes_from_other_layers only serve to handle input
            nodes from the previously visited subgraph layers.
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
                placeholder_node = _create_placeholder_node_from_existing_node(current_node, graph)
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
            
    return {'subgraph': subgraph, 
            'inputs': _get_inputs_from_subgraph(subgraph), 
            'outputs': set(output_names),
            'is_remote_op': False,
            'body_nodes': _get_regular_nodes_from_subgraph(subgraph),
            'nodes_from_other_layers': nodes_from_other_layers}
    

def _handle_nodes_from_other_layers(current_spec, execution_specs, graph, node_name_to_node_def):
    """Handle nodes that are from other layers.
    
    Add it to other layer's output, add it to current layer's input,
        and add a placeholder node to current layer.
    """
    for node_name in current_spec['nodes_from_other_layers']:
        for previous_spec in execution_specs:
            
            if node_name in previous_spec['body_nodes']:
                previous_spec['outputs'].add(node_name)
                current_spec['inputs'].add(node_name)
                
                node = node_name_to_node_def[node_name]
                placeholder = _create_placeholder_node_from_existing_node(node, graph)
                current_spec['subgraph'].node.append(placeholder)
                
    return current_spec
        

def _partition_one_remote_op_layer(remote_op_names, node_name_to_input_names):
    """Construct spec for remote ops"""
    list_of_specs = []
    for remote_op_name in remote_op_names:
        spec = {'subgraph': None, 
                'inputs': set(node_name_to_input_names[remote_op_name]), 
                'outputs': set([remote_op_name]), 
                'is_remote_op': True,
                'body_nodes': set([remote_op_name]),
                'nodes_from_other_layers': None}
        list_of_specs.append(spec)
    
    return list_of_specs


def _create_placeholder_node(dtype, shape, name):
    temp = tf.Graph()
    with temp.as_default():
        placeholder = tf.compat.v1.placeholder(dtype=dtype, shape=shape, name=name)
        return temp.as_graph_def().node[0]      # The first and the only node  


def _create_placeholder_node_from_existing_node(node, graph):
    operation = graph.get_operation_by_name('import/%s' % (node.name))
    dtype = operation.outputs[0].dtype
    return _create_placeholder_node(dtype=dtype,
                                    shape=None,
                                    name=node.name)


def _get_inputs_from_subgraph(subgraph):
    inputs = set([node.name for node in subgraph.node if _is_placeholder_op(node)])
    return inputs

    
def _get_regular_nodes_from_subgraph(subgraph):
    regular_nodes = set([node.name for node in subgraph.node
                         if not _is_placeholder_op(node)])
    return regular_nodes


class Relations(object):
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
        layer_nodes = set([])
        
        for node in self.to_be_processed:
            node_inputs = set(self.relations[node])
            if node_inputs.issubset(self.processed):
                layer_nodes.add(node)
                
        for node in layer_nodes:
            self.to_be_processed.remove(node)
            self.processed.add(node)
        
        return layer_nodes