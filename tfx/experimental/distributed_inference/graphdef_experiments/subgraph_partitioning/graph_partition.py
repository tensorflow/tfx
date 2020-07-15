"""
A library for graph partitioning.

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
    2. Operating on TensorFlow Directed Acyclic Graphs (DAG).
    3. Only one output per node/op.
    4. Not supporting tf.functions

@author: jzhunybj
"""

import tensorflow as tf
from tensorflow.core.framework import graph_pb2


"""Supporting I/O"""
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


"""Partitioning"""
def partition_all_graphs(op_to_graph_def, op_to_outputs):
    """The main method to call."""
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
    
    Variables:
        graph: a tf.Graph instance
        node_name_to_node_def: {node name: a NodeDef}
        node_name_to_input_names: {node name: a list of input names}
        remote_op_relations: {remote op name: a list of remote op children}
    
    Returns:
        execution_specs: a list of specs, each spec contains:
                         {'subgraph': a GraphDef,
                          'inputs': a set of node names,
                          'outputs': a set of node names,
                          'is_remote_op': a Boolean,
                          'body_nodes': not important,
                          'nodes_from_other_layers': not important}
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
    # DEBUG
    _debug_print_execution_specs(execution_specs)
    
    return execution_specs


def _debug_print_execution_specs(execution_specs):
    for execution_spec in execution_specs:
        print('\nIs the current spec describing a remote op?', execution_spec['is_remote_op'])
        print('Inputs:', execution_spec['inputs'])
        print('Outputs:', execution_spec['outputs'])
        print('Body nodes:', execution_spec['body_nodes'])


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
    """Get {remote op: a list of remote op children},
      
    These remote op children must be executed before executing the remote op.
    """
    remote_op_relations = {}
    
    for node in graph_def.node:
        if _check_remote_op(node):
            remote_op_relations[node.name] = _bfs_get_remote_op_children(node.name,
                                                                         node_name_to_node_def,
                                                                         node_name_to_input_names)
    return remote_op_relations


def _bfs_get_remote_op_children(remote_op_name, node_name_to_node_def, node_name_to_input_names):
    """Find the remote op children for a remote op"""
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
                
                if _check_remote_op(input_node):
                    remote_op_children.append(input_node_name)
                else:
                    queue.append(input_node_name)
    
    return remote_op_children


def _check_placeholder_op(node):
    return node.op == "Placeholder"


def _check_remote_op(node):
    return node.op == "PyFunc"


def _get_execution_specs(graph_def,
                         graph,
                         node_name_to_node_def,
                         node_name_to_input_names,
                         remote_op_relations,
                         graph_outputs):
    """Generate the execution_specs for a graph.
    
    In Beam, remote ops need to be handled differently than the regular nodes,
    so we divide a graph into two types of layers:
        1. A subgraph layer -- consists of regular nodes.
        2. A remote op layer -- consists of remote ops.
       
    Algorithm:
        while the remote op layers haven't been fully processed:
            1. Get the next remote op layer.
            2. Get the input names of the remote op layer, which are equivilant
               to the output names of the previous subgraph layer.
            3. Handle the previous subgraph layer.
            4. Handle the current remote op layer.
        Finally, handle the subgraph layer with graph_outputs.
       
    For the descriptions of arguments and return values, referred to
    partition_one_graph()'s DocString.
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
    
    Note that they are equivilant to the input names of the following remote op layer."""
    output_node_names = set([])
    
    for remote_op in remote_ops_one_layer:
        for input_node_name in node_name_to_input_names[remote_op]:
            input_node = node_name_to_node_def[input_node_name]
            
            # Assumption: graph inputs (placeholders) are pre-loaded before 
            #             executing the graph. This happens in beam_pipeline.
            if not _check_placeholder_op(input_node):
                output_node_names.add(input_node_name)
                
    return output_node_names


def _partition_one_subgraph_layer(previous_layers_visited,
                                  graph_def,
                                  graph,
                                  outputs,
                                  node_name_to_node_def,
                                  node_name_to_input_names):
    """Perform a modified BFS for graph partitioning.
    
    Expand from the outputs, until one of the stopping condition: remote op, 
    placeholder, visited before in this layer, or visited before in the previous layers.
    
    Arguments:
        previous_layers_visited: a set of nodes
        graph_def: a GraphDef
        graph: a tf.Graph
        outputs: desired outputs for this subgraph layer
        node_name_to_node_def: {node name: NodeDef}
        node_name_to_input_names: {node name: a list of input names}
        
    Returns:
        An execution spec, which stores:
        {'subgraph': a graph_def, 
         'inputs': a set of node names, 
         'outputs': a set of node names,
         'body_nodes': a set of node names, 
         'is_remote_op': a boolean status,
         'nodes_from_other_layers': a set of node names from the previous layers}
    """
    subgraph = graph_pb2.GraphDef()
    subgraph.versions.CopyFrom(graph_def.versions)
    subgraph.library.CopyFrom(graph_def.library)
    
    queue = list(outputs)
    current_layer_visited = set([])
    nodes_from_other_layers = set([])
    
    while queue:
        current_node_name = queue[0]
        current_node = node_name_to_node_def[current_node_name]
        del queue[0]
        
        if _check_remote_op(current_node) or _check_placeholder_op(current_node):
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
            'outputs': set(outputs),
            'body_nodes': _get_regular_nodes_from_subgraph(subgraph), 
            'is_remote_op': False,
            'nodes_from_other_layers': nodes_from_other_layers}
    

def _handle_nodes_from_other_layers(current_spec, execution_specs, graph, node_name_to_node_def):
    """Handle nodes that are from other layers.
    
    Add it to other layer's output, add it to current layer's input,
    and add a placeholder node to current layer."""
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
                'body_nodes': set([remote_op_name]), 
                'is_remote_op': True, 
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
    inputs = set([node.name for node in subgraph.node if _check_placeholder_op(node)])
    return inputs

    
def _get_regular_nodes_from_subgraph(subgraph):
    regular_nodes = set([node.name for node in subgraph.node
                         if not _check_placeholder_op(node)])
    return regular_nodes


class Relations(object):
    """A class that outputs remote op layers (custom topological sort).
    
    What is a layer? A layer is a set of nodes that are ready to execute."""
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
    
    def _debug_print_layers(self):
        while not self.check_if_finished():
            print(self.get_next_layer())
            
            
            
