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

After getting execution_specs from graph partitioning, we'd like to arrange,
load, and execute the subgraphs in beam.

Execution_specs:
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
"""

import tensorflow as tf
import apache_beam as beam
import copy


@beam.ptransform_fn
def ExecuteOneGraph(pcoll,
                    op_to_execution_specs,
                    op_to_remote_op_name_mapping,   # This information was stored in py_func's feed_dict. We don't have it.
                    graph_name):
    """Compute one graph, for example remote_op_b's graph.
    
    Main assumption:
        The parent graph has set up the placeholder inputs for the child graph.
        (So we need to setup PColl for the main graph)
    
    Arguments:
        pcoll: input PCollection, each unit contains {graph_name: {computed node: value}}
        op_to_execution_specs: {graph_name: [spec1, spec2, ...]}
            spec: {'subgraph': a graph_def, 
                   'inputs': a set of node names,
                   'outputs': a set of node names, 
                   'is_remote_op': a boolean}
        
        # Need improvements on extracting relationships!
        op_to_remote_op_name_mapping: {graph_name: {remote op name: {placeholder name inside subgraph: input name}}}
        graph_name: which graph am I currently executing
        
    Returns:
        pcoll with the intermediate/end results of this graph.
    """
    execution_specs = _get_execution_specs(graph_name, op_to_execution_specs)
    
    count = 0
    for spec in execution_specs:
        if not spec['is_remote_op']:
            count += 1
            pcoll = pcoll | str(count) >> beam.ParDo(_ExecuteOneSubgraph(),
                                                     spec,
                                                     graph_name)
        else:
            current_graph_name = graph_name
            remote_graph_name = list(spec['outputs'])[0]      # remote op is the only output
            
            count += 1
            pcoll = pcoll | str(count) >> _LoadRemoteGraphInputs(current_graph_name,
                                                                 remote_graph_name,
                                                                 op_to_remote_op_name_mapping)
            # A good place to add beam.Reshuffle()
            count += 1
            pcoll = pcoll | str(count) >> ExecuteOneGraph(op_to_execution_specs,
                                                          op_to_remote_op_name_mapping,
                                                          remote_graph_name)
            count += 1
            pcoll = pcoll | str(count) >> _ExtractRemoteGraphOutput(current_graph_name,
                                                                    remote_graph_name,
                                                                    op_to_execution_specs)
    return pcoll


def _get_execution_specs(graph_name, op_to_execution_specs):
    op_type = _get_op_type(graph_name, set(op_to_execution_specs.keys()))
    return op_to_execution_specs[op_type]


def _get_op_type(graph_name, ops):
    """As an example, graph_name='remote_op_b_1' has op_type='remote_op_b'"""
    for op_type in ops:
        if graph_name.startswith(op_type):
            return op_type


class _ExecuteOneSubgraph(beam.DoFn):    
    
    def process(self, element, spec, graph_name):
        """Executes the smallest unit: one subgraph.
        
        Arguments:
            element: a unit of PCollection, {graph_name: {computed node: value}}
            spec: an execution spec that contains things like graph_def, inputs, and outputs
            graph_name: which graph am I belonging to, ex: 'remote_op_a_1'
            
        Returns:
            Element with the recently computed outputs added.
        """
        element = copy.deepcopy(element)
        output_names = self._get_output_names(spec)
        feed_dict = self._get_feed_dict(element, spec, graph_name)
        
        graph = self._load_subgraph(spec['subgraph'])
        results = self._run_inference(graph, output_names, feed_dict)
        
        element = self._store_results(element, graph_name, output_names, results)
        yield element
        
    def _get_output_names(self, spec):
        return [_import_tensor_name(output_name) for output_name in spec['outputs']]
    
    def _get_feed_dict(self, element, spec, graph_name):
        feed_dict = {}
        for input_name in spec['inputs']:
            input_name = _import_tensor_name(input_name)
            feed_dict[input_name] = element[graph_name][input_name]
        return feed_dict
    
    def _load_subgraph(self, graph_def):
        graph = tf.Graph()
        with graph.as_default():
            tf.import_graph_def(graph_def)
        return graph
        
    def _run_inference(self, graph, output_names, feed_dict):
        with tf.compat.v1.Session(graph=graph) as sess:
            return sess.run(output_names, feed_dict=feed_dict)
    
    def _store_results(self, element, graph_name, output_names, results):
        for index in range(len(output_names)):
            output_name = output_names[index]
            element[graph_name][output_name] = results[index]
        return element


def _import_tensor_name(node_name):
    return 'import/%s:0' % node_name


@beam.ptransform_fn
def _LoadRemoteGraphInputs(pcoll,
                           current_graph_name,
                           remote_graph_name,
                           op_to_remote_op_name_mapping):
    """Load the remote op graph's inputs from the parent graph to the child graph."""
    current_op_type = _get_op_type(current_graph_name, set(op_to_remote_op_name_mapping.keys()))
    placeholder_name_to_input_name = op_to_remote_op_name_mapping[current_op_type][remote_graph_name]
    
    count = 0
    for placeholder_name, input_name in placeholder_name_to_input_name.items():
        count += 1
        pcoll = pcoll | str(count) >> beam.Map(_copy_tensor,
                                               current_graph_name,
                                               _import_tensor_name(input_name),
                                               remote_graph_name,
                                               _import_tensor_name(placeholder_name))
    return pcoll


def _copy_tensor(element, old_graph, old_tensor_name, new_graph, new_tensor_name):
    """Modify element: copy tensor from one graph to another."""
    element = copy.deepcopy(element)
    if new_graph not in element:
        element[new_graph] = {}
    
    element[new_graph][new_tensor_name] = element[old_graph][old_tensor_name]
    return element


@beam.ptransform_fn
def _ExtractRemoteGraphOutput(pcoll,
                             current_graph_name,
                             remote_graph_name,
                             op_to_execution_specs):
    """Extract the remote op graph's output from the child graph to the parent graph."""
    remote_graph_spec = op_to_execution_specs[_get_op_type(remote_graph_name, set(op_to_execution_specs.keys()))]
    remote_graph_output_name = list(remote_graph_spec[-1]['outputs'])[0]
    
    return (pcoll
            # Extract the output from the remote graph
            | beam.Map(_copy_tensor,
                       remote_graph_name,
                       _import_tensor_name(remote_graph_output_name),
                       current_graph_name,
                       _import_tensor_name(remote_graph_name))
            # Remove the intermediate results of the remote graph
            | beam.Map(_remove_finished_graph_info,
                       remote_graph_name))


def _remove_finished_graph_info(element, finished_graph_name):
    """Remove the intermediate results of a remote op graph after the execution completed."""
    element = copy.deepcopy(element)
    del element[finished_graph_name]
    return element