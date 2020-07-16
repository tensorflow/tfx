"""
A library for beam pipeline testing.

@author: jzhunybj
"""

import graph_partition
import beam_pipeline

import tensorflow as tf
import apache_beam as beam
from apache_beam.testing import test_pipeline
from apache_beam.testing import util
import numpy as np

# Since we're using PyFunc to mimic the behavior of a remote op, we need
# to import it so that we can run the original model. If we don't, 
# the PyFunc ops cannot be loaded into a TF graph.
import create_complex_graph


# Some info
op_to_filename = {'main': './complex_graphdefs/main_graph.pb',
                  'remote_op_a': './complex_graphdefs/graph_a.pb',
                  'remote_op_b': './complex_graphdefs/graph_b.pb',
                  }
op_to_outputs = {'main': ['AddN_1'],
                'remote_op_b': ['Add_1'],
                'remote_op_a': ['embedding_lookup/Identity'],
                }

op_to_remote_op_name_mapping = {'main': {'remote_op_a': {'ids_a': 'ids1'},
                                         'remote_op_b': {'ids_b1': 'ids1', 'ids_b2': 'ids2'},
                                         'remote_op_a_1': {'ids_a': 'FloorMod_1'},
                                         'remote_op_b_1': {'ids_b1': 'FloorMod_1', 'ids_b2': 'FloorMod'}},
                                'remote_op_b': {'remote_op_a': {'ids_a': 'FloorMod'},
                                                'remote_op_a_1': {'ids_a': 'ids_b2'}},
                                }
                                
feed_dicts_main_graph = [{'main': {'import/ids1:0': 3, 'import/ids2:0': 3}},
                         {'main': {'import/ids1:0': 10, 'import/ids2:0': 10}}]
feed_dicts_graph_b = [{'remote_op_b': {'import/ids_b1:0': 3, 'import/ids_b2:0': 3}},
                      {'remote_op_b': {'import/ids_b1:0': 10, 'import/ids_b2:0': 10}}]
feed_dicts_graph_a = [{'remote_op_a': {'import/ids_a:0': 3}},
                      {'remote_op_a': {'import/ids_a:0': 10}}]
op_to_feed_dicts = {'main': feed_dicts_main_graph,
                    'remote_op_b': feed_dicts_graph_b,
                    'remote_op_a': feed_dicts_graph_a}


class RunnerTest(tf.test.TestCase):
    
    def testResults(self):
        graph_name = 'main'
        
        result_original_model = _run_original_model(graph_name,
                                                    op_to_filename,
                                                    op_to_outputs,
                                                    op_to_feed_dicts)
        
        op_to_graph_def = graph_partition.get_op_to_graph_def(op_to_filename)
        op_to_execution_bundles = graph_partition.partition_all_graphs(op_to_graph_def, op_to_outputs)

        with test_pipeline.TestPipeline() as p:
            
            input_pcoll = p | 'LoadData' >> beam.Create(op_to_feed_dicts[graph_name])
            
            output = (input_pcoll 
                      | 'RunModel' >> beam_pipeline.ExecuteOneGraph(op_to_execution_bundles,
                                                                    op_to_remote_op_name_mapping,
                                                                    graph_name)
                      | 'ExtractOutput' >> beam.Map(_extract_outputs,
                                                    op_to_outputs,
                                                    graph_name))
            
            # Problem: Doesn't work with more complex things like tensors.
            util.assert_that(output, util.equal_to(result_original_model))
            p.run()
            

def _extract_outputs(element, op_to_outputs, graph_name):
    """Extract the outputs within output_names"""
    outputs = [element[graph_name][beam_pipeline._import_tensor_name(output_name)]
               for output_name in op_to_outputs[graph_name]]
    return outputs


def _run_original_model(graph_name, 
                        op_to_filename, 
                        op_to_outputs, 
                        op_to_feed_dicts):
    """Run the original TF model."""
    op_to_graph_def = graph_partition.get_op_to_graph_def(op_to_filename)
    graph_def = op_to_graph_def[graph_name]
    graph = graph_partition._get_graph(graph_def)
    
    output_names = [beam_pipeline._import_tensor_name(output_name)
                     for output_name in op_to_outputs[graph_name]]
    feed_dicts = op_to_feed_dicts[graph_name]
    
    results = []
    with tf.compat.v1.Session(graph=graph) as sess:
        for feed_dict in feed_dicts:
            results.append(sess.run(output_names, feed_dict[graph_name]))
    
    return results
    

if __name__ == '__main__':
    tf.test.main()



