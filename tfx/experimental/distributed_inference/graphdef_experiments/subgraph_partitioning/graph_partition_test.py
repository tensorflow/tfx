"""
A library for graph partitioning testing.

@author: jzhunybj
"""

import graph_partition
import tensorflow as tf        
        
        
class RelationTest(tf.test.TestCase):
    
    def testLayers(self):
        remote_op_relations = {'a1': [], 'a2': [], 'b1': ['a1'], 'b2': ['a1', 'a2'],
                               'c1': ['b1'], 'c2': ['b1', 'a1', 'b2', 'a2']}
        relations = graph_partition.Relations(remote_op_relations)
        
        self.assertFalse(relations.check_if_finished())
        self.assertFalse(relations.check_if_finished())
        
        self.assertEqual(relations.get_next_layer(), {'a1', 'a2'})
        self.assertFalse(relations.check_if_finished())
        self.assertEqual(relations.get_next_layer(), {'b1', 'b2'})
        self.assertFalse(relations.check_if_finished())
        self.assertEqual(relations.get_next_layer(), {'c1', 'c2'})
        
        self.assertTrue(relations.check_if_finished())
        self.assertTrue(relations.check_if_finished())
    
    
op_to_filename = {'main': './complex_graphdefs/main_graph.pb',
                  'remote_op_a': './complex_graphdefs/graph_a.pb',
                  'remote_op_b': './complex_graphdefs/graph_b.pb',
                }
op_to_outputs = {'main': ['AddN_1'],
                 'remote_op_b': ['Add_1'],
                 'remote_op_a': ['embedding_lookup/Identity'],
                }
op_to_graph_def = graph_partition.get_op_to_graph_def(op_to_filename)
op_to_execution_bundles = graph_partition.partition_all_graphs(op_to_graph_def, op_to_outputs)


class PartitionTest(tf.test.TestCase):

    def testOpNames(self):
        op_keys = set(op_to_outputs.keys())
        op_current_keys = set(op_to_execution_bundles.keys())
        self.assertEqual(op_keys, op_current_keys)
    
    
    def testSubgraphImportValidity(self):
        for op, execution_bundles in op_to_execution_bundles.items():
            for execution_bundle in execution_bundles:
                if not execution_bundle['is_remote_op']:
                    graph = tf.Graph()
                    with graph.as_default():
                        tf.import_graph_def(execution_bundle['subgraph'])
    
    
    def _get_node_names_from_subgraph(self, subgraph):
        node_names = set([node.name for node in subgraph.node])
        return node_names
        
                        
    def testSubgraphBundles(self):
        for op, execution_bundles in op_to_execution_bundles.items():
            for bundle in execution_bundles:
                if not bundle['is_remote_op']:
                    all_nodes = self._get_node_names_from_subgraph(bundle['subgraph'])
                    
                    self.assertTrue(bundle['outputs'].issubset(bundle['body_nodes']))
                    self.assertEqual(all_nodes, bundle['body_nodes'].union(bundle['inputs']))
                    
                    for input_name in bundle['inputs']:
                        self.assertNotIn(input_name, bundle['body_nodes'])
                        
                    for node_from_other_layer in bundle['nodes_from_other_layers']:
                        self.assertNotIn(node_from_other_layer, bundle['body_nodes'])
                    

    def testRemoteOpBundles(self):
        for op, execution_bundles in op_to_execution_bundles.items():
            for bundle in execution_bundles:
                if bundle['is_remote_op']:
                    self.assertIsNone(bundle['subgraph'])
                    self.assertLen(bundle['outputs'], 1)
                    self.assertLen(bundle['body_nodes'], 1)
    
    
if __name__ == '__main__':
    tf.test.main()
    
    
    
    