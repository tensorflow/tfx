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
"""Graph partitioning on TensorFlow SavedModels with remote ops.

The current implementation invokes the subgraph partitioning algorithm.

There are two libraries representing two stages: graph_partition and
beam_pipeline. In this library, we take in some SavedModel inputs, partition
the graphs, and store the partitioned subgraphs into lists of SavedModel
directory paths. The order of the list represents the order of execution.
Take a look at beam_pipeline if you want to run the partitioned subgraphs.

The current implementation has some key limitations:
  1. All the node/op should have one output.
  2. This library doesn't support tf.variable.
  3. This library doesn't support tf.function.

  Typical usage example:
  ```
  graph_name_to_unpartitioned_path = {
      graph_1_name: savedmodel_directory_path_1,
      graph_2_name: savedmodel_directory_path_2
  }

  graph_name_to_graph_def, graph_name_to_output_names = get_info_from_paths(
      graph_name_to_unpartitioned_path)
  graph_name_to_partitioned_paths = partition_all_graphs(
      graph_to_graph_def,
      graph_to_output_names)

  # Followed by the beam pipeline.
  ```
"""

import os
from typing import Dict, List, Mapping, Set, Text, Tuple
import tensorflow as tf

from tfx.experimental.distributed_inference.graphdef_experiments.subgraph_partitioning import execution_spec
from tfx.experimental.distributed_inference.graphdef_experiments.subgraph_partitioning import graph_partition


def get_info_from_paths(
    graph_name_to_unpartitioned_paths: Mapping[Text, Text]
) -> Tuple[Dict[Text, tf.compat.v1.GraphDef],
           Dict[Text, List[Text]]]:
  """Gets the `GraphDef` protos and graph output names from paths.

  Args:
    graph_name_to_unpartitioned_paths:
      A mapping from graph names to `SavedModel` directory paths. These
      SavedModels haven't been partitioned yet.

  Returns:
    A tuple consists of two mappings. The first is a mapping from graph names
    to `GraphDef` protos. The second is a mapping from graph names to lists of
    their graph output names.
  """
  graph_name_to_saved_model = {
      graph_name: _get_saved_model(path)
      for graph_name, path in graph_name_to_unpartitioned_paths.items()
  }
  graph_name_to_graph_def = {
      graph_name: _get_graph_def(saved_model)
      for graph_name, saved_model in graph_name_to_saved_model.items()
  }
  graph_name_to_output_names = {
      graph_name: _get_output_names(saved_model)
      for graph_name, saved_model in graph_name_to_saved_model.items()
  }
  return graph_name_to_graph_def, graph_name_to_output_names


def _get_saved_model(
    directory_path: Text
) -> tf.core.protobuf.saved_model_pb2.SavedModel:
  file_path = os.path.join(directory_path, 'saved_model.pb')
  saved_model = tf.core.protobuf.saved_model_pb2.SavedModel()
  with tf.io.gfile.GFile(file_path, 'rb') as f:
    saved_model.ParseFromString(f.read())
  return saved_model


def _get_graph_def(
    saved_model: tf.core.protobuf.saved_model_pb2.SavedModel
) -> tf.compat.v1.GraphDef:
  return saved_model.meta_graphs[0].graph_def


def _get_output_names(
    saved_model: tf.core.protobuf.saved_model_pb2.SavedModel
) -> List[Text]:
  meta_graph = saved_model.meta_graphs[0]
  outputs = meta_graph.signature_def['serving_default'].outputs

  # Remove the tensor name postfix ":0".
  output_names = [output.name[:-2] for output in dict(outputs).values()]
  return output_names


def partition_all_graphs(
    graph_name_to_graph_def: Mapping[Text, tf.compat.v1.GraphDef],
    graph_name_to_output_names: Mapping[Text, List[Text]],
    export_dir: Text) -> Dict[Text, List[Text]]:
  """Partitions all the graphs.

  For each graph, the partitioning algorithm takes in the graph's `GraphDef`
  proto and output names, partitions the graph, stores the partitioned
  subgraphs in SavedModels, and returns a list of SavedModel directory paths.
  Later, the beam_pipeline library can take in the SavedModel directory paths
  and execute the partitioned subgraphs.

  A partitioned subgraph can either represent a subgraph layer (consists of
  nodes other than remote ops) or a remote op in a remote op layer (consists
  of remote ops that don't have dependencies on each other).

  Args:
    graph_name_to_graph_def: A mapping from graph names to `GraphDef` protos.
    graph_name_to_output_names: A mapping from graph names to lists of their
                                output node names.

  Returns:
    A mapping from graph names to lists of SavedModel directory paths, where
    each SavedModel stores a partitioned subgraph. The order of the list
    represents the order of execution.
  """
  graph_name_to_specs = graph_partition.partition_all_graphs(
      graph_name_to_graph_def, graph_name_to_output_names)

  graph_name_to_partitioned_paths = {
      graph_name: _save_as_saved_models(export_dir, graph_name, specs)
      for graph_name, specs in graph_name_to_specs.items()
  }
  return graph_name_to_partitioned_paths


def _save_as_saved_models(
    export_dir: Text, graph_name: Text,
    specs: List[execution_spec.ExecutionSpec]) -> List[Text]:
  """Saves the partitioned subgraphs as SavedModels.

  The partitioned subgraphs were previously stored in ExecutionSpecs. Here
  we store the partitioned subgraphs in SavedModels by identifying the
  `GraphDef` proto, the input signatures, and the output signatures.

  Args:
    export_dir: The directory to save the partitioned subgraphs.
    graph_name: The name of the graph that we partitioned to get the specs.
    specs: A list of ExecutionSpecs, where each spec stores a partitioned
           subgraph. The order of the list represents the order of execution.

  Returns:
    A list of SavedModel directory paths, where each SavedModel stores a
    partitioned subgraph. The order of the list represents the order of
    execution.
  """
  partitioned_paths = []
  for counter, spec in enumerate(specs):
    saved_model_path = _get_partitioned_path(
        export_dir, graph_name, counter, spec.output_names)
    partitioned_paths.append(saved_model_path)

    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
      if not spec.is_remote_op:
        tf.import_graph_def(spec.subgraph)
      else:
        # Simple_save() requires inputs and outputs to be mappings from
        # node names to tensors. For remote op, create a graph_def full of
        # placeholders only to satisfy that requirement.
        tf.import_graph_def(
            _create_graph_def_of_placeholders(
                spec.input_names | spec.output_names))

      inputs = _get_node_name_to_tensor(sess.graph, spec.input_names)
      outputs = _get_node_name_to_tensor(sess.graph, spec.output_names)
      tf.compat.v1.saved_model.simple_save(sess,
                                           saved_model_path,
                                           inputs,
                                           outputs)
  return partitioned_paths


def _get_partitioned_path(
    export_dir: Text, graph_name: Text, counter: int,
    output_names: Set[Text]) -> Text:
  saved_model_path = os.path.join(
      export_dir, 'partitioned_saved_models',
      'graph_name_%s_subgraph_%s_output_names_%s' %
      (graph_name, str(counter), '_'.join(output_names)))
  return saved_model_path


def _create_graph_def_of_placeholders(
    node_names: Set[Text]) -> tf.compat.v1.GraphDef:
  with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    for node_name in node_names:
      tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name=node_name)
    return sess.graph_def


def _get_node_name_to_tensor(
    graph: tf.Graph, node_names: Set[Text]) -> Dict[Text, tf.Tensor]:
  node_name_to_tensor = {name: graph.get_tensor_by_name('import/%s:0' % name)
                         for name in node_names}
  return node_name_to_tensor
