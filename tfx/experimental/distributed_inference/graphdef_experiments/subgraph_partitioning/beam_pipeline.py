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
"""Batch inference with Beam on partitioned subgraphs.

There are two libraries representing two stages: graph_partition and
beam_pipeline. After graph_partition produces lists of ExecutionSpecs that
contain partitioned subgraphs, beam_pipeline arranges, loads, and executes
the partitioned subgraphs. The results from the beam_pipeline should be the
same as the results from the original model.

To understand the limitations of this library, take a look at graph_partition's
module docstring.

  Typical usage example:
  ```
  # Obtained graph_name_to_specs from the graph partition

  parent_graph_to_remote_graph_input_name_mapping = {
      graph_name:
          {remote_graph_name_extended:
              {remote_graph_placeholder_name: parent_graph_input_name}}}

  with beam.Pipeline() as p:
    # Get the input_pcoll
    output = input_pcoll | beam_pipeline.ExecuteOneGraph(
        graph_name_to_specs,
        parent_graph_to_remote_graph_input_name_mapping,
        graph_name_extended)
    # Extract the outputs and store them somewhere.
  ```
"""

import copy
from typing import Dict, Generator, List, Optional, Mapping, Set, Text
import tensorflow as tf
import apache_beam as beam

from tfx.experimental.distributed_inference.graphdef_experiments.subgraph_partitioning import execution_spec


@beam.ptransform_fn
@beam.typehints.with_input_types(Dict[Text, Dict[Text, tf.Tensor]])
@beam.typehints.with_output_types(Dict[Text, Dict[Text, tf.Tensor]])
def ExecuteOneGraph(  # pylint: disable=invalid-name
    pcoll: beam.pvalue.PCollection,
    graph_name_to_specs: Mapping[Text, List[execution_spec.ExecutionSpec]],
    parent_graph_to_remote_graph_input_name_mapping:
        Mapping[Text, Mapping[Text, Mapping[Text, Text]]],
    graph_name_extended: Text) -> beam.pvalue.PCollection:
  """Executes one graph.

  Each graph has a list of ExecutionSpecs, in which the order of the list
  represents the order of execution. An ExecutionSpec can either represent
  a subgraph layer or a remote op in a remote op layer. When executing a
  subgraph layer, we can load and execute the subgraph with a beam Pardo.
  When executing a remote op (which represents another graph), we need to
  load the remote graph inputs, call ExecuteOneGraph to switch to that graph,
  and extract the remote graph output.

  In beam, a PCollection contains a bounded or unbounded number of elements.
  Here, each element is a mapping from graph name extendeds to a mapping from
  tensor names to tensors, or {graph name extended: {tensor name: tensor}}.
  Note that at any time, PColl only stores input tensors and computed tensors.
  The input PColl should have the input tensor names and values for the graph
  ready. As we execute the partitioned subgraphs, we add the intermediate
  output names and values to PColl.

  The main rationale for graph name extended is that it allows us to find the
  exact location of an error. Therefore, we'll use graph name extended by
  default and use graph name as needed.

  Args:
    pcoll: A number of elements, where inputs of the current graph are ready.
    graph_name_to_specs:
      A mapping from graph names to a list of ExecutionSpecs, where the order
      of the list represents the order of execution.

    # We don't have this information since it was stored in PyFunc's function.
    parent_graph_to_remote_graph_input_name_mapping:
      A mapping from graph names to remote graph name extendeds to remote
      graph placeholder names to parent graph input names.
      {graph name: {remote graph extended: {placeholder name: input name}}}.

    graph_name_extended:
      The extended name of the current graph. The difference between
      graph name extended and graph_name is that if there are two or more
      remote ops inside a graph, some of them will have extended name like
      "graph_name_1" to be unique. Graph_name_extendeds sharing the same
      graph name will invoke the same graph, with different inputs.

  Returns:
    A number of elements, where each element stores the inputs, intermediate
    outputs, and final outputs of the graph.
  """
  graph_names = set(graph_name_to_specs.keys())
  graph_name = _get_graph_name(graph_name_extended, graph_names)
  specs = graph_name_to_specs[graph_name]

  for spec in specs:
    # Execute a subgraph layer
    if not spec.is_remote_op:
      step_name = "ExecuteOneSubgraph: graph: {}; output names: {}".format(
          graph_name_extended, ' '.join(spec.output_names))
      pcoll = pcoll | step_name >> beam.ParDo(
          _ExecuteOneSubgraphLayer(), spec, graph_name_extended)

    # Execute a remote op
    else:
      # ExecutionSpec stores one remote op.
      remote_graph_name_extended = list(spec.output_names)[0]
      step_descriptor = "current graph: {}; remote graph: {}".format(
          graph_name_extended, remote_graph_name_extended)

      step_name = "LoadRemoteGraphInputs: %s" % step_descriptor
      pcoll = pcoll | step_name >> _LoadRemoteGraphInputs(
          graph_name_extended,
          remote_graph_name_extended,
          parent_graph_to_remote_graph_input_name_mapping,
          graph_names)

      # A good place to add beam.Reshuffle()
      step_name = "ExecuteOneGraph: %s" % step_descriptor
      pcoll = pcoll | step_name >> ExecuteOneGraph(
          graph_name_to_specs,
          parent_graph_to_remote_graph_input_name_mapping,
          remote_graph_name_extended)

      step_name = "ExtractRemoteGraphOutput: %s" % step_descriptor
      pcoll = pcoll | step_name >> _ExtractRemoteGraphOutput(
          graph_name_extended,
          remote_graph_name_extended,
          graph_name_to_specs,
          graph_names)

  return pcoll


def _get_graph_name(
    graph_name_extended: Text, graph_names: Set[Text]) -> Optional[Text]:
  for graph_name in graph_names:
    if graph_name in graph_name_extended:
      return graph_name
  return None


class _ExecuteOneSubgraphLayer(beam.DoFn):
  """DoFn that executes one subgraph layer."""

  def process(
      self,
      # Not using mapping here because it doesn't support item assignment.
      element: Dict[Text, Dict[Text, tf.Tensor]],
      spec: execution_spec.ExecutionSpec,
      graph_name_extended: Text
  ) -> Generator[Dict[Text, Dict[Text, tf.Tensor]], None, None]:
    """Executes one subgraph layer.

    To execute a subgraph layer, we need to prepare for a feed_dict by
    extracting tensors from element. Then, we run the subgraph and store the
    outputs to element.

    Since we import `GraphDef` protos, all the nodes now associate with a
    prefix "import/". Also, TensorFlow feed_dict and outputs accept tensor
    names instead of node names. Hence, a conversion from node_name to
    "import/node_name:0" is necessary (assumption #2: one output per node).

    Args:
      element: A dictionary from graph name extendeds to a dictionary from
               tensor names to tensors. It stores the graph inputs and all the
               previous specs' outputs for the current graph.
      spec: An ExecutionSpec for a subgraph layer.
      graph_name_extended: As discussed in ExecuteOneGraph, it is the extended
                           name of the current graph.

    Returns:
      A dictionary from graph name extended to a dictionary from tensor names
      to tensors. The outputs of this subgraph layer are added.
    """
    element = copy.deepcopy(element)
    input_tensor_names = [_import_tensor_name(node_name)
                          for node_name in spec.input_names]
    output_tensor_names = [_import_tensor_name(node_name)
                           for node_name in spec.output_names]
    feed_dict = {tensor_name: element[graph_name_extended][tensor_name]
                 for tensor_name in input_tensor_names}

    outputs = []
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
      tf.import_graph_def(spec.subgraph)
      outputs = sess.run(output_tensor_names, feed_dict=feed_dict)

    for output_tensor_name, output_tensor in zip(output_tensor_names, outputs):
      element[graph_name_extended][output_tensor_name] = output_tensor

    yield element


def _import_tensor_name(node_name: Text) -> Text:
  return 'import/%s:0' % node_name


@beam.ptransform_fn
@beam.typehints.with_input_types(Dict[Text, Dict[Text, tf.Tensor]])
@beam.typehints.with_output_types(Dict[Text, Dict[Text, tf.Tensor]])
def _LoadRemoteGraphInputs(  # pylint: disable=invalid-name
    pcoll: beam.pvalue.PCollection,
    parent_graph_name_extended: Text,
    remote_graph_name_extended: Text,
    parent_graph_to_remote_graph_input_name_mapping:
        Mapping[Text, Mapping[Text, Mapping[Text, Text]]],
    graph_names: Set[Text]) -> beam.pvalue.PCollection:
  """Loads remote graph inputs.

  Before executing a remote graph, we need to prepare for its inputs. We first
  get the mapping from remote graph placeholder names to parent graph input
  names. Then, we copy the inputs from the parent graph to the remote graph
  in element.

  Args:
    pcoll: A number of elements, where remote graph inputs aren't seted up.
    parent_graph_name_extended: The extended name of the parent graph.
    remote_graph_name_extended: The extended name of the remote graph.
    parent_graph_to_remote_graph_input_name_mapping:
      A mapping from graph names to remote graph name extendeds to remote
      graph placeholder names to parent graph input names.
      {graph name: {remote graph extended: {placeholder name: input name}}}.
    graph_names: A set of all the graph names.

  Returns:
    A number of elements, where remote graph's inputs are seted up.
  """
  parent_graph_name = _get_graph_name(parent_graph_name_extended, graph_names)
  name_mapping = (parent_graph_to_remote_graph_input_name_mapping
                  [parent_graph_name][remote_graph_name_extended])

  mapping = name_mapping.items()
  # Calling _copy_tensor multiple times may introduce a burden, since
  # _copy_tensor invokes a deepcopy on element.
  for remote_graph_placeholder_name, parent_graph_input_name in mapping:

    step_name = "Prepare input %s in %s" % (
        remote_graph_placeholder_name, remote_graph_name_extended)
    pcoll = pcoll | step_name >> beam.Map(
        _copy_tensor,
        parent_graph_name_extended,
        _import_tensor_name(parent_graph_input_name),
        remote_graph_name_extended,
        _import_tensor_name(remote_graph_placeholder_name))

  return pcoll


def _copy_tensor(
    element: Dict[Text, Dict[Text, tf.Tensor]],
    old_graph: Text, old_tensor_name: Text,
    new_graph: Text, new_tensor_name: Text
) -> Dict[Text, Dict[Text, tf.Tensor]]:
  element = copy.deepcopy(element)
  if new_graph not in element:
    element[new_graph] = {}
  element[new_graph][new_tensor_name] = element[old_graph][old_tensor_name]
  return element


@beam.ptransform_fn
@beam.typehints.with_input_types(Dict[Text, Dict[Text, tf.Tensor]])
@beam.typehints.with_output_types(Dict[Text, Dict[Text, tf.Tensor]])
def _ExtractRemoteGraphOutput(  # pylint: disable=invalid-name
    pcoll: beam.pvalue.PCollection,
    parent_graph_name_extended: Text,
    remote_graph_name_extended: Text,
    graph_name_to_specs: Mapping[Text, List[execution_spec.ExecutionSpec]],
    graph_names: Set[Text]) -> beam.pvalue.PCollection:
  """Extracts remote graph output.

  After finish executing a remote graph, we need to collect its output.
  We first find the output name of the remote graph, then we copy the
  output of the remote graph to its parent graph. Finally, we clear the
  intermediate outputs of the remote graph.

  Note we assumed that each node has only one output, which also applies
  to remote op. This means that a remote graph can only have one output.

  Args:
    pcoll: A number of elements. Since the remote graph finished execution,
           its inputs, intermediate outputs, and final output are all ready.
    parent_graph_name_extended: The extended name of the parent graph.
    remote_graph_name_extended: The extended name of the remote graph.
    graph_name_to_specs: A mapping from graph names to a list of
                         ExecutionSpecs.
    graph_names: A set of all the graph names.

  Returns:
    A number of elements, where remote graph's output is extracted and
    remote graph's intermediate outputs are cleared.
  """
  remote_graph_name = _get_graph_name(remote_graph_name_extended, graph_names)
  remote_graph_specs = graph_name_to_specs[remote_graph_name]
  remote_graph_output_name = list(remote_graph_specs[-1].output_names)[0]

  step_name_extract = "Get output of %s in %s" % (remote_graph_name_extended,
                                                  parent_graph_name_extended)
  step_name_clear = ("Clear intermediate outputs for %s" %
                     (remote_graph_name_extended))

  return (pcoll
          | step_name_extract >> beam.Map(
              _copy_tensor,
              remote_graph_name_extended,
              _import_tensor_name(remote_graph_output_name),
              parent_graph_name_extended,
              _import_tensor_name(remote_graph_name_extended))
          | step_name_clear >> beam.Map(
              _clear_outputs_for_finished_graph,
              remote_graph_name_extended))


def _clear_outputs_for_finished_graph(
    element: Dict[Text, Dict[Text, tf.Tensor]],
    finished_graph: Text) -> Dict[Text, Dict[Text, tf.Tensor]]:
  element = copy.deepcopy(element)
  del element[finished_graph]
  return element
