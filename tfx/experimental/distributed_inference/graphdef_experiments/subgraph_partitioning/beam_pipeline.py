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
contain partitioned subgraphs, beam_pipeline constructs a Beam pipeline that
executes the partitioned subgraphs. The results from the Beam pipeline should
be the same as the results from the original model.

  Typical usage example:
  ```
  # Obtained graph_name_to_specs from the graph partition
  remote_op_name_to_graph_name = {
      remote_op_name: graph_name
  }
  graph_to_remote_op_input_name_mapping = {
      graph_name:
          {remote_op_name:
              {remote_graph_placeholder_name: parent_graph_input_name}
          }
  }

  with beam.Pipeline() as p:
    # Get the input_pcoll
    output = input_pcoll | beam_pipeline.ExecuteOneGraph(
        remote_op_name,
        remote_op_name_to_graph_name,
        graph_name_to_specs,
        graph_to_remote_op_input_name_mapping)
    # Extract the outputs and store them somewhere.
  ```
"""

import copy
from typing import Any, Dict, Iterator, List, Mapping, Text
import apache_beam as beam
import tensorflow as tf

from tfx.experimental.distributed_inference.graphdef_experiments.subgraph_partitioning import execution_spec


@beam.ptransform_fn
@beam.typehints.with_input_types(Dict[Text, Dict[Text, Any]])
@beam.typehints.with_output_types(Dict[Text, Dict[Text, Any]])
def ExecuteGraph(  # pylint: disable=invalid-name
    pcoll: beam.pvalue.PCollection, remote_op_name: Text,
    remote_op_name_to_graph_name: Mapping[Text, Text],
    graph_name_to_specs: Mapping[Text, List[execution_spec.ExecutionSpec]],
    graph_to_remote_op_input_name_mapping: Mapping[Text, Mapping[Text,
                                                                 Mapping[Text,
                                                                         Text]]]
) -> beam.pvalue.PCollection:
  """A PTransform that executes a graph.

  Each graph has a list of ExecutionSpecs, in which the order of the list
  represents the order of execution. An ExecutionSpec can either represent
  a subgraph layer or a remote op in a remote op layer. When executing a
  subgraph layer, we can load and execute the subgraph with a beam ParDo.
  When executing a remote op (which represents another graph), we need to
  load the remote graph inputs, call ExecuteGraph to recursively execute that
  graph, and extract the remote graph output. When executing a remote op, we
  call the current graph "parent" and the remote graph "child".

  Here, each Beam element is a dictionary from remote op names to a dictionary
  from tensor names to values, or {remote op name: {tensor name: value}}.
  Note that at any time, PColl only stores input tensor values and computed
  tensor values. The input PColl should have the input tensor names and values
  for the graph ready. As we execute the partitioned subgraphs, we add the
  intermediate output names and values to PColl.

  Args:
    pcoll: A PCollection of inputs to the graph. Each element is a dictionary
      from remote op names to a dictionary from tensor names to values. Here,
      element[remote_op_name] contains graph inputs.
    remote_op_name: The remote op name of the current graph.
    remote_op_name_to_graph_name: A mapping from remote op names to graph names.
    graph_name_to_specs: A mapping from graph names to a list of ExecutionSpecs,
      where the order of the list represents the order of execution.
    graph_to_remote_op_input_name_mapping: A mapping from graph names to remote
      op names to remote graph placeholder names to parent graph input names. We
      don't have this information since it was stored in PyFunc's function.
      {graph name: {remote op name: {placeholder name: input name}}}.

  Returns:
    A PCollection of results of this graph. Each element is a dictionary from
    remote op names to a dictionary from tensor names to values. Here,
    element[remote_op_name] stores graph inputs, intermediate results, and
    graph outputs.
  """
  specs = graph_name_to_specs[remote_op_name_to_graph_name[remote_op_name]]

  for spec in specs:
    # Construct Beam subgraph for a subgraph layer.
    if not spec.is_remote_op:
      step_name = ("SubgraphLayerDoFn[Graph_%s][Outputs_%s]" %
                   (remote_op_name, "_".join(spec.output_names)))
      pcoll = pcoll | step_name >> beam.ParDo(_SubgraphLayerDoFn(), spec,
                                              remote_op_name)

    # Construct Beam subgraph for a remote op.
    else:
      # ExecutionSpec stores one remote op.
      child_remote_op_name = list(spec.output_names)[0]
      step_descriptor = ("[Parent_%s][Child_%s]" %
                         (remote_op_name, child_remote_op_name))

      step_name = "LoadRemoteGraphInputs%s" % step_descriptor
      pcoll = pcoll | step_name >> _LoadRemoteGraphInputs(  # pylint: disable=no-value-for-parameter
          remote_op_name, child_remote_op_name, remote_op_name_to_graph_name,
          graph_to_remote_op_input_name_mapping)

      # A good place to add beam.Reshuffle() to prevent fusion.
      step_name = "ExecuteGraph%s" % step_descriptor
      pcoll = pcoll | step_name >> ExecuteGraph(  # pylint: disable=no-value-for-parameter
          child_remote_op_name, remote_op_name_to_graph_name,
          graph_name_to_specs, graph_to_remote_op_input_name_mapping)

      step_name = "ExtractRemoteGraphOutput%s" % step_descriptor
      pcoll = pcoll | step_name >> _ExtractRemoteGraphOutput(  # pylint: disable=no-value-for-parameter
          remote_op_name, child_remote_op_name, remote_op_name_to_graph_name,
          graph_name_to_specs)

  return pcoll


class _SubgraphLayerDoFn(beam.DoFn):
  """DoFn that executes one subgraph layer."""

  def process(
      self,
      # Not using mapping here because it doesn't support item assignment.
      element: Dict[Text, Dict[Text, Any]],
      spec: execution_spec.ExecutionSpec,
      remote_op_name: Text) -> Iterator[Dict[Text, Dict[Text, Any]]]:
    """Executes a subgraph layer.

    To execute a subgraph layer, we need to prepare a feed_dict by extracting
    tensor values from element. Then, we run the subgraph and store its outputs
    to a copy of element.

    Since we import `GraphDef` protos, all the node names now have the prefix
    "import/". Also, TensorFlow feed_dict and outputs accept tensor
    names instead of node names. Hence, a conversion from node_name to
    "import/node_name:0" is necessary. Note that this conversion assumes
    that there is one output per node.

    Args:
      element: A dictionary from remote op names to a dictionary from tensor
        names to values. Element[remote_op_name] stores graph inputs and
        previous specs' outputs.
      spec: An ExecutionSpec for a subgraph layer.
      remote_op_name: The remote op name of the current graph.

    Yields:
      A dictionary from remote op names to a dictionary from tensor names to
      values. The dictionary is a copy of the input element, to which the
      outputs of this subgraph layer have been added.
    """
    element = copy.deepcopy(element)
    input_tensor_names = [
        _import_tensor_name(node_name) for node_name in spec.input_names
    ]
    output_tensor_names = [
        _import_tensor_name(node_name) for node_name in spec.output_names
    ]
    feed_dict = {
        tensor_name: element[remote_op_name][tensor_name]
        for tensor_name in input_tensor_names
    }

    outputs = []
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
      tf.import_graph_def(spec.subgraph)
      outputs = sess.run(output_tensor_names, feed_dict=feed_dict)

    for output_tensor_name, output_tensor in zip(output_tensor_names, outputs):
      element[remote_op_name][output_tensor_name] = output_tensor

    yield element


def _import_tensor_name(  # pylint: disable=invalid-name
    node_name: Text) -> Text:
  return "import/%s:0" % node_name


@beam.ptransform_fn
@beam.typehints.with_input_types(Dict[Text, Dict[Text, Any]])
@beam.typehints.with_output_types(Dict[Text, Dict[Text, Any]])
def _LoadRemoteGraphInputs(  # pylint: disable=invalid-name
    pcoll: beam.pvalue.PCollection, parent_remote_op_name: Text,
    child_remote_op_name: Text, remote_op_name_to_graph_name: Mapping[Text,
                                                                      Text],
    graph_to_remote_op_input_name_mapping: Mapping[Text, Mapping[Text,
                                                                 Mapping[Text,
                                                                         Text]]]
) -> beam.pvalue.PCollection:
  """A PTransform that prepares inputs for a remote graph.

  Before executing a remote graph, we need to prepare its inputs. We first
  get the mapping from remote graph placeholder names to parent graph input
  names. Then, in a copy of element, we copy the inputs from the parent
  graph's key to the remote graph's key.

  Args:
    pcoll: A PCollection of child graph inputs not loaded yet. Each element is a
      dictionary from remote op names to a dictionary from tensor names to
      values. Here, element[child_remote_op_name] is empty now.
    parent_remote_op_name: The remote op name of the parent graph.
    child_remote_op_name: The remote op name of the child graph.
    remote_op_name_to_graph_name: A mapping from remote op names to graph names.
    graph_to_remote_op_input_name_mapping: A mapping from graph names to remote
      op names to remote graph placeholder names to parent graph input names.
      {graph name: {remote op name: {placeholder name: input name}}}.

  Returns:
    A PCollection of inputs to the child graph. Each element is a dictionary
    from remote op names to a dictionary from tensor names to values. Here,
    element[child_remote_op_name] stores the inputs of child graph.
  """
  parent_graph_name = remote_op_name_to_graph_name[parent_remote_op_name]
  name_mapping = (
      graph_to_remote_op_input_name_mapping[parent_graph_name]
      [child_remote_op_name])

  mapping = name_mapping.items()
  # Calling _copy_tensor_value multiple times may introduce a burden, since
  # _copy_tensor_value invokes a deepcopy on element.
  for child_graph_placeholder_name, parent_graph_input_name in mapping:

    step_name = ("PrepareInput[Graph_%s][Input_%s]" %
                 (child_remote_op_name, child_graph_placeholder_name))
    pcoll = pcoll | step_name >> beam.Map(
        _copy_tensor_value, parent_remote_op_name,
        _import_tensor_name(parent_graph_input_name), child_remote_op_name,
        _import_tensor_name(child_graph_placeholder_name))

  return pcoll


def _copy_tensor_value(  # pylint: disable=invalid-name
    element: Dict[Text, Dict[Text,
                             Any]], old_graph: Text, old_tensor_name: Text,
    new_graph: Text, new_tensor_name: Text) -> Dict[Text, Dict[Text, Any]]:
  element = copy.deepcopy(element)
  if new_graph not in element:
    element[new_graph] = {}
  element[new_graph][new_tensor_name] = element[old_graph][old_tensor_name]
  return element


@beam.ptransform_fn
@beam.typehints.with_input_types(Dict[Text, Dict[Text, Any]])
@beam.typehints.with_output_types(Dict[Text, Dict[Text, Any]])
def _ExtractRemoteGraphOutput(  # pylint: disable=invalid-name
    pcoll: beam.pvalue.PCollection,
    parent_remote_op_name: Text,
    child_remote_op_name: Text,
    remote_op_name_to_graph_name: Mapping[Text, Text],
    graph_name_to_specs: Mapping[Text, List[execution_spec.ExecutionSpec]],
) -> beam.pvalue.PCollection:
  """A PTransform that extracts remote graph output.

  After finish executing a remote graph, we need to collect its output.
  We first find the output name of the remote graph, then we copy the
  output of the remote graph to its parent graph. Finally, we clear the
  intermediate results of the remote graph.

  Note we assumed that each node has only one output, which also applies
  to remote op. This means that a remote graph can only have one output.

  Args:
    pcoll: A PCollection of child graph results. Each element is a dictionary
      from remote op names to a dictionary from tensor names to values. Here,
      element[child_remote_op_name] stores graph inputs, intermediate results,
      and graph output.
    parent_remote_op_name: The remote op name of the parent graph.
    child_remote_op_name: The remote op name of the child graph.
    remote_op_name_to_graph_name: A mapping from remote op names to graph names.
    graph_name_to_specs: A mapping from graph names to a list of ExecutionSpecs.

  Returns:
    A PCollection of child graph output in parent graph. Each element is a
    dictionary from remote op names to a dictionary from tensor names to
    values. Here, element[parent_remote_op_name] contains the output from
    the child graph, and element[child_remote_op_name] is deleted.
  """
  child_graph_name = remote_op_name_to_graph_name[child_remote_op_name]
  child_specs = graph_name_to_specs[child_graph_name]
  child_output_name = list(child_specs[-1].output_names)[0]

  step_name_extract = ("ExtractOutput[Graph_%s][Output_%s]" %
                       (child_remote_op_name, child_output_name))
  step_name_clear = ("ClearIntermediateOutputs[Graph_%s]" %
                     (child_remote_op_name))

  return (pcoll
          | step_name_extract >> beam.Map(
              _copy_tensor_value, child_remote_op_name,
              _import_tensor_name(child_output_name), parent_remote_op_name,
              _import_tensor_name(child_remote_op_name))
          | step_name_clear >> beam.Map(_clear_outputs_for_finished_graph,
                                        child_remote_op_name))


def _clear_outputs_for_finished_graph(  # pylint: disable=invalid-name
    element: Dict[Text, Dict[Text, Any]],
    finished_graph: Text) -> Dict[Text, Dict[Text, Any]]:
  element = copy.deepcopy(element)
  del element[finished_graph]
  return element
