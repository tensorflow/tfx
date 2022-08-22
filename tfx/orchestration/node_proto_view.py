# Copyright 2022 Google LLC. All Rights Reserved.
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
"""Helps read a node or composable pipeline's info from proto representation."""

import abc
from typing import Sequence, Union

from tfx.dsl.compiler import constants as compiler_constants
from tfx.proto.orchestration import pipeline_pb2

from google.protobuf import message


class NodeProtoView(abc.ABC):
  """Helps read a node or composable pipeline's info from proto representation.

  This adapter class uses duck typing to mimic the fields of a PipelineNode
  proto, for both Node and Composable Pipeline.
  """

  @property
  @abc.abstractmethod
  def node_info(self) -> pipeline_pb2.NodeInfo:
    pass

  @property
  @abc.abstractmethod
  def contexts(self) -> pipeline_pb2.NodeContexts:
    pass

  @property
  @abc.abstractmethod
  def inputs(self) -> pipeline_pb2.NodeInputs:
    pass

  @property
  @abc.abstractmethod
  def outputs(self) -> pipeline_pb2.NodeOutputs:
    pass

  @property
  @abc.abstractmethod
  def parameters(self) -> pipeline_pb2.NodeParameters:
    pass

  @property
  @abc.abstractmethod
  def upstream_nodes(self) -> Sequence[str]:
    pass

  @property
  @abc.abstractmethod
  def downstream_nodes(self) -> Sequence[str]:
    pass

  @property
  @abc.abstractmethod
  def execution_options(self) -> pipeline_pb2.NodeExecutionOptions:
    pass

  @abc.abstractmethod
  def HasField(self, field_name: str) -> bool:  # pylint: disable=invalid-name
    pass

  @abc.abstractmethod
  def raw_proto(
      self) -> Union[pipeline_pb2.PipelineNode, pipeline_pb2.Pipeline]:
    pass

  def __eq__(self, other: 'NodeProtoView') -> bool:
    """Two views are equal only if the underlying protos are equal."""
    return self.raw_proto() == other.raw_proto()


class PipelineNodeProtoView(NodeProtoView):
  """Reads a normal node's info from its proto representation."""

  def __init__(self, node: pipeline_pb2.PipelineNode):
    self._node = node

  @property
  def node_info(self) -> pipeline_pb2.NodeInfo:
    return self._node.node_info

  @property
  def contexts(self) -> pipeline_pb2.NodeContexts:
    return self._node.contexts

  @property
  def inputs(self) -> pipeline_pb2.NodeInputs:
    return self._node.inputs

  @property
  def outputs(self) -> pipeline_pb2.NodeOutputs:
    return self._node.outputs

  @property
  def parameters(self) -> pipeline_pb2.NodeParameters:
    return self._node.parameters

  @property
  def upstream_nodes(self) -> Sequence[str]:
    return self._node.upstream_nodes

  @property
  def downstream_nodes(self) -> Sequence[str]:
    return self._node.downstream_nodes

  @property
  def execution_options(self) -> pipeline_pb2.NodeExecutionOptions:
    return self._node.execution_options

  def HasField(self, field_name: str) -> bool:  # pylint: disable=invalid-name
    return self._node.HasField(field_name)

  def raw_proto(self) -> pipeline_pb2.PipelineNode:
    return self._node


class ComposablePipelineProtoView(NodeProtoView):
  """Reads a composable pipeline's info from its proto representation."""

  def __init__(self, pipeline: pipeline_pb2.Pipeline):
    self._pipeline = pipeline
    assert len(self._pipeline.nodes) >= 2, (
        'Got malformed composable pipeline proto. Expecting at least two nodes '
        '(PipelineBegin node and PipelineEnd node) in the composable pipeline.')

    self._begin_node = self._pipeline.nodes[0].pipeline_node
    assert self._begin_node.node_info.type.name.endswith(
        compiler_constants.PIPELINE_BEGIN_NODE_SUFFIX
    ), ('Got malformed composable pipeline proto. Expecting the first node in '
        'composable pipeline to be a PipelineBegin node.'
       )

    self._end_node = self._pipeline.nodes[-1].pipeline_node
    assert self._end_node.node_info.type.name.endswith(
        compiler_constants.PIPELINE_END_NODE_SUFFIX
    ), ('Got malformed composable pipeline proto. Expecting the last node in '
        'composable pipeline to be a PipelineEnd node.'
       )

    self._node_info = None
    self._contexts = None

  def _strip_begin_node_suffix(self, s: str) -> str:
    if not s.endswith(compiler_constants.PIPELINE_BEGIN_NODE_SUFFIX):
      raise ValueError(
          'Got malformed composable pipeline proto.'
          'Expecting the first node in composable pipeline to be a '
          'PipelineBegin node.'
      )
    return s[:-len(compiler_constants.PIPELINE_BEGIN_NODE_SUFFIX)]

  @property
  def node_info(self) -> pipeline_pb2.NodeInfo:
    # We can create a fake NodeInfo proto for a composable pipeline,
    # when it is viewed as a node, by striping the `_begin`
    # suffix from PipelineBegin node's NodeInfo.
    if not self._node_info:
      self._node_info = pipeline_pb2.NodeInfo()
      self._node_info.type.name = self._strip_begin_node_suffix(
          self._begin_node.node_info.type.name)
      self._node_info.id = self._strip_begin_node_suffix(
          self._begin_node.node_info.id)
    return self._node_info

  @property
  def contexts(self) -> pipeline_pb2.NodeContexts:
    # A composable pipeline, when viewed as a node, stores contexts in its
    # PipelineBegin node's contexts, except that we need to strip the `_begin`
    # suffix from PipelineBegin node's `node` context.
    if not self._contexts:
      self._contexts = pipeline_pb2.NodeContexts()
      self._contexts.CopyFrom(self._begin_node.contexts)
      for context in self._contexts.contexts:
        if context.type.name == compiler_constants.NODE_CONTEXT_TYPE_NAME:
          context.name.field_value.string_value = (
              self._strip_begin_node_suffix(
                  context.name.field_value.string_value))
    return self._contexts

  @property
  def inputs(self) -> pipeline_pb2.NodeInputs:
    # A composable pipeline, when viewed as a node, stores inputs in its
    # PipelineBegin node's inputs.
    return self._begin_node.inputs

  @property
  def outputs(self) -> pipeline_pb2.NodeOutputs:
    # A composable pipeline, when viewed as a node, stores outputs in its
    # PipelineEnd node's outputs. BUT, here we do NOT return the output dict,
    # to save the orchestrator from preparing an extra output dir for the inner
    # pipeline. Downstream nodes that depend on the outputs of this inner
    # pipeline has been compiled with artifact query that directly reads from
    # the PipelineEnd node's outputs.
    return pipeline_pb2.NodeOutputs()

  @property
  def parameters(self) -> pipeline_pb2.NodeParameters:
    # A composable pipeline, when viewed as a node, stores parameters in its
    # PipelineBegin node's parameters.
    return self._begin_node.parameters

  @property
  def upstream_nodes(self) -> Sequence[str]:
    # A composable pipeline, when viewed as a node, stores upstream nodes in its
    # PipelineBegin node's upstream nodes.
    return self._begin_node.upstream_nodes

  @property
  def downstream_nodes(self) -> Sequence[str]:
    # A composable pipeline, when viewed as a node, stores downstream nodes in
    # its PipelineEnd node's downstream nodes.
    return self._end_node.downstream_nodes

  @property
  def execution_options(self) -> pipeline_pb2.NodeExecutionOptions:
    # A composable pipeline, when viewed as a node, stores execution options in
    # its PipelineBegin node's execution options.
    return self._begin_node.execution_options

  def HasField(self, field_name: str) -> bool:  # pylint: disable=invalid-name
    # Despite the proto method is named `HasField`, it actually means is set or
    # is non-empty.
    return_value = getattr(self, field_name)
    if isinstance(return_value, message.Message):
      # Check if a proto message is empty.
      return return_value.ByteSize() == 0
    # For other cases, e.g. a repeated field, we convert it to bool.
    return bool(return_value)

  def raw_proto(self) -> pipeline_pb2.Pipeline:
    return self._pipeline


def get_view(
    pipeline_or_node: Union[pipeline_pb2.Pipeline.PipelineOrNode,
                            pipeline_pb2.PipelineNode, pipeline_pb2.Pipeline]
) -> NodeProtoView:
  """Builds a NodeProtoView adapter from either a Node or a Pipeline proto."""
  if isinstance(pipeline_or_node, pipeline_pb2.Pipeline.PipelineOrNode):
    which = pipeline_or_node.WhichOneof('node')
    if which == 'pipeline_node':
      return PipelineNodeProtoView(pipeline_or_node.pipeline_node)
    elif which == 'sub_pipeline':
      return ComposablePipelineProtoView(pipeline_or_node.sub_pipeline)
    else:
      raise ValueError('Got unknown pipeline or node type.')

  if isinstance(pipeline_or_node, pipeline_pb2.PipelineNode):
    return PipelineNodeProtoView(pipeline_or_node)

  if isinstance(pipeline_or_node, pipeline_pb2.Pipeline):
    return ComposablePipelineProtoView(pipeline_or_node)

  raise ValueError('Got unknown pipeline or node type.')
