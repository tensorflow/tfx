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
"""Module for compiler context classes definition."""

from __future__ import annotations

import collections
from typing import Optional, List, Set, Iterable, Dict

from tfx.dsl.compiler import compiler_utils
from tfx.dsl.components.base import base_node
from tfx.dsl.context_managers import dsl_context_registry
from tfx.dsl.control_flow import for_each_internal
from tfx.dsl.experimental.conditionals import conditional
from tfx.dsl.input_resolution import resolver_op
from tfx.orchestration import pipeline
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import channel as channel_types


class PipelineContext:
  """Encapsulates resources needed to compile a pipeline.

  `PipelineContext` is for compiling a single pipeline. If the pipeline is
  compiling a sub-pipeline (a `Pipeline` node as a `pipeline.components`) then
  another `PipelineContext` would be created and used.

  Use it as a first positional argument to the compiler extension functions as
  if you are extending a class and this is a self object.
  """

  def __init__(self,
               tfx_pipeline: pipeline.Pipeline,
               parent: Optional[PipelineContext] = None):
    self.pipeline = tfx_pipeline

    self.pipeline_info = self.pipeline.pipeline_info
    self.execution_mode = compiler_utils.resolve_execution_mode(self.pipeline)
    self.dsl_context_registry = self.pipeline.dsl_context_registry
    self.parent = parent or NullPipelineContext(child=self)

    # Stores channels available in the current pipeline scope.
    # Mapping from Channel object to compiled Channel proto.
    self.channels = dict()

    # Node ID -> NodeContext
    self._node_contexts: Dict[str, NodeContext] = {}

    # CondContext -> Context ID
    self._conditional_ids: Dict[conditional.CondContext, str] = {}

    self._pipeline_nodes_by_id = {}
    self._topological_order = {}
    self._implicit_upstream_nodes = collections.defaultdict(set)
    self._implicit_downstream_nodes = collections.defaultdict(set)
    for i, node in enumerate(tfx_pipeline.components, start=1):
      self._pipeline_nodes_by_id[node.id] = node
      self._topological_order[node.id] = i
      self._collect_conditional_dependency(node)

  @property
  def is_root(self) -> bool:
    return isinstance(self.parent, NullPipelineContext)

  @property
  def is_subpipeline(self) -> bool:
    return self.parent and self.parent.pipeline is not None

  @property
  def parent_pipelines(self) -> List[pipeline.Pipeline]:
    """Parent pipelines, in the order of outer -> inner."""
    result = []
    ctx = self.parent
    while ctx and ctx.pipeline:
      result.append(ctx.pipeline)
      ctx = ctx.parent
    return result[::-1]

  def _add_implicit_dependency(self, parent_id: str, child_id: str) -> None:
    self._implicit_upstream_nodes[child_id].add(parent_id)
    self._implicit_downstream_nodes[parent_id].add(child_id)

  def _collect_conditional_dependency(self, here: base_node.BaseNode) -> None:
    for predicate in conditional.get_predicates(here,
                                                self.dsl_context_registry):
      for chnl in predicate.dependent_channels():
        if isinstance(chnl, channel_types.OutputChannel):
          self._add_implicit_dependency(chnl.producer_component_id, here.id)

  def topologically_sorted(
      self, tfx_nodes: Iterable[base_node.BaseNode],
  ) -> List[base_node.BaseNode]:
    return sorted(tfx_nodes, key=lambda node: self._topological_order[node.id])

  @property
  def is_sync_mode(self):
    return self.execution_mode == pipeline_pb2.Pipeline.SYNC

  @property
  def is_async_mode(self):
    return self.execution_mode == pipeline_pb2.Pipeline.ASYNC

  @property
  def pipeline_node_ids(self) -> Set[str]:
    return set(self._pipeline_nodes_by_id.keys())

  def implicit_upstream_nodes(
      self, here: base_node.BaseNode) -> List[base_node.BaseNode]:
    return [
        self._pipeline_nodes_by_id[node_id]
        for node_id in self._implicit_upstream_nodes[here.id]
    ]

  def implicit_downstream_nodes(
      self, here: base_node.BaseNode) -> List[base_node.BaseNode]:
    return [
        self._pipeline_nodes_by_id[node_id]
        for node_id in self._implicit_downstream_nodes[here.id]
    ]

  def get_node_context(self, tfx_node: base_node.BaseNode) -> NodeContext:
    if tfx_node.id not in self._node_contexts:
      self._node_contexts[tfx_node.id] = NodeContext(tfx_node)
    return self._node_contexts[tfx_node.id]

  def get_conditional_id(self, cond_context: conditional.CondContext) -> str:
    if cond_context not in self._conditional_ids:
      cond_id = f'cond_{len(self._conditional_ids) + 1}'
      self._conditional_ids[cond_context] = cond_id
    return self._conditional_ids[cond_context]


class NullPipelineContext(PipelineContext):
  """PipelineContext without pipeline.

  NullPipelineContext is the parent PipelineContext for the root level pipeline.

  NullPipelineContext is used when compiling a pipeline begin node of the
  root level pipeline, since pipeline begin node is compiled as if compiling a
  pipeline as a node from the parent pipeline, which does not exist for the
  root level pipeline.
  """

  def __init__(self, *, child: PipelineContext):
    # pylint: disable=super-init-not-called
    # Doesn't invoke super().__init__() as it is not a typical PipelineContext
    # with a pipeline.
    self.pipeline = None
    self.pipeline_info = None
    self.dsl_context_registry = dsl_context_registry.DslContextRegistry()
    self.dsl_context_registry.put_node(child.pipeline)
    self.parent = None
    self.channels = {}
    self._implicit_downstream_nodes = collections.defaultdict(set)

    # Node ID -> NodeContext
    self._node_contexts: Dict[str, NodeContext] = {}


class NodeContext:
  """Per-node context for pipeline compilation."""

  def __init__(self, tfx_node: base_node.BaseNode):
    self._input_key_by_channel = {
        channel: input_key
        for input_key, channel in tfx_node.inputs.items()
    }
    self._input_graph_key_by_resolver_fn = {}

  def get_input_key(self, channel: channel_types.BaseChannel):
    """Get `NodeInput.inputs` key for the given `BaseChannel`.

    Each BaseChannel that is either directly or indirectly used as an input to
    the component is encoded as an InputSpec (1:1 relationship). This method
    gets a corresponding input_key for the channel if it has already been looked
    up, else generates a new input key if the channel has not been used for the
    node yet.

    Arguments:
      channel: A BaseChannel that is used either directly or indirectly for the
          input of the node.

    Returns:
      A fixed input key for NodeInputs.inputs that corresponds to the channel
      (i.e. the same channel argument would give the same input key).
    """
    if channel not in self._input_key_by_channel:
      try:
        input_key = compiler_utils.implicit_channel_key(channel)
      except ValueError:
        input_key = '_'.join([
            '_generated',
            channel.type_name.lower(),
            str(len(self._input_key_by_channel) + 1)])
      self._input_key_by_channel[channel] = input_key
    return self._input_key_by_channel[channel]

  def get_input_graph_key(
      self, traced_output: resolver_op.Node,
      for_each_context: Optional[for_each_internal.ForEachContext],
  ):
    """Get NodeInputs.input_graphs key for the given resolver function output.

    Arguments:
      traced_output: A resolver function traced result which is stored in
          `ResolvedChannels.output_node`.
      for_each_context: An optional ForEachContext that the resolver function
          output is bound to, Which is stored in
          `ResolvedChannels.for_each_context`.

    Returns:
      A fixed input key for NodeInputs.input_graphs.
    """
    key = traced_output, for_each_context
    if key not in self._input_graph_key_by_resolver_fn:
      self._input_graph_key_by_resolver_fn[key] = (
          f'graph_{len(self._input_graph_key_by_resolver_fn) + 1}')
    return self._input_graph_key_by_resolver_fn[key]
