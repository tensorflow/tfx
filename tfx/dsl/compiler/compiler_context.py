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

import collections

from typing import Optional, List, Set, Iterable

from tfx.dsl.compiler import compiler_utils
from tfx.dsl.components.base import base_node
from tfx.dsl.experimental.conditionals import conditional
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
               parent_pipelines: Optional[List[pipeline.Pipeline]] = None,
               parent_pipeline_context: Optional['PipelineContext'] = None):
    self.pipeline = tfx_pipeline

    self.pipeline_info = self.pipeline.pipeline_info
    self.execution_mode = compiler_utils.resolve_execution_mode(self.pipeline)
    self.dsl_context_registry = self.pipeline.dsl_context_registry
    self.parent_pipelines = parent_pipelines or []
    self.parent_pipeline_context = parent_pipeline_context

    # Stores channels available in the current pipeline scope.
    # Mapping from Channel object to compiled Channel proto.
    self.channels = dict()

    self._pipeline_nodes_by_id = {}
    self._topological_order = {}
    self._implicit_upstream_nodes = collections.defaultdict(set)
    self._implicit_downstream_nodes = collections.defaultdict(set)
    for i, node in enumerate(tfx_pipeline.components, start=1):
      self._pipeline_nodes_by_id[node.id] = node
      self._topological_order[node.id] = i
      self._collect_conditional_dependency(node)

  def _add_implicit_dependency(self, parent_id: str, child_id: str) -> None:
    self._implicit_upstream_nodes[child_id].add(parent_id)
    self._implicit_downstream_nodes[parent_id].add(child_id)

  def _collect_conditional_dependency(self, here: base_node.BaseNode) -> None:
    for predicate in conditional.get_predicates(here,
                                                self.dsl_context_registry):
      for chnl in predicate.dependent_channels():
        if isinstance(chnl, channel_types.OutputChannel):
          self._add_implicit_dependency(chnl.producer_component_id, here.id)

  def topologically_sorted(self, tfx_nodes: Iterable[base_node.BaseNode]):
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
