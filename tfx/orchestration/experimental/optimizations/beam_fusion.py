# Lint as: python2, python3
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
"""TFX BeamFusionOptimizer definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import List, Set, Mapping

from tfx.components.base import base_node
from tfx.components.base import base_executor
from tfx.orchestration.pipeline import Pipeline
from fused_component.component import FusedComponent

class BeamFusionOptimizer(object):
  """Optimizer for TFX pipelines, utilizing Beam Fusion"""

  def __init__(self, pipeline: Pipeline):
    self.pipeline = pipeline

  def optimize_pipeline(self):
    """Optimizes the pipeline execution graph.

    Generates subgraphs of Apache Beam based fuseable components, replaces them
    with a custom FusedComponent, and then rewires the pipeline execution graph.
    """
    fuseable_subgraphs = self.get_fuseable_subgraphs()
    modify_pipeline_exeuction_graph(fuseable_subgraphs)

  def _topologically_sort(self,
                          components: List[base_node.BaseNode],
                          sources: List[base_node.BaseNode]):
    # Determine the indegree for each component w.r.t the current subgraph
    in_degrees = {}
    for component in components:
      in_degrees[component] = 0
      for parent in component.upstream_nodes:
        if parent not in components:
          continue
        in_degrees[component] += 1

    # Perform a BFS on nodes that have a current indegree of 0
    sorted_components = []
    queue = [c for c in components if c in sources]
    queue = sorted(queue, key=lambda c: c.id)

    while queue:
      component = queue.pop(0)
      sorted_components.append(component)

      new_sources = []
      for child in component.downstream_nodes:
        if child not in components:
          continue

        in_degrees[child] -= 1
        if in_degrees[child] == 0:
          new_sources.append(child)

      # This extra sorting keeps a consistent topological ordering
      new_sources = sorted(new_sources, key=lambda c: c.id)
      queue.extend(new_sources)

    return sorted_components

  def _get_intersecting_subgraph(
      self,
      intersecting_component: base_node.BaseNode,
      fuseable_subgraphs: List[List[base_node.BaseNode]]):

    intersecting_subgraph = None
    for subgraph in fuseable_subgraphs:
      if intersecting_component in subgraph:
        intersecting_subgraph = subgraph
        break

    return intersecting_subgraph

  def _is_beam_component(self, component: base_node.BaseNode):
    return issubclass(component.EXECUTOR_SPEC.executor_class,
                      base_executor.FuseableBeamExecutor)

  def _is_fuseable(self, child: base_node.BaseNode,
                   current_subgraph: List[base_node.BaseNode]):
    is_fuseable = True

    # Conduct a BFS to ensure none of the child's ancestors (other than its'
    # immediate parents and those parents' ancestors) are in current_subgraph.
    queue = []
    for parent in child.upstream_nodes:
      if not parent in current_subgraph:
        queue.append(parent)

    while(is_fuseable and queue):
      component = queue.pop(0)

      for parent in component.upstream_nodes:
        if parent in current_subgraph:
          is_fuseable = False
          break
        queue.append(parent)

    return is_fuseable

  def _build_subgraph_from_source(
      self,
      source: base_node.BaseNode,
      fuseable_subgraphs: List[List[base_node.BaseNode]],
      visited: Set[base_node.BaseNode]):

    subgraph = []

    if source in visited:
      return fuseable_subgraphs, visited

    # Conduct a BFS on the candidate source.
    queue = [source]
    while queue:
      component = queue.pop(0)

      if component in visited:
        continue

      visited.add(component)
      subgraph.append(component)

      # Iterate through the children to find Beam components
      for child in component.downstream_nodes:
        if not self._is_beam_component(child):
          continue

        # Checks if the child is in an explored subgraph that needs to be
        # fused into the current subrgaph
        if child in visited:
          intersecting_subgraph = self._get_intersecting_subgraph(
              child, fuseable_subgraphs)
          if intersecting_subgraph:
            subgraph.extend(intersecting_subgraph)
            fuseable_subgraphs.remove(intersecting_subgraph)

        elif self._is_fuseable(child, subgraph):
          queue.append(child)

    if len(subgraph) > 1:
      fuseable_subgraphs.append(subgraph)

    return fuseable_subgraphs, visited

  def get_fuseable_subgraphs(self):
    """Returns a list of fuseable Beam component subgraphs in topological order.

    Conducts multiple BFS searches to build out fuseable Apache Beam component
    subgraphs. Each subgraph S must meet the following correctness constraints:
    1. Each subgraph must be a directed acyclic graph.
    2. All the components in the subgraph must be Apache Beam components.
    3. Consider a component in a subgraph being explored. If the component has
       parents that are not in the current subgraph, then the ancestors of those
       parents must not be in the current subgraph.

    Additionally, each subrgaph must meet the following optimality constraint:
    1. There can be no two subgraphs such that their union satisfies 1, 2, and 3
       as described above.
    """
    # Finds subgraphs of fuseable beam components through a BFS of each source.
    candidate_sources = self.get_subgraph_sources(self.pipeline.components)
    fuseable_subgraphs = []
    visited = set()

    for source in sorted(candidate_sources, key=lambda c: c.id):
      fuseable_subgraphs, visited = self._build_subgraph_from_source(
          source, fuseable_subgraphs, visited)

    # Topologically sort all the subgraphs
    for i in range(len(fuseable_subgraphs)): # pylint: disable=consider-using-enumerate
      fuseable_subgraphs[i] = self._topologically_sort(fuseable_subgraphs[i],
                                                       candidate_sources)
    return fuseable_subgraphs

  def get_subgraph_sources(self, subgraph: List[base_node.BaseNode]):
    """Finds sources (components with no parents that are Apache Beam based components)"""
    subgraph_sources = set()

    for component in subgraph:
      if not self._is_beam_component(component):
        continue

      is_source = True

      for parent in component.upstream_nodes:
        if self._is_beam_component(parent):
          is_source = False
          break

      if is_source:
        subgraph_sources.add(component)

    return subgraph_sources

  def get_subgraph_sinks(self, subgraph: List[base_node.BaseNode]):
    """Finds sinks (components with no children that are in the subgraph)."""
    subgraph_sinks = set()

    for component in subgraph:
      if not self._is_beam_component(component):
        continue

      is_sink = True
      for child in component.downstream_nodes:
        if child in subgraph:
          is_sink = False
          break

      if is_sink:
        subgraph_sinks.add(component)

    return subgraph_sinks

  def _in_fused_component(
      self,
      component: base_node.BaseNode,
      fused_components: List[FusedComponent]):

    for fused_component in fused_components:
      if fused_component.in_subgraph(component):
        return True

    return False

  def _get_source_and_sink_maps(
      self,
      fused_components: List[FusedComponent],
      fuseable_subgraphs: List[List[base_node.BaseNode]]):

    sources_map = {}
    sinks_map = {}

    for i, fused_component in enumerate(fused_components):
      subgraph = fuseable_subgraphs[i]
      sources_map[fused_component] = set(self.get_subgraph_sources(subgraph))
      sinks_map[fused_component] = set(self.get_subgraph_sinks(subgraph))

      # A source in this case will also contain nodes that have upstream
      # dependencies that are not in the current subgraph/fused_component
      for component in subgraph:
        for upstream_node in component.upstream_nodes:
          if upstream_node not in subgraph:
            sources_map[fused_component].add(component)

      for component in subgraph:
        for downstream_node in component.downstream_nodes:
          if downstream_node not in subgraph:
            sinks_map[fused_component].add(component)

    return sources_map, sinks_map

  def _rewire_pipeline_graph(
      self,
      sources_map: Mapping[FusedComponent, List[base_node.BaseNode]],
      sinks_map: Mapping[FusedComponent, List[base_node.BaseNode]],
      fused_components: List[FusedComponent],
      fuseable_subgraphs: List[List[base_node.BaseNode]]):

    for i, fused_component in enumerate(fused_components):
      sources = sources_map[fused_component]
      sinks = sinks_map[fused_component]
      subgraph = fuseable_subgraphs[i]

      source_parent_pairs = set()
      for source in sources:
        for parent in source.upstream_nodes:
          if parent not in subgraph:
            source_parent_pairs.add((source, parent))

      for pair in source_parent_pairs:
        source = pair[0]
        parent = pair[1]
        parent.remove_downstream_node(source)
        parent.add_downstream_node(fused_component)

      sink_child_pairs = set()
      for sink in sinks:
        for child in sink.downstream_nodes:
          if child not in subgraph:
            sink_child_pairs.add((sink, child))

      for pair in sink_child_pairs:
        sink = pair[0]
        child = pair[1]
        child.remove_upstream_node(sink)
        child.add_upstream_node(fused_component)

  def modify_pipeline_exeuction_graph(
      self,
      fuseable_subgraphs: List[List[base_node.BaseNode]]):
    """Rewires the pipeline by switching subgraphs for FusedComponents."""
    # Construct a FusedComponent for each subgraph
    fused_components = []
    for i, subgraph in enumerate(fuseable_subgraphs):
      instance_name = "subgraph_%d" % (i + 1)
      fused_component = FusedComponent(subgraph, instance_name)
      fused_components.append(fused_component)

    # Determine the sources and sinks for each FusedComponent
    sources_map, sinks_map = self._get_source_and_sink_maps(
        fused_components, fuseable_subgraphs)

    # Rewire the nodes to account for the FusedComponents
    self._rewire_pipeline_graph(
        sources_map, sinks_map, fused_components, fuseable_subgraphs)

    # Conduct a BFS to find our new components for the pipeline
    queue = []
    for component in self.pipeline.components:
      if not component.upstream_nodes:
        if not self._in_fused_component(component, fused_components):
          queue.append(component)

    for fused_component in fused_components:
      if not fused_component.upstream_nodes:
        queue.append(fused_component)

    visited = set()
    while queue:
      c = queue.pop(0)
      visited.add(c)

      for child in c.downstream_nodes:
        if child not in visited:
          queue.append(child)

    # Reset the pipeline components accounting for the new FusedComponents
    self.pipeline.components = list(visited)
