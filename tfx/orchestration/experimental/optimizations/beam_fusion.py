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

from typing import List

from tfx.components.base import base_node
from tfx.components.base import base_executor
from tfx.orchestration.pipeline import Pipeline

class BeamFusionOptimizer(object):
  """Optimizer for TFX pipelines, utilizing Beam Fusion"""

  def __init__(self, pipeline: Pipeline):
    self.pipeline = pipeline

  def _topologically_sort(self, components: List[base_node.BaseNode],
                          sources: List[base_node.BaseNode]):

    sorted_components = []
    current_layer = [c for c in components if c in sources]
    visited = set(current_layer)

    # Sorts component in topological order.
    while current_layer:
      next_layer = []
      for component in sorted(current_layer, key=lambda c: c.id):
        sorted_components.append(component)
        for child in component.downstream_nodes:
          if child in components and child not in visited:
            next_layer.append(child)
            visited.add(child)

      current_layer = next_layer

    return sorted_components

  def _get_intersecting_subgraph(self,
                                 intersecting_component: base_node.BaseNode,
                                 fuseable_subgraphs: (
                                     List[List[base_node.BaseNode]])):

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
    current_layer = []
    for parent in child.upstream_nodes:
      if not parent in current_subgraph:
        current_layer.append(parent)

    while(is_fuseable and current_layer):
      next_layer = []
      for component in current_layer:
        for parent in component.upstream_nodes:
          if parent in current_subgraph:
            is_fuseable = False
            break
          next_layer.append(parent)

      current_layer = next_layer

    return is_fuseable

  def get_fuseable_subgraphs(self):
    """Returns a list of fuseable Beam component subgraphs in topological order.

    Conducts multiple BFS searches to build out fuseable Apache Beam component
    subgraphs. Each subgraph S must meet the following correctness constraints:
    1. Each subgraph must be a directed acyclic graph
    2. All the components in the subgraph must be Apache Beam Components
    3. Consider a component in a subgraph being explored. None of the
       component's ancestors (other than its' immediate parents and those
       parents' ancestors) should be in the current subgraph being explored.

    Additionally, each subrgaph must meet the following optimality constraint:
    1. There can be no subgraphs such that their union satisfies 1, 2, and 3
       as described above.
    """

    # Finds subgraphs of fuseable beam components through a BFS of each source.
    candidate_sources = self.get_subgraph_sources(self.pipeline.components)
    fuseable_subgraphs = []
    visited = set()


    for source in sorted(candidate_sources, key=lambda c: c.id): # pylint: disable=too-many-nested-blocks
      if source in visited:
        continue

      subgraph = []
      current_layer = [source]

      # Conduct a BFS on the candidate source.
      while current_layer:
        next_layer = []
        for component in current_layer:
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
              next_layer.append(child)

          current_layer = next_layer

      if len(subgraph) > 1:
        fuseable_subgraphs.append(subgraph)

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
      is_sink = True
      for child in component.downstream_nodes:
        if child in subgraph:
          is_sink = False
          break

      if is_sink:
        subgraph_sinks.add(component)

    return subgraph_sinks
