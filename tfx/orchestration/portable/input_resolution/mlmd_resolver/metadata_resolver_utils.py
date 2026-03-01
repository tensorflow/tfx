# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utils for metadata resolver."""

import collections
import enum
from typing import Callable, Dict, List, Optional, Set

import attr

from ml_metadata.proto import metadata_store_pb2


INPUT_EVENT_TYPES = {
    metadata_store_pb2.Event.DECLARED_INPUT,
    metadata_store_pb2.Event.INPUT,
    metadata_store_pb2.Event.INTERNAL_INPUT,
}

OUTPUT_EVENT_TYPES = {
    metadata_store_pb2.Event.DECLARED_OUTPUT,
    metadata_store_pb2.Event.INTERNAL_OUTPUT,
    metadata_store_pb2.Event.OUTPUT,
    metadata_store_pb2.Event.PENDING_OUTPUT,
}


class EventType(enum.Enum):
  INPUT = 1
  OUTPUT = 2


class NodeType(enum.Enum):
  UNSPECIFIED = 0
  ARTIFACT = 1
  EXECUTION = 2
  CONTEXT = 3


def _initialize_resolver_default_dict():
  return collections.defaultdict(lambda: collections.defaultdict(list))


@attr.define
class ResolverGraph:
  """A resolver graph dedicated for in-memory graph traversal.

  The resolver graph was in the form of adjacency lists. It captures artifacts'
  and executions' information and their relations in a lineage graph.
  Please refer to`_build_resolver_graph()` for more details.
  """

  artifacts_by_id: Dict[int, metadata_store_pb2.Artifact] = attr.field(
      factory=dict
  )
  executions_by_id: Dict[int, metadata_store_pb2.Execution] = attr.field(
      factory=dict
  )
  artifact_to_execution: Dict[EventType, Dict[int, List[int]]] = attr.field(
      factory=_initialize_resolver_default_dict
  )
  execution_to_artifact: Dict[EventType, Dict[int, List[int]]] = attr.field(
      factory=_initialize_resolver_default_dict
  )


def _get_resolver_event_type(event: metadata_store_pb2.Event) -> EventType:
  """Gets an indicator of whether `event` is an input / output event.

  Args:
    event: an event object, with an event type associated.

  Returns:
    An `EventType` enum indicating whether `event` is an input / output
    event.
  """

  if event.type in INPUT_EVENT_TYPES:
    return EventType.INPUT
  elif event.type in OUTPUT_EVENT_TYPES:
    return EventType.OUTPUT
  else:
    raise ValueError("Event without type.")


def _explore_from_artifact(
    starting_artifact_id: int,
    direction: metadata_store_pb2.LineageSubgraphQueryOptions.Direction,
    resolver_graph: ResolverGraph,
    visited_ids: Dict[NodeType, Set[int]],
    subgraph: metadata_store_pb2.LineageGraph,
) -> None:
  """Given a starting artifact, runs a single dfs on the graph from it.

  Args:
    starting_artifact_id: starting artifact id.
    direction: direction of dfs. It can be single-directional or bidirectional.
    resolver_graph: resolver graph representing the lineage graph to run dfs on.
    visited_ids: a set of visited node ids.
    subgraph: lineage graph to store returned nodes from dfs.
  """
  visited_ids[NodeType.ARTIFACT].add(starting_artifact_id)
  # If no artifacts are returned with the lineage_graph from
  # `get_lineage_subgraph()`, the `resolver_graph` will also have
  # `artifacts_by_id` being empty. Therefore we don't append any artifact to the
  # returned `subgraph`.
  if resolver_graph.artifacts_by_id:
    subgraph.artifacts.append(
        resolver_graph.artifacts_by_id[starting_artifact_id]
    )
  if direction in [
      metadata_store_pb2.LineageSubgraphQueryOptions.Direction.DOWNSTREAM,
      metadata_store_pb2.LineageSubgraphQueryOptions.Direction.BIDIRECTIONAL,
  ]:
    if (
        starting_artifact_id
        in resolver_graph.artifact_to_execution[EventType.INPUT]
    ):
      for execution_id in resolver_graph.artifact_to_execution[EventType.INPUT][
          starting_artifact_id
      ]:
        if execution_id not in visited_ids[NodeType.EXECUTION]:
          _explore_from_execution(
              execution_id, direction, resolver_graph, visited_ids, subgraph
          )
  if direction in [
      metadata_store_pb2.LineageSubgraphQueryOptions.Direction.UPSTREAM,
      metadata_store_pb2.LineageSubgraphQueryOptions.Direction.BIDIRECTIONAL,
  ]:
    if (
        starting_artifact_id
        in resolver_graph.artifact_to_execution[EventType.OUTPUT]
    ):
      for execution_id in resolver_graph.artifact_to_execution[
          EventType.OUTPUT
      ][starting_artifact_id]:
        if execution_id not in visited_ids[NodeType.EXECUTION]:
          _explore_from_execution(
              execution_id, direction, resolver_graph, visited_ids, subgraph
          )


def _explore_from_execution(
    starting_execution_id: int,
    direction: metadata_store_pb2.LineageSubgraphQueryOptions.Direction,
    resolver_graph: ResolverGraph,
    visited_ids: Dict[NodeType, Set[int]],
    subgraph: metadata_store_pb2.LineageGraph,
):
  """Given a starting execution, runs a single dfs on the graph from it.

  Args:
    starting_execution_id: starting execution id.
    direction: direction of dfs. It can be single-directional or bidirectional.
    resolver_graph: resolver graph representing the lineage graph to run dfs on.
    visited_ids: a set of visited node ids.
    subgraph: lineage graph to store returned nodes from dfs.
  """
  visited_ids[NodeType.EXECUTION].add(starting_execution_id)
  # If no executions are returned with the lineage_graph from
  # `get_lineage_subgraph()`, the `resolver_graph` will also have
  # `executions_by_id` being empty. Therefore we don't append any execution to
  # the returned `subgraph`.
  if resolver_graph.executions_by_id:
    subgraph.executions.append(
        resolver_graph.executions_by_id[starting_execution_id]
    )
  if direction in [
      metadata_store_pb2.LineageSubgraphQueryOptions.Direction.UPSTREAM,
      metadata_store_pb2.LineageSubgraphQueryOptions.Direction.BIDIRECTIONAL,
  ]:
    if (
        starting_execution_id
        in resolver_graph.execution_to_artifact[EventType.INPUT].keys()
    ):
      for artifact_id in resolver_graph.execution_to_artifact[EventType.INPUT][
          starting_execution_id
      ]:
        if artifact_id not in visited_ids[NodeType.ARTIFACT]:
          _explore_from_artifact(
              artifact_id, direction, resolver_graph, visited_ids, subgraph
          )
  if direction in [
      metadata_store_pb2.LineageSubgraphQueryOptions.Direction.DOWNSTREAM,
      metadata_store_pb2.LineageSubgraphQueryOptions.Direction.BIDIRECTIONAL,
  ]:
    if (
        starting_execution_id
        in resolver_graph.execution_to_artifact[EventType.OUTPUT].keys()
    ):
      for artifact_id in resolver_graph.execution_to_artifact[EventType.OUTPUT][
          starting_execution_id
      ]:
        if artifact_id not in visited_ids[NodeType.ARTIFACT]:
          _explore_from_artifact(
              artifact_id, direction, resolver_graph, visited_ids, subgraph
          )


def get_subgraphs_by_artifact_ids(
    starting_artifact_ids: List[int],
    direction: metadata_store_pb2.LineageSubgraphQueryOptions.Direction,
    graph: metadata_store_pb2.LineageGraph,
    optional_event_filter: Optional[
        Callable[[metadata_store_pb2.Event], bool]
    ] = None,
) -> Dict[int, metadata_store_pb2.LineageGraph]:
  """Given a list of starting artifacts, retrieve the subgraphs connected.

  Args:
    starting_artifact_ids: starting artifact ids.
    direction: direction of dfs. It can be single-directional or bidirectional.
    graph: the lineage graph to run dfs on.
    optional_event_filter: an optional callable object for filtering events in
      the paths. Only an event with `optional_event_filter(event)` evaluated to
      True will be considered as valid and kept in the path.

  Returns:
    Mappings of starting artifact ids and subgraphs traced from dfs. The
    subgraphs contain only nodes.
  """
  resolver_graph = _build_resolver_graph(graph, optional_event_filter)
  artifact_to_subgraph = {}

  for artifact_id in starting_artifact_ids:
    visited_ids = {NodeType.ARTIFACT: set(), NodeType.EXECUTION: set()}
    subgraph = metadata_store_pb2.LineageGraph()
    _explore_from_artifact(
        artifact_id, direction, resolver_graph, visited_ids, subgraph
    )
    artifact_to_subgraph[artifact_id] = subgraph
  return artifact_to_subgraph


def get_visited_ids_by_artifact_ids(
    starting_artifact_ids: List[int],
    direction: metadata_store_pb2.LineageSubgraphQueryOptions.Direction,
    graph: metadata_store_pb2.LineageGraph,
    optional_event_filter: Optional[
        Callable[[metadata_store_pb2.Event], bool]
    ] = None,
) -> Dict[int, Dict[NodeType, Set[int]]]:
  """Given a list of starting artifacts, retrieve the visited ids explored.

  Given a list of starting artifacts, returns a mapping of each artifact id
  and the visited nodes of each dfs derived from it.

  Args:
    starting_artifact_ids: starting artifact ids.
    direction: direction of dfs. It can be single-directional or bidirectional.
    graph: the lineage graph to run dfs on.
    optional_event_filter: an optional callable object for filtering events in
      the paths. Only an event with `optional_event_filter(event)` evaluated to
      True will be considered as valid and kept in the path.

  Returns:
    Mappings of starting artifact ids and visited ids explored in dfs.
  """
  resolver_graph = _build_resolver_graph(graph, optional_event_filter)
  artifact_to_visited_ids = collections.defaultdict(
      lambda: collections.defaultdict(set)
  )
  for artifact_id in starting_artifact_ids:
    visited_ids = {NodeType.ARTIFACT: set(), NodeType.EXECUTION: set()}
    _explore_from_artifact(
        artifact_id,
        direction,
        resolver_graph,
        visited_ids,
        metadata_store_pb2.LineageGraph(),
    )
    artifact_to_visited_ids[artifact_id].update(visited_ids)
  return artifact_to_visited_ids


def _build_resolver_graph(
    lineage_graph: metadata_store_pb2.LineageGraph,
    optional_event_filter: Optional[
        Callable[[metadata_store_pb2.Event], bool]
    ] = None,
) -> ResolverGraph:
  """Builds a resolver graph from a lineage graph.

  For example, if lineage_graph is:
  {
      artifacts: {
          id: 1
          # other fields
      }
      artifacts: {
          id: 2
          # other fields
      }
      executions: {
          id: 10
          # other fields
      }
      events: {
          artifact_id: 1
          execution_id: 10
          type: INPUT
      }
      events: {
          artifact_id: 2
          execution_id: 10
          type: OUTPUT
      }
  }
  The resolver graph returned will be:
  ResolverGraph(
      artifacts_by_id={
          1: Artifact(id=1, #other_fields),
          2: Artifact(id=2, #other_fields)
      },
      executions_by_id={
          10: Execution(id=10, #other_fields)
      },
      artifact_to_execution={
          EventType.INPUT: {1: [10]},
          EventType.OUTPUT: {2: [10]}},
      execution_to_artifact={
          EventType.INPUT: {10: [1]},
          EventType.OUTPUT: {10: [2]}
      }
  )

  Args:
    lineage_graph: lineage graph to build the resolver graph from.
    optional_event_filter: an optional callable object for filtering events in
      the paths. Only an event with `optional_event_filter(event)` evaluated to
      True will be considered as valid and kept in the path.

  Returns:
    A resolver graph dedicated for in-memory graph traversal.
  """
  resolver_graph = ResolverGraph()

  for event in lineage_graph.events:
    if optional_event_filter is not None and not optional_event_filter(event):
      continue
    event_type = _get_resolver_event_type(event)

    resolver_graph.artifact_to_execution[event_type][event.artifact_id].append(
        event.execution_id
    )
    resolver_graph.execution_to_artifact[event_type][event.execution_id].append(
        event.artifact_id
    )

  for artifact in lineage_graph.artifacts:
    resolver_graph.artifacts_by_id[artifact.id] = artifact
  for execution in lineage_graph.executions:
    resolver_graph.executions_by_id[execution.id] = execution
  return resolver_graph
