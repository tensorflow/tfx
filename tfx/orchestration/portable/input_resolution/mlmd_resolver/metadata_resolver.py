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
"""Metadata resolver for reasoning about metadata information."""

from typing import Callable, Dict, List, Optional, Tuple

from tfx.orchestration.portable.input_resolution.mlmd_resolver import metadata_resolver_utils

import ml_metadata as mlmd
from ml_metadata.proto import metadata_store_pb2

_MAX_NUM_HOPS = 100
_MAX_NUM_STARTING_NODES = 100

# Supported field mask paths in LineageGraph message for get_lineage_subgraph().
_ARTIFACTS_FIELD_MASK_PATH = 'artifacts'
_EVENTS_FIELD_MASK_PATH = 'events'
_ARTIFACT_TYPES_MASK_PATH = 'artifact_types'


class MetadataResolver:
  """Metadata resolver for reasoning about metadata information.

  Metadata resolver composes and sends queries to get a lineage graph from
  metadata store. The lineage graph is a snapshot view of the ML pipeline's
  metadata, containing all information needed to answer quetions about the
  lineage of nodes of interest.
  Based on the lineage graph, metadata resolver provides a set of util functions
  that help users reason about metadata information by post-processing the
  graph.
  It can be considered as a wrapper layer built on top of metadata store's graph
  tracing APIs.

  Example:

  # `store` is a metadata store that has been initialized.
  resolver = MetadataResolver(store)
  # Call functions defined in MetadataResolver. For example:
  artifact_ids = [model.id]
  downstream_artifacts_dict = get_downstream_artifacts_by_artifact_ids(
      artifact_ids, max_num_hops = 2
  )
  """

  def __init__(self, store: mlmd.MetadataStore):
    self._store = store

  def get_downstream_artifacts_by_artifact_ids(
      self,
      artifact_ids: List[int],
      max_num_hops: int = _MAX_NUM_HOPS,
      filter_query: str = '',
      event_filter: Optional[Callable[[metadata_store_pb2.Event], bool]] = None,
  ) -> Dict[
      int,
      List[Tuple[metadata_store_pb2.Artifact, metadata_store_pb2.ArtifactType]],
  ]:
    """Given a list of artifact ids, get their provenance successor artifacts.

    For each artifact matched by a given `artifact_id`, treat it as a starting
    artifact and get artifacts that are connected to them within `max_num_hops`
    via a path in the downstream direction like:
    artifact_i -> INPUT_event -> execution_j -> OUTPUT_event -> artifact_k.

    A hop is defined as a jump to the next node following the path of node
    -> event -> next_node.
    For example, in the lineage graph artifact_1 -> event -> execution_1
    -> event -> artifact_2:
    artifact_2 is 2 hops away from artifact_1, and execution_1 is 1 hop away
    from artifact_1.

    Args:
        artifact_ids: ids of starting artifacts. At most 100 ids are supported.
          Returns empty result if `artifact_ids` is empty.
        max_num_hops: maximum number of hops performed for downstream tracing.
          `max_num_hops` cannot exceed 100 nor be negative.
        filter_query: a query string filtering downstream artifacts by their own
          attributes or the attributes of immediate neighbors. Please refer to
          go/mlmd-filter-query-guide for more detailed guidance. Note: if
          `filter_query` is specified and `max_num_hops` is 0, it's equivalent
          to getting filtered artifacts by artifact ids with `get_artifacts()`.
        event_filter: an optional callable object for filtering events in the
          paths towards the downstream artifacts. Only an event with
          `event_filter(event)` evaluated to True will be considered as valid
          and kept in the path.

    Returns:
    Mapping of artifact ids to a list of downstream artifacts.
    """
    # Precondition check.
    if len(artifact_ids) > _MAX_NUM_STARTING_NODES:
      raise ValueError('Number of artifact ids is larger than supported.')
    if not artifact_ids:
      return {}
    if max_num_hops > _MAX_NUM_HOPS or max_num_hops < 0:
      raise ValueError(
          'Number of hops is larger than supported or is negative.'
      )

    artifact_ids_str = ','.join(str(id) for id in artifact_ids)
    # If `max_num_hops` is set to 0, we don't need the graph traversal.
    if max_num_hops == 0:
      if not filter_query:
        artifacts = self._store.get_artifacts_by_id(artifact_ids)
      else:
        artifacts = self._store.get_artifacts(
            list_options=mlmd.ListOptions(
                filter_query=f'id IN ({artifact_ids_str}) AND ({filter_query})',
                limit=_MAX_NUM_STARTING_NODES,
            )
        )
      artifact_type_ids = [a.type_id for a in artifacts]
      artifact_types = self._store.get_artifact_types_by_id(artifact_type_ids)
      artifact_type_by_id = {t.id: t for t in artifact_types}
      return {
          artifact.id: [(artifact, artifact_type_by_id[artifact.type_id])]
          for artifact in artifacts
      }

    options = metadata_store_pb2.LineageSubgraphQueryOptions(
        starting_artifacts=metadata_store_pb2.LineageSubgraphQueryOptions.StartingNodes(
            filter_query=f'id IN ({artifact_ids_str})'
        ),
        max_num_hops=max_num_hops,
        direction=metadata_store_pb2.LineageSubgraphQueryOptions.Direction.DOWNSTREAM,
    )
    field_mask_paths = [
        _ARTIFACTS_FIELD_MASK_PATH,
        _EVENTS_FIELD_MASK_PATH,
        _ARTIFACT_TYPES_MASK_PATH,
    ]
    lineage_graph = self._store.get_lineage_subgraph(
        query_options=options,
        field_mask_paths=field_mask_paths,
    )

    artifact_type_by_id = {t.id: t for t in lineage_graph.artifact_types}

    if not filter_query:
      artifacts_to_subgraph = metadata_resolver_utils.get_subgraphs_by_artifact_ids(
          artifact_ids,
          metadata_store_pb2.LineageSubgraphQueryOptions.Direction.DOWNSTREAM,
          lineage_graph,
          event_filter,
      )
      return {
          artifact_id: [
              [a, artifact_type_by_id[a.type_id]] for a in subgraph.artifacts
          ]
          for artifact_id, subgraph in artifacts_to_subgraph.items()
      }
    else:
      artifacts_to_visited_ids = metadata_resolver_utils.get_visited_ids_by_artifact_ids(
          artifact_ids,
          metadata_store_pb2.LineageSubgraphQueryOptions.Direction.DOWNSTREAM,
          lineage_graph,
          event_filter,
      )

      candidate_artifact_ids = set()
      for visited_ids in artifacts_to_visited_ids.values():
        candidate_artifact_ids.update(
            visited_ids[metadata_resolver_utils.NodeType.ARTIFACT]
        )
      artifact_ids_str = ','.join(str(id) for id in candidate_artifact_ids)
      # Send a call to metadata_store to get filtered downstream artifacts.
      artifacts = self._store.get_artifacts(
          list_options=mlmd.ListOptions(
              filter_query=f'id IN ({artifact_ids_str}) AND ({filter_query})'
          )
      )
      artifact_id_to_artifact = {
          artifact.id: artifact for artifact in artifacts
      }
      downstream_artifacts_dict = {}
      for artifact_id, visited_ids in artifacts_to_visited_ids.items():
        downstream_artifacts = [
            (
                artifact_id_to_artifact[id],
                artifact_type_by_id[artifact_id_to_artifact[id].type_id],
            )
            for id in visited_ids[metadata_resolver_utils.NodeType.ARTIFACT]
            if id in artifact_id_to_artifact
        ]
        if downstream_artifacts:
          downstream_artifacts_dict[artifact_id] = downstream_artifacts
      return downstream_artifacts_dict

  def get_downstream_artifacts_by_artifact_uri(
      self, artifact_uri: str, max_num_hops: int = _MAX_NUM_HOPS
  ) -> Dict[int, List[metadata_store_pb2.Artifact]]:
    """Get matched artifacts of a uri and their provenance successor artifacts.

    For each artifact matched by the given `artifact_uri`, treat it as a
    starting artifact and get artifacts that are connected to them via a path in
    the downstream direction like:
    artifact_i -> INPUT_event -> execution_j -> OUTPUT_event -> artifact_k.

    Args:
        artifact_uri: the uri of starting artifacts. At most 100 artifacts
          matched by the uri are considered as starting artifacts.
        max_num_hops: maximum number of hops performed for downstream tracing. A
          hop is defined as a jump to the next node following the path of node
          -> event -> next_node. For example, in the lineage graph artifact_1 ->
          event -> execution_1 -> event -> artifact_2: artifact_2 is 2 hops away
          from artifact_1, and execution_1 is 1 hop away from artifact_1.
          `max_num_hops` cannot exceed 100 nor be negative.

    Returns:
        Mapping of artifact ids to a list of downstream artifacts.
    """
    if not artifact_uri:
      raise ValueError('`artifact_uri` is empty.')
    if max_num_hops > _MAX_NUM_HOPS or max_num_hops < 0:
      raise ValueError(
          'Number of hops is larger than supported or is negative.'
      )

    starting_artifacts_filter_query = f'uri = "{artifact_uri}"'

    options = metadata_store_pb2.LineageSubgraphQueryOptions(
        starting_artifacts=metadata_store_pb2.LineageSubgraphQueryOptions.StartingNodes(
            filter_query=starting_artifacts_filter_query
        ),
        max_num_hops=max_num_hops,
        direction=metadata_store_pb2.LineageSubgraphQueryOptions.Direction.DOWNSTREAM,
    )
    lineage_graph = self._store.get_lineage_subgraph(
        query_options=options,
        field_mask_paths=[
            _ARTIFACTS_FIELD_MASK_PATH,
            _EVENTS_FIELD_MASK_PATH,
        ],
    )

    artifact_ids = [
        artifact.id
        for artifact in lineage_graph.artifacts
        if artifact.uri == artifact_uri
    ]
    artifacts_to_subgraph = (
        metadata_resolver_utils.get_subgraphs_by_artifact_ids(
            artifact_ids,
            metadata_store_pb2.LineageSubgraphQueryOptions.Direction.DOWNSTREAM,
            lineage_graph,
        )
    )
    return {
        artifact_id: list(subgraph.artifacts)
        for artifact_id, subgraph in artifacts_to_subgraph.items()
    }

  def get_upstream_artifacts_by_artifact_ids(
      self,
      artifact_ids: List[int],
      max_num_hops: int = _MAX_NUM_HOPS,
      filter_query: str = '',
      event_filter: Optional[Callable[[metadata_store_pb2.Event], bool]] = None,
  ) -> Dict[
      int,
      List[Tuple[metadata_store_pb2.Artifact, metadata_store_pb2.ArtifactType]],
  ]:
    """Given a list of artifact ids, get their provenance ancestor artifacts.

    For each artifact matched by a given `artifact_id`, treat it as a starting
    artifact and get artifacts that are connected to them within `max_num_hops`
    via a path in the upstream direction like:
    artifact_i -> OUTPUT_event -> execution_j -> INPUT_event -> artifact_k.

    A hop is defined as a jump to the next node following the path of node
    -> event -> next_node.
    For example, in the lineage graph artifact_1 -> event -> execution_1
    -> event -> artifact_2:
    artifact_2 is 2 hops away from artifact_1, and execution_1 is 1 hop away
    from artifact_1.

    Args:
        artifact_ids: ids of starting artifacts. At most 100 ids are supported.
          Returns empty result if `artifact_ids` is empty.
        max_num_hops: maximum number of hops performed for upstream tracing.
          `max_num_hops` cannot exceed 100 nor be negative.
        filter_query: a query string filtering upstream artifacts by their own
          attributes or the attributes of immediate neighbors. Please refer to
          go/mlmd-filter-query-guide for more detailed guidance. Note: if
          `filter_query` is specified and `max_num_hops` is 0, it's equivalent
          to getting filtered artifacts by artifact ids with `get_artifacts()`.
        event_filter: an optional callable object for filtering events in the
          paths towards the upstream artifacts. Only an event with
          `event_filter(event)` evaluated to True will be considered as valid
          and kept in the path.

    Returns:
    Mapping of artifact ids to a list of upstream artifacts.
    """
    if len(artifact_ids) > _MAX_NUM_STARTING_NODES:
      raise ValueError('Number of artifact ids is larger than supported.')
    if not artifact_ids:
      return {}
    if max_num_hops > _MAX_NUM_HOPS or max_num_hops < 0:
      raise ValueError(
          'Number of hops is larger than supported or is negative.'
      )

    artifact_ids_str = ','.join(str(id) for id in artifact_ids)
    # If `max_num_hops` is set to 0, we don't need the graph traversal.
    if max_num_hops == 0:
      if not filter_query:
        artifacts = self._store.get_artifacts_by_id(artifact_ids)
      else:
        artifacts = self._store.get_artifacts(
            list_options=mlmd.ListOptions(
                filter_query=f'id IN ({artifact_ids_str}) AND ({filter_query})',
                limit=_MAX_NUM_STARTING_NODES,
            )
        )
      artifact_type_ids = [a.type_id for a in artifacts]
      artifact_types = self._store.get_artifact_types_by_id(artifact_type_ids)
      artifact_type_by_id = {t.id: t for t in artifact_types}
      return {
          artifact.id: [(artifact, artifact_type_by_id[artifact.type_id])]
          for artifact in artifacts
      }

    options = metadata_store_pb2.LineageSubgraphQueryOptions(
        starting_artifacts=metadata_store_pb2.LineageSubgraphQueryOptions.StartingNodes(
            filter_query=f'id IN ({artifact_ids_str})'
        ),
        max_num_hops=max_num_hops,
        direction=metadata_store_pb2.LineageSubgraphQueryOptions.Direction.UPSTREAM,
    )
    field_mask_paths = [
        _ARTIFACTS_FIELD_MASK_PATH,
        _EVENTS_FIELD_MASK_PATH,
        _ARTIFACT_TYPES_MASK_PATH,
    ]
    lineage_graph = self._store.get_lineage_subgraph(
        query_options=options,
        field_mask_paths=field_mask_paths,
    )

    artifact_type_by_id = {t.id: t for t in lineage_graph.artifact_types}

    if not filter_query:
      artifacts_to_subgraph = (
          metadata_resolver_utils.get_subgraphs_by_artifact_ids(
              artifact_ids,
              metadata_store_pb2.LineageSubgraphQueryOptions.Direction.UPSTREAM,
              lineage_graph,
              event_filter,
          )
      )
      return {
          artifact_id: [
              [a, artifact_type_by_id[a.type_id]] for a in subgraph.artifacts
          ]
          for artifact_id, subgraph in artifacts_to_subgraph.items()
      }
    else:
      artifacts_to_visited_ids = (
          metadata_resolver_utils.get_visited_ids_by_artifact_ids(
              artifact_ids,
              metadata_store_pb2.LineageSubgraphQueryOptions.Direction.UPSTREAM,
              lineage_graph,
              event_filter,
          )
      )
      candidate_artifact_ids = set()
      for visited_ids in artifacts_to_visited_ids.values():
        candidate_artifact_ids.update(
            visited_ids[metadata_resolver_utils.NodeType.ARTIFACT]
        )
      artifact_ids_str = ','.join(str(id) for id in candidate_artifact_ids)
      # Send a call to metadata_store to get filtered upstream artifacts.
      artifacts = self._store.get_artifacts(
          list_options=mlmd.ListOptions(
              filter_query=f'id IN ({artifact_ids_str}) AND ({filter_query})'
          )
      )
      artifact_id_to_artifact = {
          artifact.id: artifact for artifact in artifacts
      }
      upstream_artifacts_dict = {}
      for artifact_id, visited_ids in artifacts_to_visited_ids.items():
        upstream_artifacts = [
            (
                artifact_id_to_artifact[id],
                artifact_type_by_id[artifact_id_to_artifact[id].type_id],
            )
            for id in visited_ids[metadata_resolver_utils.NodeType.ARTIFACT]
            if id in artifact_id_to_artifact
        ]
        if upstream_artifacts:
          upstream_artifacts_dict[artifact_id] = upstream_artifacts
      return upstream_artifacts_dict

  def get_upstream_artifacts_by_artifact_uri(
      self, artifact_uri: str, max_num_hops: int = _MAX_NUM_HOPS
  ) -> Dict[int, List[metadata_store_pb2.Artifact]]:
    """Get matched artifacts of a uri and their provenance ancestor artifacts.

    For each artifact matched by the given `artifact_uri`, treat it as a
    starting artifact and get artifacts that are connected to them via a path in
    the upstream direction like:
    artifact_i -> OUTPUT_event -> execution_j -> INPUT_event -> artifact_k.

    Args:
        artifact_uri: the uri of starting artifacts. At most 100 artifacts
          matched by the uri are considered as starting artifacts.
        max_num_hops: maximum number of hops performed for upstream tracing. A
          hop is defined as a jump to the next node following the path of node
          -> event -> next_node. For example, in the lineage graph artifact_1 ->
          event -> execution_1 -> event -> artifact_2: artifact_2 is 2 hops away
          from artifact_1, and execution_1 is 1 hop away from artifact_1.
          `max_num_hops` cannot exceed 100 nor be negative.

    Returns:
        Mapping of artifact ids to a list of upstream artifacts.
    """
    if not artifact_uri:
      raise ValueError('`artifact_uri` is empty.')
    if max_num_hops > _MAX_NUM_HOPS or max_num_hops < 0:
      raise ValueError(
          'Number of hops is larger than supported or is negative.'
      )

    starting_artifacts_filter_query = f'uri = "{artifact_uri}"'

    options = metadata_store_pb2.LineageSubgraphQueryOptions(
        starting_artifacts=metadata_store_pb2.LineageSubgraphQueryOptions.StartingNodes(
            filter_query=starting_artifacts_filter_query
        ),
        max_num_hops=max_num_hops,
        direction=metadata_store_pb2.LineageSubgraphQueryOptions.Direction.UPSTREAM,
    )
    lineage_graph = self._store.get_lineage_subgraph(
        query_options=options,
        field_mask_paths=[
            _ARTIFACTS_FIELD_MASK_PATH,
            _EVENTS_FIELD_MASK_PATH,
        ],
    )

    artifact_ids = [
        artifact.id
        for artifact in lineage_graph.artifacts
        if artifact.uri == artifact_uri
    ]
    artifacts_to_subgraph = (
        metadata_resolver_utils.get_subgraphs_by_artifact_ids(
            artifact_ids,
            metadata_store_pb2.LineageSubgraphQueryOptions.Direction.UPSTREAM,
            lineage_graph,
        )
    )
    return {
        artifact_id: list(subgraph.artifacts)
        for artifact_id, subgraph in artifacts_to_subgraph.items()
    }
