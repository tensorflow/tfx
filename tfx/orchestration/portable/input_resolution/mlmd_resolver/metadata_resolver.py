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

import collections
from typing import Callable, Dict, List, Optional, Tuple, Union

from tfx.orchestration import mlmd_connection_manager as mlmd_cm
from tfx.orchestration.portable.input_resolution.mlmd_resolver import metadata_resolver_utils
from tfx.types import external_artifact_utils

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

  def __init__(
      self,
      store: mlmd.MetadataStore,
      mlmd_connection_manager: Optional[mlmd_cm.MLMDConnectionManager] = None,
  ):
    self._store = store
    self._mlmd_connection_manager = mlmd_connection_manager

  def _evaluate_filter_query(
      self,
      artifact: metadata_store_pb2.Artifact,
      artifact_type: Optional[metadata_store_pb2.ArtifactType],
      filter_query: str,
  ) -> bool:
    """Evaluates simple metadata resolver filter queries locally in python."""
    if not filter_query:
      return True

    query = filter_query.strip()

    if ' OR ' in query or ' or ' in query:
      or_clauses = query.replace(' OR ', ' or ').split(' or ')
      return any(
          self._evaluate_filter_query(artifact, artifact_type, c)
          for c in or_clauses
      )

    if ' AND ' in query or ' and ' in query:
      and_clauses = query.replace(' AND ', ' and ').split(' and ')
      return all(
          self._evaluate_filter_query(artifact, artifact_type, c)
          for c in and_clauses
      )

    if ' IN ' in query or ' in ' in query:
      field, values_str = query.replace(' IN ', ' in ').split(' in ')
      field = field.strip()
      values = [v.strip('"\' ') for v in values_str.strip('()').split(',')]
      if field == 'name':
        return artifact.name in values
      elif field == 'type' and artifact_type:
        return artifact_type.name in values
      elif field == 'id':
        return artifact.id in [int(v) for v in values]
      return False

    if '=' in query:
      field, val = query.split('=', 1)
      field = field.strip()
      val = val.strip('"\' ')
      if field == 'name':
        return artifact.name == val
      elif field == 'type' and artifact_type:
        return artifact_type.name == val
      elif field == 'id':
        return str(artifact.id) == val
      return False

    return True

  def _get_filtered_artifacts(
      self,
      artifact_ids: List[int],
      filter_query: Optional[str] = None,
      limit: Optional[int] = None,
  ) -> List[metadata_store_pb2.Artifact]:
    """Gets artifacts by ID and applies filter query fallback locally if ZetaSQL is missing."""
    if not artifact_ids:
      return []

    try:
      artifact_ids_str = ','.join(str(id) for id in artifact_ids)
      fq = f'id IN ({artifact_ids_str})'
      if filter_query:
        fq = f'{fq} AND ({filter_query})'
      list_options = mlmd.ListOptions(filter_query=fq)
      if limit:
        list_options.limit = limit
      return self._store.get_artifacts(list_options=list_options)
    except Exception as e:
      if 'ZetaSQL dependency removed' not in str(e):
        raise e

      # Non-ZetaSQL Fallback Query Processing:
      artifacts = self._store.get_artifacts_by_id(artifact_ids)
      if not filter_query:
        filtered = artifacts
      else:
        type_ids = {a.type_id for a in artifacts}
        artifact_types = self._store.get_artifact_types_by_id(list(type_ids))
        artifact_type_by_id = {t.id: t for t in artifact_types}
        filtered = [
            a
            for a in artifacts
            if self._evaluate_filter_query(
                a, artifact_type_by_id.get(a.type_id), filter_query
            )
        ]
      if limit:
        filtered = filtered[:limit]
      return filtered

  def _get_lineage_subgraph_fallback(
      self,
      direction: metadata_store_pb2.LineageSubgraphQueryOptions.Direction,
      starting_artifact_ids: List[int],
      max_num_hops: int,
  ) -> metadata_store_pb2.LineageGraph:
    """Builds a lineage subgraph recursively in Python for ZetaSQL-disabled environments."""
    artifacts_by_id = {}
    events_by_key = {}

    starting_artifacts = self._store.get_artifacts_by_id(starting_artifact_ids)
    for a in starting_artifacts:
      artifacts_by_id[a.id] = a

    current_artifact_ids = set(starting_artifact_ids)
    hops_remaining = max_num_hops

    while current_artifact_ids and hops_remaining > 0:
      events = self._store.get_events_by_artifact_ids(
          list(current_artifact_ids)
      )

      if (
          direction
          == metadata_store_pb2.LineageSubgraphQueryOptions.Direction.DOWNSTREAM
      ):
        target_events = [
            e
            for e in events
            if e.type
            in [
                metadata_store_pb2.Event.INPUT,
                metadata_store_pb2.Event.DECLARED_INPUT,
            ]
        ]
      else:
        target_events = [
            e
            for e in events
            if e.type
            in [
                metadata_store_pb2.Event.OUTPUT,
                metadata_store_pb2.Event.DECLARED_OUTPUT,
                metadata_store_pb2.Event.PENDING_OUTPUT,
            ]
        ]

      if not target_events:
        break

      execution_ids = {e.execution_id for e in target_events}

      all_exec_events = self._store.get_events_by_execution_ids(
          list(execution_ids)
      )

      if (
          direction
          == metadata_store_pb2.LineageSubgraphQueryOptions.Direction.DOWNSTREAM
      ):
        neighbor_events = [
            e
            for e in all_exec_events
            if e.type
            in [
                metadata_store_pb2.Event.OUTPUT,
                metadata_store_pb2.Event.DECLARED_OUTPUT,
                metadata_store_pb2.Event.PENDING_OUTPUT,
            ]
        ]
      else:
        neighbor_events = [
            e
            for e in all_exec_events
            if e.type
            in [
                metadata_store_pb2.Event.INPUT,
                metadata_store_pb2.Event.DECLARED_INPUT,
            ]
        ]

      if not neighbor_events:
        break

      # Verify if any new path links have been mapped during this hop
      new_events_found = False
      for e in target_events + neighbor_events:
        key = (e.artifact_id, e.execution_id, e.type)
        if key not in events_by_key:
          events_by_key[key] = e
          new_events_found = True

      if not new_events_found:
        break

      next_artifact_ids = {e.artifact_id for e in neighbor_events}
      new_artifact_ids = next_artifact_ids - set(artifacts_by_id.keys())

      if new_artifact_ids:
        next_artifacts = self._store.get_artifacts_by_id(list(new_artifact_ids))
        for a in next_artifacts:
          artifacts_by_id[a.id] = a

      current_artifact_ids = next_artifact_ids
      hops_remaining -= 2

    lineage_graph = metadata_store_pb2.LineageGraph()
    lineage_graph.artifacts.extend(artifacts_by_id.values())
    lineage_graph.events.extend(events_by_key.values())

    type_ids = {a.type_id for a in artifacts_by_id.values()}
    artifact_types = self._store.get_artifact_types_by_id(list(type_ids))
    lineage_graph.artifact_types.extend(artifact_types)

    return lineage_graph

  def _get_lineage_subgraph(
      self,
      query_options: metadata_store_pb2.LineageSubgraphQueryOptions,
      field_mask_paths: List[str],
  ) -> metadata_store_pb2.LineageGraph:
    """Invokes get_lineage_subgraph, with local python fallback if ZetaSQL is missing."""
    try:
      return self._store.get_lineage_subgraph(
          query_options=query_options,
          field_mask_paths=field_mask_paths,
      )
    except Exception as e:
      if 'ZetaSQL dependency removed' not in str(e):
        raise e

      starting_nodes = query_options.starting_artifacts
      if 'id IN (' in starting_nodes.filter_query:
        ids_str = starting_nodes.filter_query.split('id IN (')[1].split(')')[0]
        starting_artifact_ids = [
            int(i.strip()) for i in ids_str.split(',') if i.strip()
        ]
      elif 'uri = ' in starting_nodes.filter_query:
        uri = starting_nodes.filter_query.split('uri = ')[1].strip('"\' ')
        starting_artifacts = self._store.get_artifacts_by_uri(uri)
        starting_artifact_ids = [a.id for a in starting_artifacts]
      else:
        raise NotImplementedError(
            'Unsupported filter query for starting nodes fallback:'
            f' {starting_nodes.filter_query}'
        )

      return self._get_lineage_subgraph_fallback(
          direction=query_options.direction,
          starting_artifact_ids=starting_artifact_ids,
          max_num_hops=query_options.max_num_hops,
      )

  def _get_external_upstream_or_downstream_artifacts(
      self,
      external_artifact_ids: List[str],
      max_num_hops: int = _MAX_NUM_HOPS,
      filter_query: str = '',
      event_filter: Optional[Callable[[metadata_store_pb2.Event], bool]] = None,
      downstream: bool = True,
  ):
    """Gets downstream or upstream artifacts from external artifact ids.

    Args:
      external_artifact_ids: A list of external artifact ids.
      max_num_hops: maximum number of hops performed for tracing. `max_num_hops`
        cannot exceed 100 nor be negative.
      filter_query: a query string filtering artifacts by their own attributes
        or the attributes of immediate neighbors. Please refer to
        go/mlmd-filter-query-guide for more detailed guidance. Note: if
        `filter_query` is specified and `max_num_hops` is 0, it's equivalent to
        getting filtered artifacts by artifact ids with `get_artifacts()`.
      event_filter: an optional callable object for filtering events in the
        paths towards the artifacts. Only an event with `event_filter(event)`
        evaluated to True will be considered as valid and kept in the path.
      downstream: If true, get downstream artifacts. Otherwise, get upstream
        artifacts.

    Returns:
      Mapping of artifact ids to a list of downstream or upstream artifacts.

    Raises:
      ValueError: If mlmd_connection_manager is not initialized.
    """
    if not self._mlmd_connection_manager:
      raise ValueError(
          'mlmd_connection_manager is not initialized. There are external'
          'artifacts, so we need it to query the external MLMD instance.'
      )

    store_by_pipeline_asset: Dict[str, mlmd.MetadataStore] = {}
    external_ids_by_pipeline_asset: Dict[str, List[str]] = (
        collections.defaultdict(list)
    )
    for external_id in external_artifact_ids:
      connection_config = (
          external_artifact_utils.get_external_connection_config(external_id)
      )
      store = self._mlmd_connection_manager.get_mlmd_handle(
          connection_config
      ).store
      pipeline_asset = (
          external_artifact_utils.get_pipeline_asset_from_external_id(
              external_id
          )
      )
      external_ids_by_pipeline_asset[pipeline_asset].append(external_id)
      store_by_pipeline_asset[pipeline_asset] = store

    result = {}
    # Gets artifacts from each external store.
    for pipeline_asset, external_ids in external_ids_by_pipeline_asset.items():
      store = store_by_pipeline_asset[pipeline_asset]
      external_id_by_id = {
          external_artifact_utils.get_id_from_external_id(e): e
          for e in external_ids
      }
      artifacts_by_artifact_ids_fn = (
          self.get_downstream_artifacts_by_artifact_ids
          if downstream
          else self.get_upstream_artifacts_by_artifact_ids
      )
      artifacts_and_types_by_artifact_id = artifacts_by_artifact_ids_fn(
          list(external_id_by_id.keys()),
          max_num_hops,
          filter_query,
          event_filter,
          store,
      )

      pipeline_owner = pipeline_asset.split('/')[0]
      pipeline_name = pipeline_asset.split('/')[1]
      artifacts_by_external_id = {}
      for (
          artifact_id,
          artifacts_and_types,
      ) in artifacts_and_types_by_artifact_id.items():
        external_id = external_id_by_id[artifact_id]
        imported_artifacts_and_types = []
        for a, t in artifacts_and_types:
          imported_artifact = external_artifact_utils.cold_import_artifacts(
              t, [a], pipeline_owner, pipeline_name
          )[0]
          imported_artifacts_and_types.append(
              (imported_artifact.mlmd_artifact, imported_artifact.artifact_type)
          )
        artifacts_by_external_id[external_id] = imported_artifacts_and_types

      result.update(artifacts_by_external_id)

    return result

  def get_downstream_artifacts_by_artifacts(
      self,
      artifacts: List[metadata_store_pb2.Artifact],
      max_num_hops: int = _MAX_NUM_HOPS,
      filter_query: str = '',
      event_filter: Optional[Callable[[metadata_store_pb2.Event], bool]] = None,
  ) -> Dict[
      Union[str, int],
      List[Tuple[metadata_store_pb2.Artifact, metadata_store_pb2.ArtifactType]],
  ]:
    """Given a list of artifacts, get their provenance successor artifacts.

    For each provided artifact, treat it as a starting
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
        artifacts: a list of starting artifacts. At most 100 ids are supported.
          Returns empty result if `artifacts` is empty.
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
    if not artifacts:
      return {}

    # Precondition check.
    if len(artifacts) > _MAX_NUM_STARTING_NODES:
      raise ValueError(
          'Number of artifacts is larger than supported value of %d.'
          % _MAX_NUM_STARTING_NODES
      )
    if max_num_hops > _MAX_NUM_HOPS or max_num_hops < 0:
      raise ValueError(
          'Number of hops %d is larger than supported value of %d or is'
          ' negative.' % (max_num_hops, _MAX_NUM_HOPS)
      )

    internal_artifact_ids = [a.id for a in artifacts if not a.external_id]
    external_artifact_ids = [a.external_id for a in artifacts if a.external_id]
    if internal_artifact_ids and external_artifact_ids:
      raise ValueError(
          'Provided artifacts contain both internal and external artifacts. It'
          ' is not supported.'
      )

    if not external_artifact_ids:
      return self.get_downstream_artifacts_by_artifact_ids(
          internal_artifact_ids, max_num_hops, filter_query, event_filter
      )

    return self._get_external_upstream_or_downstream_artifacts(
        external_artifact_ids,
        max_num_hops,
        filter_query,
        event_filter,
        downstream=True,
    )

  def get_downstream_artifacts_by_artifact_ids(
      self,
      artifact_ids: List[int],
      max_num_hops: int = _MAX_NUM_HOPS,
      filter_query: str = '',
      event_filter: Optional[Callable[[metadata_store_pb2.Event], bool]] = None,
      store: Optional[mlmd.MetadataStore] = None,
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
        store: A metadata_store.MetadataStore instance.

    Returns:
    Mapping of artifact ids to a list of downstream artifacts.
    """
    # Precondition check.
    if not artifact_ids:
      return {}

    if len(artifact_ids) > _MAX_NUM_STARTING_NODES:
      raise ValueError(
          'Number of artifact ids is larger than supported value of %d.'
          % _MAX_NUM_STARTING_NODES
      )
    if max_num_hops > _MAX_NUM_HOPS or max_num_hops < 0:
      raise ValueError(
          'Number of hops %d is larger than supported value of %d or is'
          ' negative.' % (max_num_hops, _MAX_NUM_HOPS)
      )

    if store is None:
      store = self._store
    if store is None:
      raise ValueError('MetadataStore provided to MetadataResolver is None.')

    artifact_ids_str = ','.join(str(id) for id in artifact_ids)
    # If `max_num_hops` is set to 0, we don't need the graph traversal.
    if max_num_hops == 0:
      if not filter_query:
        artifacts = store.get_artifacts_by_id(artifact_ids)
      else:
        artifacts = self._get_filtered_artifacts(
            artifact_ids, filter_query=filter_query, limit=_MAX_NUM_STARTING_NODES
        )
      artifact_type_ids = [a.type_id for a in artifacts]
      artifact_types = store.get_artifact_types_by_id(artifact_type_ids)
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
    lineage_graph = self._get_lineage_subgraph(
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
      # Send a call to metadata_store to get filtered downstream artifacts.
      artifacts = self._get_filtered_artifacts(
          list(candidate_artifact_ids), filter_query=filter_query
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
    lineage_graph = self._get_lineage_subgraph(
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

  def get_upstream_artifacts_by_artifacts(
      self,
      artifacts: List[metadata_store_pb2.Artifact],
      max_num_hops: int = _MAX_NUM_HOPS,
      filter_query: str = '',
      event_filter: Optional[Callable[[metadata_store_pb2.Event], bool]] = None,
  ) -> Dict[
      Union[str, int],
      List[Tuple[metadata_store_pb2.Artifact, metadata_store_pb2.ArtifactType]],
  ]:
    """Given a list of artifacts, get their provenance ancestor artifacts.

    For each provided artifact, treat it as a starting
    artifact and get artifacts that are connected to them within `max_num_hops`
    via a path in the upstream direction like:
    artifact_i -> INPUT_event -> execution_j -> OUTPUT_event -> artifact_k.

    A hop is defined as a jump to the next node following the path of node
    -> event -> next_node.
    For example, in the lineage graph artifact_1 -> event -> execution_1
    -> event -> artifact_2:
    artifact_2 is 2 hops away from artifact_1, and execution_1 is 1 hop away
    from artifact_1.

    Args:
        artifacts: a list of starting artifacts. At most 100 ids are supported.
          Returns empty result if `artifacts` is empty.
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
    if not artifacts:
      return {}

    # Precondition check.
    if len(artifacts) > _MAX_NUM_STARTING_NODES:
      raise ValueError(
          'Number of artifacts is larger than supported value of %d.'
          % _MAX_NUM_STARTING_NODES
      )
    if max_num_hops > _MAX_NUM_HOPS or max_num_hops < 0:
      raise ValueError(
          'Number of hops %d is larger than supported value of %d or is'
          ' negative.' % (max_num_hops, _MAX_NUM_HOPS)
      )

    internal_artifact_ids = [a.id for a in artifacts if not a.external_id]
    external_artifact_ids = [a.external_id for a in artifacts if a.external_id]
    if internal_artifact_ids and external_artifact_ids:
      raise ValueError(
          'Provided artifacts contain both internal and external artifacts. It'
          ' is not supported.'
      )

    if not external_artifact_ids:
      return self.get_upstream_artifacts_by_artifact_ids(
          internal_artifact_ids, max_num_hops, filter_query, event_filter
      )

    return self._get_external_upstream_or_downstream_artifacts(
        external_artifact_ids,
        max_num_hops,
        filter_query,
        event_filter,
        downstream=False,
    )

  def get_upstream_artifacts_by_artifact_ids(
      self,
      artifact_ids: List[int],
      max_num_hops: int = _MAX_NUM_HOPS,
      filter_query: str = '',
      event_filter: Optional[Callable[[metadata_store_pb2.Event], bool]] = None,
      store: Optional[mlmd.MetadataStore] = None,
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
        store: A metadata_store.MetadataStore instance.

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

    if store is None:
      store = self._store
    if store is None:
      raise ValueError('MetadataStore provided to MetadataResolver is None.')

    artifact_ids_str = ','.join(str(id) for id in artifact_ids)
    # If `max_num_hops` is set to 0, we don't need the graph traversal.
    if max_num_hops == 0:
      if not filter_query:
        artifacts = store.get_artifacts_by_id(artifact_ids)
      else:
        artifacts = self._get_filtered_artifacts(
            artifact_ids, filter_query=filter_query, limit=_MAX_NUM_STARTING_NODES
        )
      artifact_type_ids = [a.type_id for a in artifacts]
      artifact_types = store.get_artifact_types_by_id(artifact_type_ids)
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
    lineage_graph = self._get_lineage_subgraph(
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
      # Send a call to metadata_store to get filtered upstream artifacts.
      artifacts = self._get_filtered_artifacts(
          list(candidate_artifact_ids), filter_query=filter_query
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
    lineage_graph = self._get_lineage_subgraph(
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
