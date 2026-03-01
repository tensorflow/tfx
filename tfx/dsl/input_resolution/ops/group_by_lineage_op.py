# Copyright 2023 Google LLC. All Rights Reserved.
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
"""Module for GroupByDisjointLineage operator."""

import collections
from typing import List, Iterable, Tuple

from tfx.dsl.input_resolution import resolver_op
from tfx.orchestration.portable.input_resolution import exceptions
from tfx.utils import typing_utils

from ml_metadata.proto import metadata_store_pb2


def _get_neighbor_artifact_pairs(
    events: List[metadata_store_pb2.Event],
) -> Iterable[Tuple[int, int]]:
  """Gets artifact_id pair of neighbors from the list of Events.

  Artifact a and b is considered neighbor if there exist events e1 and e2 s.t.
  (e1.artifact_id = a) AND (e2.artifact_id = b) AND
  (e1.execution_id = e2.execution_id)

  Args:
    events: A list of MLMD Events.

  Yields:
    Edge as a tuple (artifact_id_1, artifact_id_2).
  """
  execs_by_art = collections.defaultdict(set)
  arts_by_exec = collections.defaultdict(set)
  for event in events:
    execs_by_art[event.artifact_id].add(event.execution_id)
    arts_by_exec[event.execution_id].add(event.artifact_id)
  for a1 in execs_by_art:
    for a2 in set.union(*[arts_by_exec[e] for e in execs_by_art[a1]]):
      if a1 < a2:  # Skip symmetric or self edge.
        yield a1, a2


def _find_disjoint_sets(
    verts: Iterable[int], edges: Iterable[Tuple[int, int]]
) -> List[List[int]]:
  """Finds disjoint sets."""
  parents = {a: a for a in verts}

  def find(a: int):
    if parents[a] != a:
      parents[a] = find(parents[a])
    return parents[a]

  def union(a: int, b: int):
    x, y = find(a), find(b)
    if x != y:
      # Union in a direction that smaller number node becomes the parent node.
      # By result, the root node of each disjoint set will be the one with the
      # smallest number.
      parents[max(x, y)] = min(x, y)

  for a, b in edges:
    union(a, b)

  # Python dict "order is guaranteed to be insertion order" from python 3.7
  # (https://docs.python.org/3/library/stdtypes.html#dict).
  # As it loops over the sorted node number, and since the root node of each
  # disjoint set is the one with the smallest node number, both the inner and
  # the outer lists of the result would be sorted.
  disjoint_sets = {}
  for a in sorted(verts):
    disjoint_sets.setdefault(find(a), []).append(a)
  return list(disjoint_sets.values())


class GroupByDisjointLineage(
    resolver_op.ResolverOp,
    canonical_name='tfx.GroupByDisjointLineage',
    arg_data_types=(resolver_op.DataType.ARTIFACT_MULTIMAP,),
    return_data_type=resolver_op.DataType.ARTIFACT_MULTIMAP_LIST,
):
  """GroupByDisjointLineage operator.

  Let's say we have a lineage of artifacts (executions omitted for brevity):

  ```dot
  digraph {
      a1 -> b1 -> c1
      a2 -> b2 -> c2
      a2 -> b3 -> c3
      a3 -> b4 -> c4
  }
  ```

  Then `GroupByDisjointLineage` would group artifacts by each disjoint lineage
  where artifacts from different group is not reachable.

  ```python
  GroupByDisjointLineage({
      'a': [a1, a2, a3],
      'b': [b1, b2, b3, b4],
      'c': [c1, c2, c3, c4],
  }) == [
      {'a': [a1], 'b': [b1], 'c': [c1]},
      {'a': [a2], 'b': [b2, b3], 'c': [c2, c3]},
      {'a': [a3], 'b': [b4], 'c': [c4]},
  ]
  ```

  CAVEAT: Lineage is only searched for the 2-hop distances (i.e. artifact ->
  execution -> artifact), so in order to traverse for the deeper relationships,
  provide the intermediate artifacts as well so that there exists a chain of
  2-hop connections.
  """

  # If require_all is True, then any dictionary from the result that contains
  # empty list would be dropped. In other words, at least 1 artifact should be
  # present from each key of each result dictionary.
  require_all = resolver_op.Property(type=bool, default=False)

  def apply(
      self, artifact_map: typing_utils.ArtifactMultiMap
  ) -> List[typing_utils.ArtifactMultiDict]:
    artifacts_by_id = {}
    input_keys_by_id = collections.defaultdict(set)
    for input_key, artifacts in artifact_map.items():
      for a in artifacts:
        input_keys_by_id[a.id].add(input_key)
        artifacts_by_id[a.id] = a

    if not artifacts_by_id:
      return []

    events = self.context.store.get_events_by_artifact_ids(
        artifact_ids=artifacts_by_id
    )

    result = []
    for disjoint_set in _find_disjoint_sets(
        artifacts_by_id, _get_neighbor_artifact_pairs(events)
    ):
      result_item = {input_key: [] for input_key in artifact_map}
      for artifact_id in disjoint_set:
        for input_key in input_keys_by_id[artifact_id]:
          result_item[input_key].append(artifacts_by_id[artifact_id])
      if not self.require_all or all(result_item.values()):
        result.append(result_item)
    return result


class GroupByPivot(
    resolver_op.ResolverOp,
    canonical_name='tfx.GroupByPivot',
    arg_data_types=(resolver_op.DataType.ARTIFACT_MULTIMAP,),
    return_data_type=resolver_op.DataType.ARTIFACT_MULTIMAP_LIST,
):
  """GroupByPivot operator.

  Let's say we have a lineage of artifacts (executions omitted for brevity):

  ```dot
  digraph {
      a1 -> b1 -> c1
      a2 -> b2 -> c2
      a2 -> b3 -> c3
      a3 -> b4 -> c4
  }
  ```

  Then `GroupByPivot` would group artifacts by each pivot artifact from
  the input artifacts where artifacts in the same group is reachable from the
  pivot.

  ```python
  inputs = {
      'a': [a1, a2, a3],
      'b': [b1, b2, b3, b4],
      'c': [c1, c2, c3, c4],
  }
  # 'c' is empty they are not adjacent provenance from 'a'.
  GroupByPivot(inputs, pivot_key='a') == [
      {'a': [a1], 'b': [b1], 'c': []},
      {'a': [a2], 'b': [b2, b3], 'c': []},
      {'a': [a3], 'b': [b4], 'c': []},
  ]
  # Both 'a' and 'c' is not empty as they are adjacent provenance from 'b'.
  GroupByPivot(inputs, pivot_key='b') == [
      {'a': [a1], 'b': [b1], 'c': [c1]},
      {'a': [a2], 'b': [b2], 'c': [c2]},
      {'a': [a2], 'b': [b3], 'c': [c3]},
      {'a': [a3], 'b': [b4], 'c': [c4]},
  ]
  ```

  The result of the operator is a list of dictionary, where each dictionary
  contains individual pivot artifact. Non-pivot artifacts could be included in
  multiple dictionaries if they are associated with multiple pivots, or some
  dictionary might have empty artifact list for non-pivot artifacts if adjacent
  provenances are not found.

  CAVEAT: Lineage is only searched for the 2-hop distances (i.e. artifact ->
  execution -> artifact) and the artifacts farther than 2 hops from the pivot
  artifacts would NOT be included in the result.
  """
  # Input key that is used for a pivot.
  pivot_key = resolver_op.Property(type=str)

  # If require_all is True, then any dictionary from the result that contains
  # empty list would be dropped. In other words, at least 1 artifact should be
  # present from each key of each result dictionary.
  require_all = resolver_op.Property(type=bool, default=False)

  def apply(
      self, artifact_map: typing_utils.ArtifactMultiMap
  ) -> List[typing_utils.ArtifactMultiDict]:
    if self.pivot_key not in artifact_map:
      raise exceptions.FailedPreconditionError(
          f'Pivot "{self.pivot_key}" does not exist in the artifact map. '
          f'Containing keys: {list(artifact_map.keys())}'
      )
    if not artifact_map[self.pivot_key]:
      return []

    artifacts_by_id = {}
    input_keys_by_id = collections.defaultdict(set)
    for input_key, artifacts in artifact_map.items():
      for a in artifacts:
        input_keys_by_id[a.id].add(input_key)
        artifacts_by_id[a.id] = a

    events = self.context.store.get_events_by_artifact_ids(
        artifact_ids=artifacts_by_id
    )

    neighbors = collections.defaultdict(set)
    for a, b in _get_neighbor_artifact_pairs(events):
      neighbors[a].add(b)
      neighbors[b].add(a)

    result = []
    # Preserve the initial order in artifact_map[pivot_key].
    for pivot in artifact_map[self.pivot_key]:
      result_item = {input_key: [] for input_key in artifact_map}
      result_item[self.pivot_key].append(pivot)
      # Sort for deterministic result.
      for artifact_id in sorted(neighbors[pivot.id]):
        for input_key in input_keys_by_id[artifact_id]:
          result_item[input_key].append(artifacts_by_id[artifact_id])
      if not self.require_all or all(result_item.values()):
        result.append(result_item)
    return result
