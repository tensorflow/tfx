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
"""Module for MetadataStore extension functions.

All public functions should accepts the first parameter of MetadataStore.
"""
import collections
from typing import Optional, List, Sequence, Union, Callable, Dict

from tfx.dsl.compiler import compiler_utils
from tfx.dsl.compiler import constants
from tfx.orchestration.portable.mlmd import event_lib
from tfx.orchestration.portable.mlmd import filter_query_builder as q

import ml_metadata as mlmd


_Metadata = Union[mlmd.proto.Artifact, mlmd.proto.Execution, mlmd.proto.Context]
_ArtifactState = mlmd.proto.Artifact.State
_ArtifactPredicate = Callable[[mlmd.proto.Artifact], bool]


def _ids(values: Sequence[_Metadata]) -> Sequence[int]:
  return [v.id for v in values]


def _maybe_clause(clause: Optional[str]) -> List[str]:
  return [clause] if clause is not None else []


def get_node_executions(
    store: mlmd.MetadataStore,
    *,
    pipeline_id: str,
    node_id: str,
    pipeline_run_id: Optional[str] = None,
    order_by: mlmd.OrderByField = mlmd.OrderByField.ID,
    is_asc: bool = True,
    limit: Optional[int] = None,
    execution_states: Optional[List['mlmd.proto.Execution.State']] = None,
) -> List[mlmd.proto.Execution]:
  """Gets all successful node executions."""
  if not execution_states:
    execution_states = [
        mlmd.proto.Execution.COMPLETE,
        mlmd.proto.Execution.CACHED,
    ]
  node_context_name = compiler_utils.node_context_name(pipeline_id, node_id)
  state_query = [
      f'last_known_state = {mlmd.proto.Execution.State.Name(s)}'
      for s in execution_states
  ]
  node_executions_query = q.And([
      f'contexts_0.type = "{constants.NODE_CONTEXT_TYPE_NAME}"',
      f'contexts_0.name = "{node_context_name}"',
      q.Or(state_query),
  ])
  if pipeline_run_id:
    node_executions_query.append(
        q.And([
            f'contexts_1.type = "{constants.PIPELINE_RUN_CONTEXT_TYPE_NAME}"',
            f'contexts_1.name = "{pipeline_run_id}"',
        ])
    )
  return store.get_executions(
      list_options=mlmd.ListOptions(
          filter_query=str(node_executions_query),
          order_by=order_by,
          is_asc=is_asc,
          limit=limit,
      )
  )


def get_output_artifacts_from_execution_ids(
    store: mlmd.MetadataStore,
    *,
    execution_ids: Sequence[int],
    artifact_filter: Optional[_ArtifactPredicate] = None,
) -> List[mlmd.proto.Artifact]:
  """Gets artifacts associated with OUTPUT events from execution IDs.

  Args:
    store: A MetadataStore object.
    execution_ids: A list of Execution IDs.
    artifact_filter: Optional artifact predicate to apply.

  Returns:
    A list of output artifacts of the given executions.
  """
  events = store.get_events_by_execution_ids(execution_ids)
  output_events = [e for e in events if event_lib.is_valid_output_event(e)]
  artifact_ids = {e.artifact_id for e in output_events}
  result = store.get_artifacts_by_id(artifact_ids)
  if artifact_filter is not None:
    result = [a for a in result if artifact_filter(a)]
  return result


def get_live_output_artifacts_of_node(
    store: mlmd.MetadataStore,
    *,
    pipeline_id: str,
    node_id: str,
) -> List[mlmd.proto.Artifact]:
  """Get LIVE output artifacts of the given node.

  This is a 2-hop query with 2 MLMD API calls:
  1. get_executions() to get node executions.
  3. get_artifacts() to get associated output artifacts.

  Args:
    store: A MetadataStore object.
    pipeline_id: A pipeline ID.
    node_id: A node ID.

  Returns:
    A list of output artifacts from the given node.
  """
  # First query: Get all successful executions of the node.
  node_executions = get_node_executions(
      store,
      pipeline_id=pipeline_id,
      node_id=node_id,
  )
  if not node_executions:
    return []

  # Second query: Get all output artifacts associated with the executions from
  # the first query.
  return get_output_artifacts_from_execution_ids(
      store,
      execution_ids=_ids(node_executions),
      artifact_filter=lambda a: a.state == mlmd.proto.Artifact.LIVE,
  )


def get_live_output_artifacts_of_node_by_output_key(
    store: mlmd.MetadataStore,
    *,
    pipeline_id: str,
    node_id: str,
    pipeline_run_id: Optional[str] = None,
    execution_limit: Optional[int] = None,
    execution_states: Optional[List['mlmd.proto.Execution.State']] = None,
) -> Dict[str, List[List[mlmd.proto.Artifact]]]:
  """Get LIVE output artifacts of the given node grouped by output key.

  The LIVE output artifacts associated with an output key are represented as a
  list of a list of artifacts.
  1. The outer list represents artifacts generated across all executions.
  2. The inner list represents artifacts generated by one execution.
  3. Elements in the outer list are returned in descending order of the creation
  time of the execution associated with them.
  4. Elements in the inner list have no order guarantee.
  5. If no LIVE output artifacts found for one execution, an empty list will be
  returned.

  The value of execution_limit must be None or non-negative.
  1. If None or 0, live output artifacts from all executions will be returned.
  2. If the node has fewer executions than execution_limit, live output
     artifacts from all executions will be returned.
  3. If the node has more or equal executions than execution_limit, only live
     output artifacts from the execution_limit latest executions will be
     returned.

  Args:
    store: A MetadataStore object.
    pipeline_id: A pipeline ID.
    node_id: A node ID.
    pipeline_run_id: The pipeline run ID that the node belongs to. Only
      artifacts from the specified pipeline run are returned if specified.
    execution_limit: Maximum number of latest executions from which live output
      artifacts will be returned.
    execution_states: The MLMD execution state(s) to pull LIVE artifacts from.
      Defaults to [COMPLETE, CACHED].

  Returns:
    A mapping from output key to all output artifacts from the given node.
  """
  node_executions_ordered_by_desc_creation_time = get_node_executions(
      store,
      pipeline_id=pipeline_id,
      node_id=node_id,
      pipeline_run_id=pipeline_run_id,
      order_by=mlmd.OrderByField.CREATE_TIME,
      is_asc=False,
      limit=execution_limit,
      execution_states=execution_states,
  )
  if not node_executions_ordered_by_desc_creation_time:
    return {}

  all_events = store.get_events_by_execution_ids(
      _ids(node_executions_ordered_by_desc_creation_time)
  )
  output_events = [e for e in all_events if event_lib.is_valid_output_event(e)]

  output_artifact_ids = [e.artifact_id for e in output_events]
  output_artifacts = store.get_artifacts_by_id(output_artifact_ids)

  # Create a mapping from exec_id to an empty list first to make sure iteration
  # orders of events_by_exec_id and output_artifacts_map_by_exec_id are
  # both in desc order of execution's creation time.
  # The desc order is guaranteed by execution_ids and dict is guaranteed to be
  # iterated in the insertion order of keys.
  events_by_exec_id = {
      exec.id: [] for exec in node_executions_ordered_by_desc_creation_time
  }
  for e in output_events:
    events_by_exec_id[e.execution_id].append(e)

  output_artifacts_map_by_exec_id = {}
  for exec_id, events in events_by_exec_id.items():
    output_artifacts_map = event_lib.reconstruct_artifact_multimap(
        output_artifacts, events
    )
    output_artifacts_map_by_exec_id[exec_id] = output_artifacts_map

  output_artifacts_by_output_key = collections.defaultdict(list)
  for (
      exec_id,
      output_artifacts_map,
  ) in output_artifacts_map_by_exec_id.items():
    for output_key, artifact_list in output_artifacts_map.items():
      live_output_artifacts = [
          a for a in artifact_list if a.state == mlmd.proto.Artifact.LIVE
      ]
      output_artifacts_by_output_key[output_key].append(live_output_artifacts)
  return output_artifacts_by_output_key
