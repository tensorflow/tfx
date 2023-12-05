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
import itertools
from typing import Callable, Dict, List, Optional, Sequence, Union

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


def _get_node_live_artifacts(
    store: mlmd.MetadataStore,
    *,
    pipeline_id: str,
    node_id: str,
    pipeline_run_id: Optional[str] = None,
) -> List[mlmd.proto.Artifact]:
  """Gets all LIVE node artifacts.

  Args:
    store: A MetadataStore object.
    pipeline_id: The pipeline ID.
    node_id: The node ID.
    pipeline_run_id: The pipeline run ID that the node belongs to. Only
      artifacts from the specified pipeline run are returned if specified.

  Returns:
    A list of LIVE artifacts of the given pipeline node.
  """
  artifact_state_filter_query = (
      f'state = {mlmd.proto.Artifact.State.Name(mlmd.proto.Artifact.LIVE)}'
  )
  node_context_name = compiler_utils.node_context_name(pipeline_id, node_id)
  node_filter_query = q.And([
      f'contexts_0.type = "{constants.NODE_CONTEXT_TYPE_NAME}"',
      f'contexts_0.name = "{node_context_name}"',
  ])

  artifact_filter_query = q.And([
      node_filter_query,
      artifact_state_filter_query,
  ])

  if pipeline_run_id:
    artifact_filter_query.append(
        q.And([
            f'contexts_1.type = "{constants.PIPELINE_RUN_CONTEXT_TYPE_NAME}"',
            f'contexts_1.name = "{pipeline_run_id}"',
        ])
    )

  return store.get_artifacts(
      list_options=mlmd.ListOptions(filter_query=str(artifact_filter_query))
  )


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
    min_last_update_time_since_epoch: Optional[int] = None,
) -> List[mlmd.proto.Execution]:
  """Gets all successful node executions."""
  # TODO(b/301507304): Relax constraint on execution states:
  # If `execution_states` is unspecified or empty, the query should consider all
  # execution states.
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
  if min_last_update_time_since_epoch:
    node_executions_query.append(
        f'last_update_time_since_epoch >= {min_last_update_time_since_epoch}'
    )
  return store.get_executions(
      list_options=mlmd.ListOptions(
          filter_query=str(node_executions_query),
          order_by=order_by,
          is_asc=is_asc,
          limit=limit,
      )
  )


# TODO(b/301507304): Integrate this function into garbage_collection module.
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
  # Step 1: Get LIVE artifacts attributed to node with `node_id`.
  live_artifacts = _get_node_live_artifacts(
      store,
      pipeline_id=pipeline_id,
      node_id=node_id,
      pipeline_run_id=pipeline_run_id,
  )
  if not live_artifacts:
    return {}

  # Step 2: Get executions associated with node that created `live_artifacts`
  # ordered by execution creation time in descending order.
  # These executions should satisfy the constraint:
  # min (execution update time) >= min (artifact create time)
  min_live_artifact_create_time = min(
      [a.create_time_since_epoch for a in live_artifacts], default=0
  )

  # Within one transaction that updates both artifacts and execution, the
  # timestamp of execution is larger or equal than that of the artifacts.
  # Apply time skew for the artifacts created before cl/574333630 is rolled out.
  # TODO(b/275231956): Remove the following 2 lines if we are sure that there
  # are no more artifacts older than the timestamp.
  if min_live_artifact_create_time < 1700985600000:  # Nov 26, 2023 12:00:00 AM
    min_live_artifact_create_time -= 24 * 3600 * 1000

  executions_ordered_by_desc_creation_time = get_node_executions(
      store,
      pipeline_id=pipeline_id,
      node_id=node_id,
      pipeline_run_id=pipeline_run_id,
      order_by=mlmd.OrderByField.CREATE_TIME,
      is_asc=False,
      limit=execution_limit,
      execution_states=execution_states,
      min_last_update_time_since_epoch=min_live_artifact_create_time,
  )
  if not executions_ordered_by_desc_creation_time:
    return {}

  # Step 3: Get output events by executions obtained in step 2.
  events_by_executions = store.get_events_by_execution_ids(
      _ids(executions_ordered_by_desc_creation_time)
  )
  output_events = [
      e for e in events_by_executions if event_lib.is_valid_output_event(e)
  ]

  # Step 4: Construct and return `output_artifacts_by_output_key` from events.
  #
  # Create a mapping from execution_id to an empty list first to make sure
  # iteration orders of output_events_by_execution_id and
  # output_artifacts_map_by_execution_id are both in desc order of execution's
  # creation time.
  #
  # The desc order is guaranteed by execution_ids and dict is guaranteed to be
  # iterated in the insertion order of keys.
  output_events_by_execution_id = {
      execution.id: [] for execution in executions_ordered_by_desc_creation_time
  }
  for event in output_events:
    output_events_by_execution_id[event.execution_id].append(event)

  artifact_ids_by_output_key_map_by_execution_id = {}
  for exec_id, events in output_events_by_execution_id.items():
    output_artifacts_map = event_lib.reconstruct_artifact_id_multimap(events)
    artifact_ids_by_output_key_map_by_execution_id[exec_id] = (
        output_artifacts_map
    )

  output_artifacts_by_output_key = collections.defaultdict(list)

  # Keep only LIVE output artifacts when constructing the result.
  live_artifacts_by_id = {a.id: a for a in live_artifacts}
  for (
      artifact_ids_by_output_key
  ) in artifact_ids_by_output_key_map_by_execution_id.values():
    for output_key, artifact_ids in artifact_ids_by_output_key.items():
      live_output_artifacts = [
          live_artifacts_by_id[artifact_id]
          for artifact_id in artifact_ids
          if artifact_id in live_artifacts_by_id
      ]
      output_artifacts_by_output_key[output_key].append(live_output_artifacts)
  return output_artifacts_by_output_key


def get_live_output_artifacts_of_node(
    store: mlmd.MetadataStore,
    *,
    pipeline_id: str,
    node_id: str,
) -> List[mlmd.proto.Artifact]:
  """Gets LIVE output artifacts of the given node.

  The function query is composed of 3 MLMD API calls:
  1. Call get_artifacts() to get LIVE artifacts attributed to the given node.
  2. Call get_executions() to get executions that created artifacts from step 1.
  3. Call get_events_by_execution_ids() and filter artifacts on whether they are
  output artifacts of executions from step 2.

  Args:
    store: A MetadataStore object.
    pipeline_id: A pipeline ID.
    node_id: A node ID.

  Returns:
    A list of output artifacts from the given node.
  """
  live_output_artifacts_of_node_by_output_key = (
      get_live_output_artifacts_of_node_by_output_key(
          store, pipeline_id=pipeline_id, node_id=node_id
      )
  )
  live_output_artifacts = list()
  for (
      nested_artifact_lists
  ) in live_output_artifacts_of_node_by_output_key.values():
    live_output_artifacts.extend(
        itertools.chain.from_iterable(nested_artifact_lists)
    )
  return live_output_artifacts
