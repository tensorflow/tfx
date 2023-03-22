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
from typing import Optional, List, Sequence, Union, Callable

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


def get_successful_node_executions(
    store: mlmd.MetadataStore,
    *,
    pipeline_id: str,
    node_id: str,
) -> List[mlmd.proto.Execution]:
  """Gets all successful node executions."""
  node_context_name = compiler_utils.node_context_name(pipeline_id, node_id)
  node_executions_query = q.And([
      f'contexts_0.type = "{constants.NODE_CONTEXT_TYPE_NAME}"',
      f'contexts_0.name = "{node_context_name}"',
      q.Or([
          'last_known_state = COMPLETE',
          'last_known_state = CACHED',
      ]),
  ])
  return store.get_executions(list_options=node_executions_query.list_options())


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
  node_executions = get_successful_node_executions(
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
