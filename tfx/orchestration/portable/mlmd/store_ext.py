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
from typing import Optional, List, Sequence, Union

from tfx.orchestration.portable.mlmd import filter_query_builder as q

import ml_metadata as mlmd
from ml_metadata.proto import metadata_store_pb2


_Metadata = Union[
    metadata_store_pb2.Artifact,
    metadata_store_pb2.Execution,
    metadata_store_pb2.Context,
]


def _ids(values: Sequence[_Metadata]) -> Sequence[int]:
  return [v.id for v in values]


def _maybe_clause(clause: Optional[str]) -> List[str]:
  return [clause] if clause is not None else []


def get_successful_node_executions(
    store: mlmd.MetadataStore,
    *,
    pipeline_id: str,
    node_id: str,
) -> List[metadata_store_pb2.Execution]:
  """Gets all successful node executions."""
  node_executions_query = q.And([
      'contexts_0.type = "node"',
      f'contexts_0.name = "{pipeline_id}.{node_id}"',
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
    artifact_filter: Optional[str] = None,
) -> List[metadata_store_pb2.Artifact]:
  """Gets artifacts associated with OUTPUT events from execution IDs.

  Args:
    store: A MetadataStore object.
    execution_ids: A list of Execution IDs.
    artifact_filter: Optional MLMD filter query to apply for the artifact query,
      e.g. "state = LIVE".

  Returns:
    A list of artifacts.
  """
  artifact_query = q.And([
      q.Or([
          'events_0.type = OUTPUT',
          'events_0.type = INTERNAL_OUTPUT',
          'events_0.type = DECLARED_OUTPUT',
      ]),
      q.Or(
          [
              f'events_0.execution_id = {execution_id}'
              for execution_id in execution_ids
          ]
      ),
      *_maybe_clause(artifact_filter),
  ])
  return store.get_artifacts(list_options=artifact_query.list_options())


def get_live_output_artifacts_of_node(
    store: mlmd.MetadataStore,
    *,
    pipeline_id: str,
    node_id: str,
) -> List[metadata_store_pb2.Artifact]:
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
      artifact_filter='state = LIVE',
  )
