# Copyright 2022 Google LLC. All Rights Reserved.
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
"""Module for InputSpec.Channel resolution."""

from typing import Sequence, List, Optional, Tuple

from absl import logging
from tfx import types
from tfx.orchestration import metadata
from tfx.orchestration.portable.mlmd import event_lib
from tfx.orchestration.portable.mlmd import execution_lib
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import artifact_utils

import ml_metadata as mlmd
from ml_metadata import errors
from ml_metadata.proto import metadata_store_pb2


# TODO(b/233044350): Move to a general metadata utility.
def _get_executions_by_all_contexts(
    store: mlmd.MetadataStore,
    contexts: Sequence[metadata_store_pb2.Context],
) -> List[metadata_store_pb2.Execution]:
  """Get executions bound with ALL given contexts."""
  if not contexts:
    return []
  executions_by_id = {}
  valid_ids = None
  for context in contexts:
    executions = store.get_executions_by_context(context.id)
    if valid_ids is None:
      valid_ids = {a.id for a in executions}
    else:
      valid_ids.intersection_update({a.id for a in executions})
    executions_by_id.update({a.id: a for a in executions})
  return [executions_by_id[i] for i in valid_ids]


def _get_context_from_context_query(
    store: mlmd.MetadataStore,
    context_query: pipeline_pb2.InputSpec.Channel.ContextQuery
) -> Optional[metadata_store_pb2.Context]:
  """Get Context from InputSpec.Channel.ContextQuery."""
  if not context_query.HasField('type') or not context_query.type.name:
    raise ValueError('ContextQuery.type.name should be set.')
  if (not context_query.name or
      not context_query.name.field_value or
      not context_query.name.field_value.string_value):
    raise ValueError(
        'ContextQuery.name.field_value.string_value should be set.')
  if context_query.HasField('property_predicate'):
    logging.warning('ContextQuery.property_predicate is not supported.')
  return store.get_context_by_type_and_name(
      context_query.type.name,
      context_query.name.field_value.string_value)


# TODO(b/233044350): Move to a general metadata utility.
def _get_output_artifacts(
    store: mlmd.MetadataStore,
    executions: Sequence[metadata_store_pb2.Execution],
    output_key: Optional[str] = None,
) -> List[metadata_store_pb2.Artifact]:
  """Get all output artifacts of the given executions."""
  executions_ids = [
      e.id for e in executions
      if execution_lib.is_execution_successful(e)
  ]
  events = [
      v for v in store.get_events_by_execution_ids(executions_ids)
      if event_lib.is_valid_output_event(v, output_key)
  ]
  artifact_ids = [v.artifact_id for v in events]
  return store.get_artifacts_by_id(artifact_ids)


def _filter_by_artifact_query(
    store: mlmd.MetadataStore,
    artifacts: Sequence[metadata_store_pb2.Artifact],
    artifact_query: pipeline_pb2.InputSpec.Channel.ArtifactQuery,
    live_only: bool = True,
) -> Tuple[List[metadata_store_pb2.Artifact], metadata_store_pb2.ArtifactType]:
  """Filters initial artifacts with given ArtifactQuery."""
  predicates = []
  if not artifact_query.type or not artifact_query.type.name:
    raise ValueError('ArtifactQuery.type.name should be set.')
  artifact_type = artifact_query.type
  if not artifact_type.id:
    try:
      artifact_type = store.get_artifact_type(artifact_query.type.name)
    except errors.NotFoundError:
      # If artifac type does not exist, it means no artifacts satisfy the
      # artifact query. Returns empty.
      return [], artifact_type
  predicates.append(lambda a: a.type_id == artifact_type.id)
  if artifact_query.property_predicate:
    logging.warning('ArtifactQuery.property_predicate is not supported.')
  if live_only:
    predicates.append(lambda a: a.state == metadata_store_pb2.Artifact.LIVE)
  return (
      [a for a in artifacts if all(p(a) for p in predicates)],
      artifact_type,
  )


# TODO(b/234806996): Migrate to MLMD filter query.
def resolve_single_channel(
    mlmd_handle: metadata.Metadata,
    channel: pipeline_pb2.InputSpec.Channel,
) -> List[types.Artifact]:
  """Evaluate a single InputSpec.Channel."""
  store = mlmd_handle.store
  contexts = []
  for context_query in channel.context_queries:
    maybe_context = _get_context_from_context_query(store, context_query)
    if maybe_context is None:
      # If the context does not exist, it means no artifacts satisfy the given
      # context query. Returns empty.
      return []
    else:
      contexts.append(maybe_context)
  executions = _get_executions_by_all_contexts(store, contexts)
  if not executions:
    return []
  artifacts = _get_output_artifacts(store, executions, channel.output_key)
  if not artifacts:
    return []
  artifacts, artifact_type = _filter_by_artifact_query(
      store, artifacts, channel.artifact_query)
  return [
      artifact_utils.deserialize_artifact(artifact_type, a) for a in artifacts
  ]


def resolve_union_channels(
    mlmd_handle: metadata.Metadata,
    channels: Sequence[pipeline_pb2.InputSpec.Channel],
) -> List[types.Artifact]:
  """Evaluate InputSpec.channels."""
  seen = set()
  result = []
  for channel in channels:
    for artifact in resolve_single_channel(mlmd_handle, channel):
      if artifact.id not in seen:
        seen.add(artifact.id)
        result.append(artifact)
  return result
