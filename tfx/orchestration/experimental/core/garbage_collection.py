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
"""Utilities for garbage collecting artifacts."""

# TODO(yifanmai): Delete

from typing import List, Set

from tfx.orchestration import metadata
from tfx.orchestration.portable.mlmd import event_lib
from tfx.orchestration.portable.mlmd import execution_lib

from ml_metadata.proto import metadata_store_pb2


def _get_in_use_artifact_ids(mlmd_handle: metadata.Metadata,
                             artifact_ids: List[int]) -> Set[int]:
  """Returns IDs of artifacts that are currently used by executions."""
  events = mlmd_handle.store.get_events_by_artifact_ids(artifact_ids)
  input_events = [e for e in events if event_lib.is_valid_input_event(e)]
  execution_ids = [e.execution_id for e in input_events]
  executions = mlmd_handle.store.get_executions_by_id(execution_ids)
  execution_id_to_execution = {e.id: e for e in executions}
  in_use_artifact_ids = set()
  for event in input_events:
    if event.execution_id not in execution_id_to_execution:
      raise RuntimeError('Could not find execution with id: %d' %
                         event.execution_id)
    execution = execution_id_to_execution[event.execution_id]
    if execution_lib.is_execution_active(execution):
      in_use_artifact_ids.add(event.artifact_id)
  return in_use_artifact_ids


def _get_pipeline_artifacts(
    mlmd_handle: metadata.Metadata,
    pipeline_id: str) -> List[metadata_store_pb2.Artifact]:
  """Returns all artifacts of the given pipeline."""
  pipeline_context = mlmd_handle.store.get_context_by_type_and_name(
      'pipeline', pipeline_id)
  if not pipeline_context:
    raise RuntimeError('Could not find context for pipeline with id: %d' %
                       pipeline_id)
  result = mlmd_handle.store.get_artifacts_by_context(pipeline_context.id)
  return result


def _get_artifacts_to_garbage_collect(
    mlmd_handle: metadata.Metadata,
    pipeline_id: str) -> List[metadata_store_pb2.Artifact]:
  """Returns all artifacts that are eligible for garbage collection."""
  pipeline_artifacts = _get_pipeline_artifacts(mlmd_handle, pipeline_id)
  in_use_artifact_ids = _get_in_use_artifact_ids(
      mlmd_handle, [a.id for a in pipeline_artifacts])
  return [a for a in pipeline_artifacts if a.id not in in_use_artifact_ids]
