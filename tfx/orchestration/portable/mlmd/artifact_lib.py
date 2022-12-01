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
"""Portable APIs for managing artifacts in MLMD."""
import itertools
from typing import Optional, Sequence

from tfx import types
from tfx.orchestration import metadata
from tfx.types import artifact_utils
from tfx.utils import typing_utils


def get_artifacts_by_ids(
    metadata_handler: metadata.Metadata,
    artifact_ids: Sequence[int]) -> Sequence[types.Artifact]:
  """Gets TFX artifacts from MLMD by ID.

  Args:
    metadata_handler: A handler to access MLMD.
    artifact_ids: The IDs of existing artifacts to query.

  Returns:
    A list of the deserialized TFX artifacts.

  Raises:
    ValueError if one or more of the artifact IDs does not exist in MLMD.
  """
  mlmd_artifacts = metadata_handler.store.get_artifacts_by_id(artifact_ids)
  if len(artifact_ids) != len(mlmd_artifacts):
    raise ValueError(
        f'Could not find all MLMD artifacts for ids: {artifact_ids}')

  # Fetch artifact types and create a map keyed by artifact type id.
  artifact_type_ids = set(a.type_id for a in mlmd_artifacts)
  artifact_types = metadata_handler.store.get_artifact_types_by_id(
      artifact_type_ids)
  artifact_types_by_id = {a.id: a for a in artifact_types}

  # Set `type` field in the artifact proto which is not filled by MLMD.
  for mlmd_artifact in mlmd_artifacts:
    mlmd_artifact.type = artifact_types_by_id[mlmd_artifact.type_id].name

  # Return a list with MLMD artifacts deserialized to TFX Artifact instances.
  return [
      artifact_utils.deserialize_artifact(
          artifact_types_by_id[mlmd_artifact.type_id], mlmd_artifact)
      for mlmd_artifact in mlmd_artifacts
  ]


def update_artifacts(metadata_handler: metadata.Metadata,
                     tfx_artifact_map: typing_utils.ArtifactMultiMap,
                     new_artifact_state: Optional[str] = None) -> None:
  """Updates existing TFX artifacts in MLMD."""
  mlmd_artifacts_to_update = []
  for tfx_artifact in itertools.chain.from_iterable(tfx_artifact_map.values()):
    if not tfx_artifact.mlmd_artifact.HasField('id'):
      raise ValueError('Artifact must have an MLMD ID in order to be updated.')
    if new_artifact_state:
      tfx_artifact.state = new_artifact_state
    mlmd_artifacts_to_update.append(tfx_artifact.mlmd_artifact)
  if mlmd_artifacts_to_update:
    metadata_handler.store.put_artifacts(mlmd_artifacts_to_update)
