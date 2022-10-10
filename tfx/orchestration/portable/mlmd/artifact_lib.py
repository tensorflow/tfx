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
"""Common MLMD utility libraries."""

from tfx.orchestration import metadata

from ml_metadata import errors
from ml_metadata.google.services.client.cross_db import reference_utils
from ml_metadata.google.services.mlmd_service.proto import mlmd_service_pb2
from ml_metadata.proto import metadata_store_pb2


def update_artifact_for_target_mlmd(
    source_artifact: metadata_store_pb2.Artifact,
    source_owner: str,
    source_name: str,
    target_mlmd_handle: metadata.Metadata,
) -> metadata_store_pb2.Artifact:
  """Copies an artifact to the target mlmd db."""
  pipeline_asset = mlmd_service_pb2.PipelineAsset(
      owner=source_owner, name=source_name)
  target_artifact = reference_utils.add_reference_to_artifact(
      source_artifact.mlmd_artifact, pipeline_asset, source_artifact.id)

  try:
    target_artifact_type_id = target_mlmd_handle.store.get_artifact_type(
        type_name=source_artifact.type_name).id
  except errors.NotFoundError:
    source_artifact_type = source_artifact.artifact_type
    source_artifact_type.ClearField('id')
    target_artifact_type_id = target_mlmd_handle.store.put_artifact_type(
        source_artifact_type)
  target_artifact.type_id = target_artifact_type_id

  return target_artifact
