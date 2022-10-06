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

from absl import logging

from tfx import types
from tfx.orchestration import metadata
from ml_metadata import errors
from ml_metadata.google.services.client.cross_db import reference_utils
from ml_metadata.google.services.mlmd_service.proto import mlmd_service_pb2


def copy_artifact(
    source_artifact: types.Artifact,
    source_owner: str,
    source_name: str,
    target_mlmd_handle: metadata.Metadata,
):
  """Registers a metadata type if not exists."""
  source_artifact = source_artifact.mlmd_artifact
  source_artifact_type = source_artifact.artifact_type
  logging.error('Guowei copy_artifact source_artifact %s', source_artifact)
  logging.error('Guowei copy_artifact source_artifact_type %s',
                source_artifact_type)

  pipeline_asset = mlmd_service_pb2.PipelineAsset(
      owner=source_owner, name=source_name)
  logging.error('Guowei copy_artifact pipeline_asset %s', pipeline_asset)

  target_artifact = reference_utils.add_reference_to_artifact(
      source_artifact, pipeline_asset, source_artifact.id)
  logging.error('Guowei copy_artifact target_artifact %s', target_artifact)

  try:
    target_artifact_type = target_mlmd_handle.store.get_artifact_type(
        type_name=source_artifact_type.name)
    logging.error('Guowei copy_artifact target_artifact_type %s',
                  target_artifact_type)
    target_artifact.type_id = target_artifact_type.id
  except errors.NotFoundError:
    source_artifact_type.ClearField('id')
    target_artifact_type_id = target_mlmd_handle.store.put_artifact_type(
        source_artifact_type)
    target_artifact.type_id = target_artifact_type_id

  logging.error('Guowei copy_artifact target_artifact %s', target_artifact)
