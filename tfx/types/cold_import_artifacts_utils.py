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
"""Utilities for importing artifacts from external MLMD db."""

from tfx import types
from tfx.types import artifact_utils

from ml_metadata.proto import metadata_store_pb2


def cold_import_artifacts(artifact: metadata_store_pb2.Artifact,
                          artifact_type: metadata_store_pb2.ArtifactType,
                          db_owner: str = '',
                          db_name: str = '') -> types.Artifact:
  """cold import artifacts.

  Args:
    artifact: An artifact.
    artifact_type: The type of the artifact.
    db_owner: The owner of the MLMD db the artifact belongs to.
    db_name: The name of the MLMD db the artifact belongs to.

  Returns:
    A tuple, in which the first element is types.Artifact and the second element
    is a unique identifier of the artifact.
  """
  if db_owner or db_name:
    pass

  return artifact_utils.deserialize_artifact(artifact_type, artifact)


def artifact_identifier(artifact: types.Artifact) -> str:
  return str(artifact.id)
