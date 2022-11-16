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

from ml_metadata.proto import metadata_store_pb2


def prepare_external_artifact(
    external_artifact_type: metadata_store_pb2.ArtifactType,
    external_artifact: metadata_store_pb2.Artifact, project_owner: str,
    project_name: str) -> types.Artifact:
  """Deserialize the artifact into types.Artifact.

  Args:
    external_artifact_type: The type of the artifact.
    external_artifact: An artifact from an external project.
    project_owner: The owner of the project the artifact belongs to.
    project_name: The name of the project the artifact belongs to.

  Returns:
    NotImplementedError, since OSS version does not have external artifacts.
  """
  raise NotImplementedError('OSS version does not have external artifacts.')


def artifact_identifier(artifact: types.Artifact) -> str:
  return str(artifact.id)
