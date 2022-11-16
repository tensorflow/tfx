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
"""Portable library for merging data during pipeline orchestration."""

from tfx import types

from ml_metadata.proto import metadata_store_pb2


def merge_output_artifact(
    original_artifact: types.Artifact,
    updated_artifact_proto: metadata_store_pb2.Artifact,
) -> types.Artifact:
  """Merges an original output artifact with its post-execution updated version.

  Args:
    original_artifact: The original Artifact object that was created by the
      Orchestrator and passed to the component in the ExecutionInvocation.
    updated_artifact_proto: The updated Artifact proto returned by the component
      in the ExecutorOutput.

  Returns:
    A merged Artifact object combining the original and updated artifacts.
  """
  updated_artifact = types.Artifact(original_artifact.artifact_type)
  updated_artifact.set_mlmd_artifact(updated_artifact_proto)

  # Ensure the updated artifact has a consistent type ID with the original type.
  if original_artifact.artifact_type.HasField('id'):
    updated_artifact.type_id = original_artifact.artifact_type.id

  # Enforce that the component does not update the externality of the artifact.
  updated_artifact.is_external = original_artifact.is_external

  return updated_artifact
