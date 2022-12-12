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
import copy
import os
from typing import Mapping, Optional, Sequence

from absl import logging
from tfx import types
from tfx.orchestration.portable import outputs_utils
from tfx.utils import typing_utils

from ml_metadata.proto import metadata_store_pb2

_RESOLVED_AT_RUNTIME = outputs_utils.RESOLVED_AT_RUNTIME


def _merge_output_artifact(
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

  # Ensure the updated artifact has a consistent ID with the original.
  if original_artifact.mlmd_artifact.HasField('id'):
    updated_artifact_proto.id = original_artifact.mlmd_artifact.id
  else:
    updated_artifact_proto.ClearField('id')

  # Ensure the updated artifact has a consistent type ID with the original type.
  if original_artifact.artifact_type.HasField('id'):
    updated_artifact.type_id = original_artifact.artifact_type.id

  # Enforce that the component does not update the externality of the artifact.
  updated_artifact.is_external = original_artifact.is_external

  return updated_artifact


def _validate_updated_artifact(updated_artifact: metadata_store_pb2.Artifact,
                               original_artifact: types.Artifact,
                               has_multiple_artifacts: bool) -> None:
  """Check the validity of an updated artifact against the original artifact."""
  if (original_artifact.mlmd_artifact.HasField('type_id') and
      updated_artifact.HasField('type_id') and
      updated_artifact.type_id != original_artifact.type_id):
    raise RuntimeError('Executor output should not change artifact type '
                       f'(original type_id={original_artifact.type_id}, '
                       f'new type_id={updated_artifact.type_id}).')

  # If the artifact is external and the uri is resolved during runtime, it
  # doesn't check the validity of uri.
  if (original_artifact.is_external and
      original_artifact.uri == _RESOLVED_AT_RUNTIME):
    return

  if has_multiple_artifacts:
    # If there are multiple artifacts in the executor output, their URIs should
    # be a direct sub-dir of the system generated URI.
    if os.path.dirname(updated_artifact.uri) != original_artifact.uri:
      raise RuntimeError(
          'When there are multiple artifacts to publish, their URIs '
          'should be direct sub-directories of the URI of the system generated '
          'artifact.')
  else:
    # If there is only one output artifact, its URI should not be changed
    if updated_artifact.uri != original_artifact.uri:
      # TODO(b/175426744): Data Binder will modify the uri.
      logging.warning(
          'When there is one artifact to publish, the URI of it should be '
          'identical to the URI of system generated artifact.')


def merge_updated_output_artifacts(
    original_output_artifacts: Optional[typing_utils.ArtifactMultiMap] = None,
    updated_output_artifacts: Optional[Mapping[
        str, Sequence[metadata_store_pb2.Artifact]]] = None
) -> typing_utils.ArtifactMultiMap:
  """Merges the updated output artifacts returned by a pipeline node.

  Before an execution begins, the orchestrator will generate expected output
  artifacts based on the pipeline IR. These output artifacts are registered in
  MLMD as pending output artifacts, and are what is passed to the pipeline node
  in the ExecutionInvocation.

  Following the execution, the node returns an ExecutorOutput to the
  orchestrator, which can contain output artifacts for the execution that the
  node wants the orchestrator to update. A common use case for this is that the
  node populates properties on the output artifacts during execution.

  This function handles the merging logic to reconcile the pre-registered copy
  of the output artifacts with the final set provided back by the pipeline node.

  Args:
    original_output_artifacts: The output artifacts pre-registered before the
      start of the execution, which were generated based on the pipeline IR.
    updated_output_artifacts: The output artifacts returned by the pipeline
      component (in the MLMD artifact format) as the result of an execution.

  Returns:
    A merged output artfiact map, representing the output artifacts that should
    be published for the execution.

  Raises:
    RuntimeError: If the output artifacts from the executor output add keys that
    don't exist in the pre-registered output artifacts.
  """
  if original_output_artifacts is not None:
    original_output_artifacts = {
        key: [copy.deepcopy(a) for a in artifact_list
             ] for key, artifact_list in original_output_artifacts.items()
    }
  else:
    original_output_artifacts = {}

  if updated_output_artifacts:
    if not set(updated_output_artifacts.keys()).issubset(
        original_output_artifacts.keys()):
      raise RuntimeError(
          'Executor output %s contains more keys than output skeleton %s.' %
          (updated_output_artifacts, original_output_artifacts))
    for key, original_artifact_list in original_output_artifacts.items():
      if key not in updated_output_artifacts:
        # The executor output did not include the output key, which implies the
        # component doesn't need to update these output artifacts. In this case,
        # we remove any output artifacts with a URI value of RESOLVED_AT_RUNTIME
        # and publish the remaining output artifacts as-is.
        filtered_artifacts = [
            artifact for artifact in original_artifact_list
            if artifact.uri != _RESOLVED_AT_RUNTIME
        ]
        original_artifact_list.clear()
        original_artifact_list.extend(filtered_artifacts)
        continue

      updated_artifact_list = updated_output_artifacts[key]

      # We assume the original output dict must include at least one output
      # artifact and all artifacts in the list share the same type/properties.
      default_original_artifact = copy.deepcopy(original_artifact_list[0])
      default_original_artifact.mlmd_artifact.ClearField('id')

      # Update the artifact list with what's in the executor output. Note the
      # original artifacts may have existing artifact IDs if they were
      # registered in MLMD before the execution.
      original_artifacts_by_uri = {x.uri: x for x in original_artifact_list}
      original_artifact_list.clear()
      # TODO(b/175426744): revisit this:
      # 1) Whether multiple output is needed or not after TFX components
      #    are upgraded.
      # 2) If multiple output are needed and is a common practice, should we
      #    use driver instead to create the list of output artifact instead
      #    of letting executor to create them.
      for updated_artifact_proto in updated_artifact_list:
        updated_artifact_uri = updated_artifact_proto.uri
        if updated_artifact_uri == _RESOLVED_AT_RUNTIME:
          # Don't publish the output artifact if the component didn't set the
          # actual resolved artifact URI in the executor output.
          continue

        # Determine which original artifact to merge with this updated artifact.
        if updated_artifact_uri in original_artifacts_by_uri:
          original_artifact = original_artifacts_by_uri[updated_artifact_uri]
          del original_artifacts_by_uri[updated_artifact_uri]
        else:
          # The updated artifact proto doesn't match one of the original
          # artifacts, so it will be newly created in MLMD.
          original_artifact = copy.deepcopy(default_original_artifact)

        _validate_updated_artifact(updated_artifact_proto, original_artifact,
                                   len(updated_artifact_list) > 1)
        merged_artifact = _merge_output_artifact(original_artifact,
                                                 updated_artifact_proto)
        original_artifact_list.append(merged_artifact)

  return original_output_artifacts
