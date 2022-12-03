# Copyright 2020 Google LLC. All Rights Reserved.
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
"""Portable library for registering and publishing executions."""
import copy
import itertools
import os
from typing import Mapping, Optional, Sequence
import uuid
from absl import logging

from tfx import types
from tfx.orchestration import metadata
from tfx.orchestration.portable import outputs_utils
from tfx.orchestration.portable import merge_utils
from tfx.orchestration.portable.mlmd import execution_lib
from tfx.proto.orchestration import execution_result_pb2
from tfx.utils import typing_utils

from ml_metadata.proto import metadata_store_pb2

_RESOLVED_AT_RUNTIME = outputs_utils.RESOLVED_AT_RUNTIME


def _check_validity(new_artifact: metadata_store_pb2.Artifact,
                    original_artifact: types.Artifact,
                    has_multiple_artifacts: bool) -> None:
  """Check the validity of new artifact against the original artifact."""
  if (original_artifact.mlmd_artifact.HasField('type_id') and
      new_artifact.HasField('type_id') and
      new_artifact.type_id != original_artifact.type_id):
    raise RuntimeError('Executor output should not change artifact type '
                       f'(original type_id={original_artifact.type_id}, '
                       f'new type_id={new_artifact.type_id}).')

  # If the artifact is external and the uri is resolved during runtime, it
  # doesn't check the validity of uri.
  if original_artifact.is_external and original_artifact.uri == outputs_utils.RESOLVED_AT_RUNTIME:
    return

  if has_multiple_artifacts:
    # If there are multiple artifacts in the executor output, their URIs should
    # be a direct sub-dir of the system generated URI.
    if os.path.dirname(new_artifact.uri) != original_artifact.uri:
      raise RuntimeError(
          'When there are multiple artifacts to publish, their URIs '
          'should be direct sub-directories of the URI of the system generated '
          'artifact.')
  else:
    # If there is only one output artifact, its URI should not be changed
    if new_artifact.uri != original_artifact.uri:
      # TODO(b/175426744): Data Binder will modify the uri.
      logging.warning(
          'When there is one artifact to publish, the URI of it should be '
          'identical to the URI of system generated artifact.')


def publish_cached_execution(
    metadata_handler: metadata.Metadata,
    contexts: Sequence[metadata_store_pb2.Context],
    execution_id: int,
    output_artifacts: Optional[typing_utils.ArtifactMultiMap] = None,
) -> None:
  """Marks an existing execution as using cached outputs from a previous execution.

  Args:
    metadata_handler: A handler to access MLMD.
    contexts: MLMD contexts to associated with the execution.
    execution_id: The id of the execution.
    output_artifacts: Output artifacts of the execution. Each artifact will be
      linked with the execution through an event with type OUTPUT.
  """
  [execution] = metadata_handler.store.get_executions_by_id([execution_id])
  execution.last_known_state = metadata_store_pb2.Execution.CACHED

  execution_lib.put_executions(
      metadata_handler, [execution],
      contexts,
      input_artifacts_maps=None,
      output_artifacts_maps=[output_artifacts] if output_artifacts else None)


def _set_execution_result_if_not_empty(
    executor_output: Optional[execution_result_pb2.ExecutorOutput],
    execution: metadata_store_pb2.Execution) -> None:
  """Sets execution result as a custom property of the execution."""
  if executor_output and (executor_output.execution_result.result_message or
                          executor_output.execution_result.metadata_details or
                          executor_output.execution_result.code):
    execution_lib.set_execution_result(executor_output.execution_result,
                                       execution)


def publish_succeeded_execution(
    metadata_handler: metadata.Metadata,
    execution_id: int,
    contexts: Sequence[metadata_store_pb2.Context],
    output_artifacts: Optional[typing_utils.ArtifactMultiMap] = None,
    executor_output: Optional[execution_result_pb2.ExecutorOutput] = None
) -> Optional[typing_utils.ArtifactMultiMap]:
  """Marks an existing execution as success.

  Also publishes the output artifacts produced by the execution. This method
  will also merge the executor produced info into system generated output
  artifacts. The `last_know_state` of the execution will be changed to
  `COMPLETE` and the output artifacts will be marked as `LIVE`.

  Args:
    metadata_handler: A handler to access MLMD.
    execution_id: The id of the execution to mark successful.
    contexts: MLMD contexts to associated with the execution.
    output_artifacts: Output artifacts skeleton of the execution, generated by
      the system. Each artifact will be linked with the execution through an
      event with type OUTPUT.
    executor_output: Executor outputs. `executor_output.output_artifacts` will
      be used to update system-generated output artifacts passed in through
      `output_artifacts` arg. There are three contraints to the update: 1. The
        keys in `executor_output.output_artifacts` are expected to be a subset
        of the system-generated output artifacts dict. 2. An update to a certain
        key should contains all the artifacts under that key. 3. An update to an
        artifact should not change the type of the artifact.

  Returns:
    The maybe updated output_artifacts, note that only outputs whose key are in
    executor_output will be updated and others will be untouched. That said,
    it can be partially updated.
  Raises:
    RuntimeError: if the executor output to a output channel is partial.
  """
  if output_artifacts is not None:
    output_artifacts = {key: [copy.deepcopy(a) for a in artifacts]
                        for key, artifacts in output_artifacts.items()}
  else:
    output_artifacts = {}
  if executor_output:
    if not set(executor_output.output_artifacts.keys()).issubset(
        output_artifacts.keys()):
      raise RuntimeError(
          'Executor output %s contains more keys than output skeleton %s.' %
          (executor_output, output_artifacts))
    for key, artifact_list in output_artifacts.items():
      if key not in executor_output.output_artifacts:
        # The executor output did not include the output key, which implies the
        # component doesn't need to update these output artifacts. In this case,
        # we remove any output artifacts with a URI value of RESOLVED_AT_RUNTIME
        # and publish the remaining output artifacts as-is.
        filtered_artifacts = [
            artifact for artifact in artifact_list
            if artifact.uri != _RESOLVED_AT_RUNTIME
        ]
        artifact_list.clear()
        artifact_list.extend(filtered_artifacts)
        continue

      updated_artifact_list = executor_output.output_artifacts[key].artifacts

      # We assume the original output dict must include at least one output
      # artifact and all artifacts in the list share the same type/properties.
      default_original_artifact = copy.deepcopy(artifact_list[0])
      default_original_artifact.mlmd_artifact.ClearField('id')

      # Update the artifact list with what's in the executor output. Note the
      # original artifacts may have existing artifact IDs if they were
      # registered in MLMD before the execution.
      original_artifacts_by_uri = {x.uri: x for x in artifact_list}
      artifact_list.clear()
      # TODO(b/175426744): revisit this:
      # 1) Whether multiple output is needed or not after TFX componets
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

        _check_validity(updated_artifact_proto, original_artifact,
                        len(updated_artifact_list) > 1)
        merged_artifact = merge_utils.merge_output_artifact(
            original_artifact, updated_artifact_proto)
        artifact_list.append(merged_artifact)

  # Marks output artifacts as PUBLISHED (i.e. LIVE in MLMD).
  for artifact in itertools.chain.from_iterable(output_artifacts.values()):
    artifact.state = types.artifact.ArtifactState.PUBLISHED

  [execution] = metadata_handler.store.get_executions_by_id([execution_id])
  execution.last_known_state = metadata_store_pb2.Execution.COMPLETE
  if executor_output:
    for key, value in executor_output.execution_properties.items():
      execution.custom_properties[key].CopyFrom(value)
  _set_execution_result_if_not_empty(executor_output, execution)

  execution_lib.put_executions(
      metadata_handler, [execution],
      contexts,
      output_artifacts_maps=[output_artifacts])

  return output_artifacts


def publish_failed_execution(
    metadata_handler: metadata.Metadata,
    contexts: Sequence[metadata_store_pb2.Context],
    execution_id: int,
    executor_output: Optional[execution_result_pb2.ExecutorOutput] = None
) -> None:
  """Marks an existing execution as failed.

  Args:
    metadata_handler: A handler to access MLMD.
    contexts: MLMD contexts to associated with the execution.
    execution_id: The id of the execution.
    executor_output: The output of executor.
  """
  [execution] = metadata_handler.store.get_executions_by_id([execution_id])
  execution.last_known_state = metadata_store_pb2.Execution.FAILED
  _set_execution_result_if_not_empty(executor_output, execution)

  execution_lib.put_executions(metadata_handler, [execution], contexts)


def publish_internal_execution(
    metadata_handler: metadata.Metadata,
    contexts: Sequence[metadata_store_pb2.Context],
    execution_id: int,
    output_artifacts: Optional[typing_utils.ArtifactMultiMap] = None
) -> None:
  """Marks an exeisting execution as as success and links its output to an INTERNAL_OUTPUT event.

  Args:
    metadata_handler: A handler to access MLMD.
    contexts: MLMD contexts to associated with the execution.
    execution_id: The id of the execution.
    output_artifacts: Output artifacts of the execution. Each artifact will be
      linked with the execution through an event with type INTERNAL_OUTPUT.
  """
  [execution] = metadata_handler.store.get_executions_by_id([execution_id])
  execution.last_known_state = metadata_store_pb2.Execution.COMPLETE

  execution_lib.put_executions(
      metadata_handler, [execution],
      contexts,
      output_artifacts_maps=[output_artifacts] if output_artifacts else None,
      output_event_type=metadata_store_pb2.Event.INTERNAL_OUTPUT)


def register_execution(
    metadata_handler: metadata.Metadata,
    execution_type: metadata_store_pb2.ExecutionType,
    contexts: Sequence[metadata_store_pb2.Context],
    input_artifacts: Optional[typing_utils.ArtifactMultiMap] = None,
    exec_properties: Optional[Mapping[str, types.ExecPropertyTypes]] = None,
    last_known_state: metadata_store_pb2.Execution.State = metadata_store_pb2
    .Execution.RUNNING
) -> metadata_store_pb2.Execution:
  """Registers a new execution in MLMD.

  Along with the execution:
  -  the input artifacts will be linked to the execution.
  -  the contexts will be linked to both the execution and its input artifacts.

  Args:
    metadata_handler: A handler to access MLMD.
    execution_type: The type of the execution.
    contexts: MLMD contexts to associated with the execution.
    input_artifacts: Input artifacts of the execution. Each artifact will be
      linked with the execution through an event.
    exec_properties: Execution properties. Will be attached to the execution.
    last_known_state: The last known state of the execution.

  Returns:
    An MLMD execution that is registered in MLMD, with id populated.
  """
  # Setting exec_name is required to make sure that only one execution is
  # registered in MLMD. If there is a RPC retry, AlreadyExistError will raise.
  # After this fix (b/221103319), AlreadyExistError may not raise. Instead,
  # execution may be updated again upon RPC retries.
  exec_name = str(uuid.uuid4())
  execution = execution_lib.prepare_execution(
      metadata_handler,
      execution_type,
      last_known_state,
      exec_properties,
      execution_name=exec_name)

  return execution_lib.put_executions(
      metadata_handler, [execution],
      contexts,
      input_artifacts_maps=[input_artifacts] if input_artifacts else None)[0]
