# Lint as: python3
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
"""Portable libraries for execution related APIs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, MutableMapping, Mapping, Optional, Sequence, Text, Tuple

from absl import logging

from ml_metadata.proto import metadata_store_pb2
from tfx import types
from tfx.orchestration import metadata
from tfx.orchestration.portable.mlmd import common_utils
from tfx.orchestration.portable.mlmd import event_lib


def is_execution_successful(execution: metadata_store_pb2.Execution) -> bool:
  """Whether or not an execution is successful.

  Args:
    execution: An execution message.

  Returns:
    A bool value indicating whether or not the execution is successful.
  """
  return (execution.last_known_state == metadata_store_pb2.Execution.COMPLETE or
          execution.last_known_state == metadata_store_pb2.Execution.CACHED)


def prepare_execution(
    metadata_handler: metadata.Metadata,
    execution_type: metadata_store_pb2.ExecutionType,
    state: metadata_store_pb2.Execution.State,
    exec_properties: Optional[Mapping[Text, types.Property]] = None,
) -> metadata_store_pb2.Execution:
  """Creates an execution proto based on the information provided.

  Args:
    metadata_handler: A handler to access MLMD store.
    execution_type: A metadata_pb2.ExecutionType message describing the type of
      the execution.
    state: The state of the execution.
    exec_properties: Execution properties that need to be attached.

  Returns:
    A metadata_store_pb2.Execution message.
  """
  execution = metadata_store_pb2.Execution()
  execution.last_known_state = state
  execution.type_id = common_utils.register_type_if_not_exist(
      metadata_handler, execution_type).id

  exec_properties = exec_properties or {}
  # For every execution property, put it in execution.properties if its key is
  # in execution type schema. Otherwise, put it in execution.custom_properties.
  for k, v in exec_properties.items():
    if (execution_type.properties.get(k) ==
        common_utils.get_metadata_value_type(v)):
      common_utils.set_metadata_value(execution.properties[k], v)
    else:
      common_utils.set_metadata_value(execution.custom_properties[k], v)
  logging.debug('Prepared EXECUTION:\n %s', execution)
  return execution


def _create_artifact_and_event_pairs(
    metadata_handler: metadata.Metadata,
    artifact_dict: MutableMapping[Text, Sequence[types.Artifact]],
    event_type: metadata_store_pb2.Event.Type,
) -> List[Tuple[metadata_store_pb2.Artifact, metadata_store_pb2.Event]]:
  """Creates a list of [Artifact, Event] tuples.

  The result of this function will be used in a MLMD put_execution() call.

  Args:
    metadata_handler: A handler to access MLMD store.
    artifact_dict: The source of artifacts to work on. For each artifact in the
      dict, creates a tuple for that. Note that all artifacts of the same key in
      the artifact_dict are expected to share the same artifact type.
    event_type: The event type of the event to be attached to the artifact

  Returns:
    A list of [Artifact, Event] tuples
  """
  result = []
  for key, artifact_list in artifact_dict.items():
    artifact_type = None
    for index, artifact in enumerate(artifact_list):
      # TODO(b/153904840): If artifact id is present, skip putting the artifact
      # into the pair when MLMD API is ready.
      event = event_lib.generate_event(
          event_type=event_type, key=key, index=index)
      # Reuses already registered type in the same list whenever possible as
      # the artifacts in the same list share the same artifact type.
      if artifact_type:
        assert artifact_type.name == artifact.artifact_type.name, (
            'Artifacts under the same key should share the same artifact type.')
      artifact_type = common_utils.register_type_if_not_exist(
          metadata_handler, artifact.artifact_type)
      artifact.set_mlmd_artifact_type(artifact_type)
      result.append((artifact.mlmd_artifact, event))
  return result


def put_execution(
    metadata_handler: metadata.Metadata,
    execution: metadata_store_pb2.Execution,
    contexts: Sequence[metadata_store_pb2.Context],
    input_artifacts: Optional[MutableMapping[Text,
                                             Sequence[types.Artifact]]] = None,
    output_artifacts: Optional[MutableMapping[Text,
                                              Sequence[types.Artifact]]] = None
) -> metadata_store_pb2.Execution:
  """Writes an execution-centric subgraph to MLMD.

  This function mainly leverages metadata.put_execution() method to write the
  execution centric subgraph to MLMD.

  Args:
    metadata_handler: A handler to access MLMD.
    execution: The execution to be written to MLMD.
    contexts: MLMD contexts to associated with the execution.
    input_artifacts: Input artifacts of the execution. Each artifact will be
      linked with the execution through an event with type INPUT. Each artifact
      will also be linked with every context in the `contexts` argument.
    output_artifacts: Output artifacts of the execution. Each artifact will be
      linked with the execution through an event with type OUTPUT. Each artifact
      will also be linked with every context in the `contexts` argument.

  Returns:
    An MLMD execution that is written to MLMD, with id pupulated.
  """
  artifacts_and_events = []
  if input_artifacts:
    artifacts_and_events.extend(
        _create_artifact_and_event_pairs(
            metadata_handler=metadata_handler,
            artifact_dict=input_artifacts,
            event_type=metadata_store_pb2.Event.INPUT))
  if output_artifacts:
    artifacts_and_events.extend(
        _create_artifact_and_event_pairs(
            metadata_handler=metadata_handler,
            artifact_dict=output_artifacts,
            event_type=metadata_store_pb2.Event.OUTPUT))
  execution_id, artifact_ids, contexts_ids = (
      metadata_handler.store.put_execution(execution, artifacts_and_events,
                                           contexts))
  execution.id = execution_id
  for artifact_and_event, a_id in zip(artifacts_and_events, artifact_ids):
    artifact, _ = artifact_and_event
    artifact.id = a_id
  for context, c_id in zip(contexts, contexts_ids):
    context.id = c_id

  return execution
