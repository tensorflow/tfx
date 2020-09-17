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
"""Portable library for input artifacts resolution."""
import collections
from typing import Dict, List, Optional, Sequence

from absl import logging
from tfx import types
from tfx.orchestration import metadata
from tfx.orchestration.portable.mlmd import common_utils
from tfx.orchestration.portable.mlmd import event_lib
from tfx.orchestration.portable.mlmd import execution_lib
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import artifact_utils
import ml_metadata as mlmd

from ml_metadata.proto import metadata_store_pb2


def get_qualified_artifacts(
    metadata_handler: metadata.Metadata,
    contexts: Sequence[metadata_store_pb2.Context],
    artifact_type: metadata_store_pb2.ArtifactType,
    output_key: Optional[str] = None,
) -> List[types.Artifact]:
  """Gets qualified artifacts that have the right producer info.

  Args:
    metadata_handler: A metadata handler to access MLMD store.
    contexts: Context constraints to filter artifacts
    artifact_type: Type constraint to filter artifacts
    output_key: Output key constraint to filter artifacts

  Returns:
    A list of qualified TFX Artifacts.
  """
  # We expect to have at least one context for input resolution.
  assert contexts, 'Must have at least one context.'

  try:
    artifact_type_name = artifact_type.name
    artifact_type = metadata_handler.store.get_artifact_type(artifact_type_name)
  except mlmd.errors.NotFoundError:
    logging.warning('Artifact type %s is not found in MLMD.',
                    artifact_type.name)
    artifact_type = None

  if not artifact_type:
    return []

  # Gets the executions that are associated with all contexts.
  executions_dict = None
  for context in contexts:
    executions = metadata_handler.store.get_executions_by_context(context.id)
    if executions_dict is None:
      executions_dict = {e.id: e for e in executions}
    else:
      executions_dict = {e.id: e for e in executions if e.id in executions_dict}

  executions_within_context = executions_dict.values()  # pytype: disable=attribute-error

  # Filters out non-success executions.
  qualified_producer_executions = [
      e.id
      for e in executions_within_context
      if execution_lib.is_execution_successful(e)
  ]
  # Gets the output events that have the matched output key.
  qualified_output_events = [
      ev for ev in metadata_handler.store.get_events_by_execution_ids(
          qualified_producer_executions)
      if event_lib.validate_output_event(ev, output_key)
  ]

  # Gets the candidate artifacts from output events.
  candidate_artifacts = metadata_handler.store.get_artifacts_by_id(
      list(set(ev.artifact_id for ev in qualified_output_events)))
  # Filters the artifacts that have the right artifact type and state.
  qualified_artifacts = [
      a for a in candidate_artifacts if a.type_id == artifact_type.id and
      a.state == metadata_store_pb2.Artifact.LIVE
  ]
  return [
      artifact_utils.deserialize_artifact(artifact_type, a)
      for a in qualified_artifacts
  ]


def _resolve_single_channel(
    metadata_handler: metadata.Metadata,
    channel: pipeline_pb2.InputSpec.Channel) -> List[types.Artifact]:
  """Resolves input artifacts from a single channel."""

  artifact_type = channel.artifact_query.type
  output_key = channel.output_key
  contexts = [
      metadata_handler.store.get_context_by_type_and_name(
          context_query.type.name, common_utils.get_value(context_query.name))
      for context_query in channel.context_queries
  ]
  return get_qualified_artifacts(
      metadata_handler=metadata_handler,
      contexts=contexts,
      artifact_type=artifact_type,
      output_key=output_key)


def resolve_input_artifacts(
    metadata_handler: metadata.Metadata, node_inputs: pipeline_pb2.NodeInputs
) -> Optional[Dict[str, List[types.Artifact]]]:
  """Resolves input artifacts of a pipeline node.

  Args:
    metadata_handler: A metadata handler to access MLMD store.
    node_inputs: A pipeline_pb2.NodeInputs message that instructs artifact
      resolution for a pipeline node.

  Returns:
    If `min_count` for every input is met, returns a Dict[str, List[Artifact]].
    Otherwise, return None.
  """
  result = collections.defaultdict(set)
  all_input_satisfied = True
  for key, input_spec in node_inputs.inputs.items():
    for channel in input_spec.channels:
      artifacts = _resolve_single_channel(
          metadata_handler=metadata_handler, channel=channel)
      result[key].update(artifacts)

    # If `min_count` is not satisfied, return None for the whole result.
    if input_spec.min_count > len(result[key]):
      logging.warning(
          "Input %s doesn't have enough data to resolve, required number %d, "
          "got %d", key, input_spec.min_count, len(result[key]))
      all_input_satisfied = False

  return (
      {k: list(v) for k, v in result.items()} if all_input_satisfied else None)


def resolve_parameters(
    node_parameters: pipeline_pb2.NodeParameters) -> Dict[str, types.Property]:
  """Resolves parameters given parameter spec.

  Args:
    node_parameters: The spec to get parameters.

  Returns:
    A Dict of parameters.

  Raises:
    RuntimeError: When there is at least one parameter still in runtime
      parameter form.
  """
  result = {}
  for key, value in node_parameters.parameters.items():
    if not value.HasField('field_value'):
      raise RuntimeError('Parameter value not ready for %s' % key)
    result[key] = getattr(value.field_value,
                          value.field_value.WhichOneof('value'))

  return result
