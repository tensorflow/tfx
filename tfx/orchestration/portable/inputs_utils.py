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
from typing import Dict, Iterable, List, Optional, Mapping, Sequence, Union

from absl import logging
from tfx import types
from tfx.orchestration import data_types_utils
from tfx.orchestration import metadata
from tfx.orchestration.portable.input_resolution import exceptions
from tfx.orchestration.portable.input_resolution import processor
from tfx.orchestration.portable.mlmd import event_lib
from tfx.orchestration.portable.mlmd import execution_lib
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import artifact_utils
from tfx.utils import deprecation_utils
from tfx.utils import typing_utils

import ml_metadata as mlmd
from ml_metadata.proto import metadata_store_pb2


def get_qualified_artifacts(
    metadata_handler: metadata.Metadata,
    contexts: Iterable[metadata_store_pb2.Context],
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
  # We expect to have at least one context for input resolution. Otherwise,
  # return empty list.
  if not contexts:
    return []

  try:
    artifact_type_name = artifact_type.name
    artifact_type = metadata_handler.store.get_artifact_type(artifact_type_name)
  except mlmd.errors.NotFoundError:
    logging.warning('Artifact type %s is not found in MLMD.',
                    artifact_type.name)
    artifact_type = None

  if not artifact_type:
    return []

  executions_within_context = (
      execution_lib.get_executions_associated_with_all_contexts(
          metadata_handler, contexts))

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
      if event_lib.is_valid_output_event(ev, output_key)
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
  output_key = channel.output_key or None

  contexts = []
  for context_query in channel.context_queries:
    context = metadata_handler.store.get_context_by_type_and_name(
        context_query.type.name, data_types_utils.get_value(context_query.name))
    if context:
      contexts.append(context)
    else:
      return []
  return get_qualified_artifacts(
      metadata_handler=metadata_handler,
      contexts=contexts,
      artifact_type=artifact_type,
      output_key=output_key)


def _resolve_initial_dict(
    metadata_handler: metadata.Metadata,
    node_inputs: pipeline_pb2.NodeInputs) -> typing_utils.ArtifactMultiMap:
  """Resolves initial input dict from input channel definition."""
  result = {}
  for key, input_spec in node_inputs.inputs.items():
    artifacts_by_id = {}  # Deduplicate by ID.
    for channel in input_spec.channels:
      artifacts = _resolve_single_channel(metadata_handler, channel)
      artifacts_by_id.update({a.id: a for a in artifacts})
    result[key] = list(artifacts_by_id.values())
  return result


def _is_sufficient(artifact_multimap: Mapping[str, Sequence[types.Artifact]],
                   node_inputs: pipeline_pb2.NodeInputs) -> bool:
  """Checks given artifact multimap has enough artifacts per channel."""
  return all(
      len(artifacts) >= node_inputs.inputs[key].min_count
      for key, artifacts in artifact_multimap.items()
      if key in node_inputs.inputs)


@deprecation_utils.deprecated(
    '2021-06-01', 'Use resolve_input_artifacts_v2() instead.')
def resolve_input_artifacts(
    metadata_handler: metadata.Metadata, node_inputs: pipeline_pb2.NodeInputs
) -> Optional[typing_utils.ArtifactMultiMap]:
  """Resolves input artifacts of a pipeline node.

  Args:
    metadata_handler: A metadata handler to access MLMD store.
    node_inputs: A pipeline_pb2.NodeInputs message that instructs artifact
      resolution for a pipeline node.

  Returns:
    If `min_count` for every input is met, returns a Dict[str, List[Artifact]].
    Otherwise, return None.
  """
  initial_dict = _resolve_initial_dict(metadata_handler, node_inputs)
  if not _is_sufficient(initial_dict, node_inputs):
    min_counts = {key: input_spec.min_count
                  for key, input_spec in node_inputs.inputs.items()}
    logging.warning('Resolved inputs should have %r artifacts, but got %r.',
                    min_counts, initial_dict)
    return None

  try:
    result = processor.run_resolver_steps(
        initial_dict,
        resolver_steps=node_inputs.resolver_config.resolver_steps,
        store=metadata_handler.store)
  except exceptions.InputResolutionError:
    # If ResolverStrategy has returned None in the middle, InputResolutionError
    # is raised. Legacy input resolution has returned None in this case.
    return None
  except exceptions.SkipSignal:
    # SkipSignal is not fully representable return value in legacy input
    # resolution, but None is the best effort.
    return None

  if not typing_utils.is_artifact_multimap(result):
    raise TypeError(f'Invalid input resolution result: {result}. Should be '
                    'Mapping[str, Sequence[Artifact]].')
  return result


class Trigger(tuple, Sequence[typing_utils.ArtifactMultiMap]):
  """Input resolution result of list of dict."""

  def __new__(cls, resolved_inputs: Sequence[typing_utils.ArtifactMultiMap]):
    assert resolved_inputs, 'resolved inputs should be non-empty.'
    return super().__new__(cls, resolved_inputs)


class Skip(tuple, Sequence[typing_utils.ArtifactMultiMap]):
  """Input resolution result of empty list."""

  def __new__(cls):
    return super().__new__(cls)


def resolve_input_artifacts_v2(
    *,
    pipeline_node: pipeline_pb2.PipelineNode,
    metadata_handler: metadata.Metadata,
) -> Union[Trigger, Skip]:
  """Resolve input artifacts according to a pipeline node IR definition.

  Input artifacts are resolved in the following steps:

  1. An initial input dict (Mapping[str, Sequence[Artifact]]) is fetched from
     the input channel definitions (in NodeInputs.inputs.channels).
  2. Optionally input resolution logic is performed if specified (in
     NodeInputs.resolver_config).
  3. Filters input map with enough number of artifacts as specified in
     NodeInputs.inputs.min_count.

  Args:
    pipeline_node: Current PipelineNode on which input resolution is running.
    metadata_handler: MetadataHandler instance for MLMD access.

  Raises:
    InputInputResolutionError: If input resolution went wrong.

  Returns:
    Trigger: a non-empty list of input dicts. All resolved input dicts should be
        executed.
    Skip: an empty list. Should effectively skip the current component
        execution.
  """
  node_inputs = pipeline_node.inputs
  initial_dict = _resolve_initial_dict(metadata_handler, node_inputs)
  try:
    resolved = processor.run_resolver_steps(
        initial_dict,
        resolver_steps=node_inputs.resolver_config.resolver_steps,
        store=metadata_handler.store)
  except exceptions.SkipSignal:
    return Skip()
  except exceptions.InputResolutionError:
    raise
  except Exception as e:
    raise exceptions.InputResolutionError(
        f'Error occurred during input resolution: {str(e)}.') from e

  if typing_utils.is_artifact_multimap(resolved):
    result = [resolved]
  else:
    raise exceptions.FailedPreconditionError(
        f'Invalid input resolution result: {resolved}.')
  result = [d for d in result if _is_sufficient(d, node_inputs)]
  if not result:
    raise exceptions.FailedPreconditionError('No valid inputs.')
  return Trigger(result)


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


def resolve_parameters_with_schema(
    node_parameters: pipeline_pb2.NodeParameters
) -> Dict[str, pipeline_pb2.Value]:
  """Resolves parameter schemas given parameter spec.

  Args:
    node_parameters: The spec to get parameters.

  Returns:
    A Dict of parameters with schema.

  Raises:
    RuntimeError: When there is no field_value available.
  """
  result = {}
  for key, value in node_parameters.parameters.items():
    if not value.HasField('field_value'):
      raise RuntimeError('Parameter value not ready for %s' % key)
    result[key] = value

  return result
