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
import importlib
import sys
import traceback
from typing import Dict, Iterable, List, Optional, Union, Any, Mapping, Sequence

from absl import logging
from tfx import types
from tfx.dsl.components.common import resolver
from tfx.dsl.input_resolution import exceptions
from tfx.dsl.input_resolution import resolver_op
from tfx.orchestration import data_types_utils
from tfx.orchestration import metadata
from tfx.orchestration.portable.mlmd import event_lib
from tfx.orchestration.portable.mlmd import execution_lib
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import artifact_utils
from tfx.utils import deprecation_utils
from tfx.utils import json_utils

import ml_metadata as mlmd
from ml_metadata.proto import metadata_store_pb2

_ArtifactMultimap = resolver_op.ArtifactMultimap


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
  return get_qualified_artifacts(
      metadata_handler=metadata_handler,
      contexts=contexts,
      artifact_type=artifact_type,
      output_key=output_key)


def _resolve_initial_dict(
    metadata_handler: metadata.Metadata,
    node_inputs: pipeline_pb2.NodeInputs) -> _ArtifactMultimap:
  """Resolve initial input dict from input channel definition."""
  result = {}
  for key, input_spec in node_inputs.inputs.items():
    artifacts_by_id = {}  # Deduplicate by ID.
    for channel in input_spec.channels:
      artifacts = _resolve_single_channel(metadata_handler, channel)
      artifacts_by_id.update({a.id: a for a in artifacts})
    result[key] = list(artifacts_by_id.values())
  return result


def _is_artifact_multimap(value: Any) -> bool:
  """Check value is Mapping[str, Sequence[Artifact]] type."""
  if not isinstance(value, collections.abc.Mapping):
    return False
  for key, list_artifacts in value.items():
    if (not isinstance(key, str) or
        not isinstance(list_artifacts, collections.abc.Sequence) or
        not all(isinstance(v, types.Artifact) for v in list_artifacts)):
      return False
  return True


def _is_list_of_artifact_multimap(value: Any) -> bool:
  """Check value is Sequence[Mapping[str, Sequence[Artifact]]] type."""
  return (isinstance(value, collections.abc.Sequence) and
          all(_is_artifact_multimap(v) for v in value))


def _is_sufficient(artifact_multimap: Mapping[str, Sequence[types.Artifact]],
                   node_inputs: pipeline_pb2.NodeInputs) -> bool:
  """Check given artifact multimap has enough artifacts per channel."""
  return all(
      len(artifacts) >= node_inputs.inputs[key].min_count
      for key, artifacts in artifact_multimap.items()
      if key in node_inputs.inputs)


@deprecation_utils.deprecated(
    '2021-06-01', 'Use resolve_input_artifacts_v2() instead.')
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
  initial_dict = _resolve_initial_dict(metadata_handler, node_inputs)
  if not _is_sufficient(initial_dict, node_inputs):
    min_counts = {key: input_spec.min_count
                  for key, input_spec in node_inputs.inputs.items()}
    logging.warning('Resolved inputs should have %r artifacts, but got %r.',
                    min_counts, initial_dict)
    return None

  try:
    result = _run_resolver_steps(
        initial_dict,
        resolver_steps=node_inputs.resolver_config.resolver_steps,
        store=metadata_handler.store)
  except (exceptions.SkipSignal, exceptions.IgnoreSignal):
    return None
  if not _is_artifact_multimap(result):
    raise TypeError(f'Invalid input resolution result: {result}. Should be '
                    'Mapping[str, Sequence[Artifact]].')
  return result


class ResolutionSucceeded(tuple, Sequence[_ArtifactMultimap]):
  """Successful input resolution result as a list of input dicts.

  Although input resolution was successful (i.e. no exception raised), if the
  result is empty, it will be regarded as an invalid result in synchronous
  mode.
  """

  def __new__(cls, values: Sequence[_ArtifactMultimap]):
    for value in values:
      if not _is_artifact_multimap(value):
        raise TypeError(f'Invalid value type: {type(value)}. Must be '
                        'Mapping[str, Sequence[Artifact]].')
    return super().__new__(cls, values)


class ResolutionFailed:
  """Input resolution has failed with an error.

  ResolutionFailed should always be noted to the user so that user can take
  appropriate action to make the input resolution result correct.
  """

  def __init__(self, reason: Optional[str] = None):
    self._reason = reason
    self._exc_info = sys.exc_info()
    if self._reason is None and self._exc_info[0] is None:
      raise ValueError('ResolutionFailed.reason should be given if no '
                       'exception is raised')

  def __str__(self) -> str:
    if self._reason is not None:
      return self._reason
    else:
      return ''.join(traceback.format_exception(*self._exc_info))


class Ignore:
  """Ignore the node of this input resolution result as if it has not existed.

  This is a special input resolution result to effectively erase the nodes
  from the pipeline dynamically during runtime. For example, components of
  untaken conditional branch is *Ignored* so that all nodes in the pipeline DAG
  are run, but ignored components are not executed.
  """


InputResolutionResult = Union[ResolutionSucceeded, ResolutionFailed, Ignore]


def resolve_input_artifacts_v2(
    *,
    pipeline_node: pipeline_pb2.PipelineNode,
    metadata_handler: metadata.Metadata,
) -> InputResolutionResult:
  """Resolve input artifacts according to a pipeline node IR definition.

  Input artifacts are resolved in the following steps:

  1. An initial input dict (Mapping[str, Sequence[Artifact]]) is fetched from
     the input channel definitions (configured in NodeInputs.inputs.channels).
  2. Input resolution logic (configured in NodeInputs.resolver_config) is
     applied to produce the list of input dicts. If no input resolution logic
     is configured, it simply produces a list of single item: [input_dict].
  3. Finally input dicts without enough number of artifacts (configured in
     NodeInputs.inputs.min_count) is filtered out.

  There are three types of input resolution result:

  * ResolutionSucceeded: In normal cases input resolution result is a list of
    input dicts, or Sequence[Mapping[str, Sequence[Artifact]]]. All resolved
    inputs should be executed by Executor.Do(). Result can also be empty, which
    means there are no inputs available for the component.
  * ResolutionFailed: Any uncatched exceptions or invalid input resolution
    output value would result in ResolutionFailed.
  * Ignore: Special input resolution result to ignore the current node as if it
    has not been existsed. This is used to dynamically erase nodes during
    runtime, for example conditional branch that is not taken. This is only
    valid in synchronous mode.

  Args:
    pipeline_node: Current PipelineNode on which input resolution is running.
    metadata_handler: MetadataHandler instance for MLMD access.

  Returns:
    One of ResolutionSucceeded, ResolutionFailed, or Ignore.
  """
  try:
    node_inputs = pipeline_node.inputs
    initial_dict = _resolve_initial_dict(metadata_handler, node_inputs)
    try:
      result = _run_resolver_steps(
          initial_dict,
          resolver_steps=node_inputs.resolver_config.resolver_steps,
          store=metadata_handler.store,
          node_info=pipeline_node.node_info)
    except exceptions.SkipSignal:
      return ResolutionSucceeded([])
    except exceptions.IgnoreSignal:
      return Ignore()

    if _is_artifact_multimap(result):
      result = [result]
    elif not _is_list_of_artifact_multimap(result):
      return ResolutionFailed(
          f'Invalid input resolution result: {result}. '
          'Should be Sequence[Mapping[str, Sequence[Artifact]]].')
    valid_inputs = [d for d in result if _is_sufficient(d, node_inputs)]
    return ResolutionSucceeded(valid_inputs)
  except Exception:  # pylint: disable=broad-except
    return ResolutionFailed()


def _run_resolver_strategy(
    input_dict: Any,
    *,
    strategy: resolver.ResolverStrategy,
    input_keys: Iterable[str],
    store: mlmd.MetadataStore,
) -> Dict[str, List[types.Artifact]]:
  """Run single ResolverStrategy with MLMD store."""
  if not _is_artifact_multimap(input_dict):
    raise TypeError(f'Invalid argument type: {input_dict!r}. Must be '
                    'Mapping[str, Sequence[Artifact]].')
  valid_keys = input_keys or set(input_dict.keys())
  valid_inputs = {
      key: list(value)
      for key, value in input_dict.items()
      if key in valid_keys
  }
  bypassed_inputs = {
      key: list(value)
      for key, value in input_dict.items()
      if key not in valid_keys
  }
  result = strategy.resolve_artifacts(store, valid_inputs)
  if result is None:
    raise exceptions.SkipSignal()
  else:
    result.update(bypassed_inputs)
    return result


def _run_resolver_op(
    arg: Any,
    *,
    op: resolver_op.ResolverOp,
    context: resolver_op.InputResolutionContext,
) -> Any:
  """Run single ResolverOp with InputResolutionContext."""
  op.set_context(context)
  return op.apply(arg)


def _run_resolver_steps(
    input_dict: _ArtifactMultimap,
    *,
    resolver_steps: Iterable[pipeline_pb2.ResolverConfig.ResolverStep],
    store: mlmd.MetadataStore,
    node_info: Optional[pipeline_pb2.NodeInfo] = None,
) -> Any:
  """Factory function for creating resolver processors from ResolverConfig."""
  result = input_dict
  context = resolver_op.InputResolutionContext(
      store=store,
      node_info=node_info)
  for step in resolver_steps:
    module_name, class_name = step.class_path.rsplit('.', maxsplit=1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    if issubclass(cls, resolver.ResolverStrategy):
      strategy = cls(**json_utils.loads(step.config_json))
      result = _run_resolver_strategy(
          result,
          strategy=strategy,
          input_keys=step.input_keys,
          store=store)
    elif issubclass(cls, resolver_op.ResolverOp):
      op = cls.create(**json_utils.loads(step.config_json))
      result = _run_resolver_op(result, op=op, context=context)
    else:
      raise TypeError(f'Invalid class {cls}. Should be a subclass of '
                      'tfx.dsl.components.common.resolver.ResolverStrategy or '
                      'tfx.dsl.input_resolution.resolver_op.ResolverOp.')
  return result


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
