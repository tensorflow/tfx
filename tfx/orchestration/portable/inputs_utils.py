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
from typing import Dict, Sequence, Union

from absl import logging
from tfx import types
from tfx.dsl.compiler import placeholder_utils
from tfx.orchestration import mlmd_connection_manager as mlmd_cm
from tfx.orchestration.portable import data_types
from tfx.orchestration.portable.input_resolution import exceptions
from tfx.orchestration.portable.input_resolution import node_inputs_resolver
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import typing_utils


class Trigger(tuple, Sequence[typing_utils.ArtifactMultiMap]):
  """Input resolution result of list of dict."""

  def __new__(cls, resolved_inputs: Sequence[typing_utils.ArtifactMultiMap]):
    assert resolved_inputs, 'resolved inputs should be non-empty.'
    return super().__new__(cls, resolved_inputs)


class Skip(tuple, Sequence[typing_utils.ArtifactMultiMap]):
  """Input resolution result of empty list."""

  def __new__(cls):
    return super().__new__(cls)


def resolve_input_artifacts(
    *,
    pipeline_node: pipeline_pb2.PipelineNode,
    metadata_handler: mlmd_cm.HandleLike,
) -> Union[Trigger, Skip]:
  """Resolve input artifacts according to a pipeline node IR definition.

  Args:
    pipeline_node: Current PipelineNode on which input resolution is running.
    metadata_handler: Metadata or MLMDConnectionManager instance for handling
      mlmd db connections.

  Raises:
    InputResolutionError: If input resolution went wrong.

  Returns:
    Trigger: a non-empty list of input dicts. All resolved input dicts should be
        executed.
    Skip: an empty list. Should effectively skip the current component
        execution.
  """
  try:
    node_inputs = pipeline_node.inputs
    resolved = node_inputs_resolver.resolve(metadata_handler, node_inputs)
    return Trigger(resolved) if resolved else Skip()
  except exceptions.SkipSignal as e:
    logging.info('Input resolution skipped; reason = %s', e)
    return Skip()
  except exceptions.InputResolutionError as e:
    error_msg = (
        f'Error while resolving inputs for {pipeline_node.node_info.id}: {e}')
    e.args = (error_msg,)
    raise
  except Exception as e:
    raise exceptions.InputResolutionError(
        f'Error while resolving inputs for {pipeline_node.node_info.id}') from e


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
    if value.HasField('placeholder'):
      continue
    if not value.HasField('field_value'):
      raise RuntimeError('Parameter value not ready for %s' % key)
    result[key] = value

  return result


def resolve_dynamic_parameters(
    node_parameters: pipeline_pb2.NodeParameters,
    input_artifacts: typing_utils.ArtifactMultiMap
) -> Dict[str, types.ExecPropertyTypes]:
  """Resolves dynamic execution properties given the input artifacts.

  Args:
    node_parameters: The spec to get parameters.
    input_artifacts: The input dict.

  Returns:
    A Dict of resolved dynamic parameters.

  Raises:
    InputResolutionError: If the resolution of dynamic exec property fails.
  """
  result = {}
  converted_input_artifacts = {}
  for key, value in input_artifacts.items():
    converted_input_artifacts[key] = list(value)
  for key, value in node_parameters.parameters.items():
    if value.HasField('placeholder'):
      execution_info = data_types.ExecutionInfo(
          input_dict=converted_input_artifacts,
          output_dict={},
          exec_properties={})
      context = placeholder_utils.ResolutionContext(
          exec_info=execution_info)
      try:
        resolved_val = placeholder_utils.resolve_placeholder_expression(
            value.placeholder, context)
        if resolved_val is None:
          raise exceptions.InvalidArgument(
              f'Cannot find input artifact to resolve dynamic exec property. '
              f'Key: {key}. '
              f'Value: {placeholder_utils.debug_str(value.placeholder)}')
        result[key] = resolved_val
      except Exception as e:
        raise exceptions.InvalidArgument(
            f'Failed to resolve dynamic exec property. Key: {key}. '
            f'Value: {placeholder_utils.debug_str(value.placeholder)}') from e

  return result
