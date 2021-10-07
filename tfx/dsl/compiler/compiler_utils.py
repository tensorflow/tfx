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
"""Utility functions for DSL Compiler."""

from typing import List, Optional, Type

from tfx import types
from tfx.dsl.components.base import base_node
from tfx.dsl.components.common import importer
from tfx.dsl.components.common import resolver
from tfx.orchestration import pipeline
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import channel_utils


def set_runtime_parameter_pb(
    pb: pipeline_pb2.RuntimeParameter,
    name: str,
    ptype: Type[types.Property],
    default_value: Optional[types.Property] = None
) -> pipeline_pb2.RuntimeParameter:
  """Helper function to fill a RuntimeParameter proto.

  Args:
    pb: A RuntimeParameter proto to be filled in.
    name: Name to be set at pb.name.
    ptype: The Python type to be set at pb.type.
    default_value: Optional. If provided, it will be pb.default_value.

  Returns:
    A RuntimeParameter proto filled with provided values.
  """
  pb.name = name
  if ptype == int:
    pb.type = pipeline_pb2.RuntimeParameter.Type.INT
    if default_value:
      pb.default_value.int_value = default_value
  elif ptype == float:
    pb.type = pipeline_pb2.RuntimeParameter.Type.DOUBLE
    if default_value:
      pb.default_value.double_value = default_value
  elif ptype == str:
    pb.type = pipeline_pb2.RuntimeParameter.Type.STRING
    if default_value:
      pb.default_value.string_value = default_value
  else:
    raise ValueError("Got unsupported runtime parameter type: {}".format(ptype))
  return pb


def resolve_execution_mode(tfx_pipeline: pipeline.Pipeline):
  """Resolves execution mode for a tfx pipeline.

  Args:
    tfx_pipeline: a TFX pipeline python object assembled by SDK.

  Returns:
    a proto enum reflecting the execution mode of the pipeline.

  Raises:
    RuntimeError: when execution mode is ASYNC while `enable_cache` is true.
    ValueError: when seeing unrecognized execution mode.
  """
  if tfx_pipeline.execution_mode == pipeline.ExecutionMode.SYNC:
    return pipeline_pb2.Pipeline.ExecutionMode.SYNC
  elif tfx_pipeline.execution_mode == pipeline.ExecutionMode.ASYNC:
    if tfx_pipeline.enable_cache:
      raise RuntimeError(
          "Caching is a feature only available to synchronous execution pipelines."
      )
    return pipeline_pb2.Pipeline.ExecutionMode.ASYNC
  else:
    raise ValueError(
        f"Got unsupported execution mode: {tfx_pipeline.execution_mode}")


def is_resolver(node: base_node.BaseNode) -> bool:
  """Helper function to check if a TFX node is a Resolver."""
  return isinstance(node, resolver.Resolver)


def is_importer(node: base_node.BaseNode) -> bool:
  """Helper function to check if a TFX node is an Importer."""
  return isinstance(node, importer.Importer)


def ensure_topological_order(nodes: List[base_node.BaseNode]) -> bool:
  """Helper function to check if nodes are topologically sorted."""
  visited = set()
  for node in nodes:
    for upstream_node in node.upstream_nodes:
      if upstream_node not in visited:
        return False
    visited.add(node)
  return True


def has_task_dependency(tfx_pipeline: pipeline.Pipeline):
  """Checks if a pipeline contains task dependency."""
  producer_map = {}
  for component in tfx_pipeline.components:
    for output_channel in component.outputs.values():
      producer_map[output_channel] = component.id

  for component in tfx_pipeline.components:
    upstream_data_dep_ids = set()
    for value in component.inputs.values():
      # Resolver node is a special case. It sets producer_component_id, but not
      # upstream_nodes. Excludes the case by filtering using producer_map.
      upstream_data_dep_ids.update([
          input_channel.producer_component_id
          for input_channel in channel_utils.get_individual_channels(value)
          if input_channel in producer_map
      ])
    upstream_deps_ids = {node.id for node in component._upstream_nodes}  # pylint: disable=protected-access

    # Compares a node's all upstream nodes and all upstream data dependencies.
    # A task dependency is a dependency between nodes that do not have artifact
    # associated.
    if upstream_data_dep_ids != upstream_deps_ids:
      return True
  return False


def node_context_name(pipeline_context_name: str, node_id: str):
  """Defines the name used to reference a node context in MLMD."""
  return f"{pipeline_context_name}.{node_id}"


def implicit_channel_key(channel: types.Channel):
  """Key of a channel to the node that consumes the channel as input."""
  return f"_{channel.producer_component_id}.{channel.output_key}"


def build_channel_to_key_fn(implicit_keys_map):
  """Builds a function that returns the key of a channel for consumer node."""

  def channel_to_key_fn(channel: types.Channel) -> str:
    implicit_key = implicit_channel_key(channel)
    if implicit_key in implicit_keys_map:
      return implicit_keys_map[implicit_key]
    return implicit_key

  return channel_to_key_fn
