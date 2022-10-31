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

from typing import cast, List, Optional, Sequence, Tuple, Type, Union, Dict, Any

from tfx import types
from tfx.dsl.compiler import constants
from tfx.dsl.components.base import base_node
from tfx.dsl.components.common import importer
from tfx.dsl.components.common import resolver
from tfx.dsl.context_managers import dsl_context_registry
from tfx.dsl.placeholder import placeholder as ph
from tfx.orchestration import data_types_utils
from tfx.orchestration import pipeline
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import channel as channel_types
from tfx.types import channel_utils

from ml_metadata.proto import metadata_store_pb2


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


def set_structural_runtime_parameter_pb(
    pb: pipeline_pb2.StructuralRuntimeParameter,
    str_or_params: Sequence[Union[str, Tuple[str, Type[types.Property]],
                                  Tuple[str, Type[types.Property],
                                        types.Property]]]
) -> pipeline_pb2.StructuralRuntimeParameter:
  """Helper function to fill a StructuralRuntimeParameter proto.

  Args:
    pb: A StructuralRuntimeParameter proto to be filled in.
    str_or_params: A list of either a constant string, or args (name, type,
      default value) to construct a normal runtime parameter. The args will be
      unpacked and passed to set_runtime_parameter_pb. The parts in a structural
      runtime parameter will have the same order as the elements in this list.

  Returns:
    A StructuralRuntimeParameter proto filled with provided values.
  """
  for str_or_param in str_or_params:
    str_or_param_pb = pb.parts.add()
    if isinstance(str_or_param, str):
      str_or_param_pb.constant_value = str_or_param
    else:
      set_runtime_parameter_pb(str_or_param_pb.runtime_parameter, *str_or_param)
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


def pipeline_begin_node_type_name(p: pipeline.Pipeline) -> str:
  """Builds the type name of a Pipeline Begin node."""
  return f"{p.type}{constants.PIPELINE_BEGIN_NODE_SUFFIX}"


def pipeline_end_node_type_name(p: pipeline.Pipeline) -> str:
  """Builds the type name of a Pipeline End node."""
  return f"{p.type}{constants.PIPELINE_END_NODE_SUFFIX}"


def pipeline_begin_node_id(p: pipeline.Pipeline) -> str:
  """Builds the node id of a Pipeline Begin node."""
  return f"{p.id}{constants.PIPELINE_BEGIN_NODE_SUFFIX}"


def pipeline_end_node_id(p: pipeline.Pipeline) -> str:
  """Builds the node id of a Pipeline End node."""
  return pipeline_end_node_id_from_pipeline_id(p.id)


def pipeline_end_node_id_from_pipeline_id(pipeline_id: str) -> str:
  """Builds the node id of a Pipeline End node."""
  return f"{pipeline_id}{constants.PIPELINE_END_NODE_SUFFIX}"


def node_context_name(pipeline_context_name: str, node_id: str):
  """Defines the name used to reference a node context in MLMD."""
  return f"{pipeline_context_name}.{node_id}"


def implicit_channel_key(channel: types.BaseChannel):
  """Key of a channel to the node that consumes the channel as input."""
  if isinstance(channel, channel_types.PipelineInputChannel):
    channel = cast(channel_types.PipelineInputChannel, channel)
    return f"_{channel.pipeline.id}.{channel.output_key}"
  elif isinstance(channel, types.Channel):
    if channel.producer_component_id and channel.output_key:
      return f"_{channel.producer_component_id}.{channel.output_key}"
    raise ValueError(
        "Cannot create implicit input key for Channel that has no"
        "producer_component_id and output_key.")
  else:
    raise ValueError("Unsupported channel type for implicit channel key.")


def build_channel_to_key_fn(implicit_keys_map):
  """Builds a function that returns the key of a channel for consumer node."""

  def channel_to_key_fn(channel: types.Channel) -> str:
    implicit_key = implicit_channel_key(channel)
    if implicit_key in implicit_keys_map:
      return implicit_keys_map[implicit_key]
    return implicit_key

  return channel_to_key_fn


def validate_dynamic_exec_ph_operator(placeholder: ph.ArtifactPlaceholder):
  # Supported format for dynamic exec prop:
  # component.output['ouput_key'].future()[0].value
  if len(placeholder._operators) != 2:  # pylint: disable=protected-access
    raise ValueError("dynamic exec property should contain two placeholder "
                     "operator, while pass %d operaters" %
                     len(placeholder._operators))  # pylint: disable=protected-access
  if (not isinstance(placeholder._operators[0], ph._IndexOperator) or  # pylint: disable=protected-access
      not isinstance(placeholder._operators[1], ph._ArtifactValueOperator)):  # pylint: disable=protected-access
    raise ValueError("dynamic exec property should be in form of "
                     "component.output[\'ouput_key\'].future()[0].value")


def output_spec_from_channel(channel: types.BaseChannel,
                             node_id: str) -> pipeline_pb2.OutputSpec:
  """Generates OutputSpec proto given OutputChannel."""
  result = pipeline_pb2.OutputSpec()
  artifact_type = channel.type._get_artifact_type()  # pylint: disable=protected-access
  result.artifact_spec.type.CopyFrom(artifact_type)

  if not isinstance(channel, channel_types.OutputChannel):
    return result
  output_channel = cast(channel_types.OutputChannel, channel)

  # Compile OutputSpec.artifact_spec.additional_parameters.
  for property_name, property_value in (
      output_channel.additional_properties.items()):
    _check_property_value_type(property_name, property_value, artifact_type)
    value_field = result.artifact_spec.additional_properties[
        property_name].field_value
    try:
      data_types_utils.set_metadata_value(value_field, property_value)
    except ValueError:
      raise ValueError(
          f"Node {node_id} got unsupported parameter {property_name} with type "
          f"{type(property_value)}.") from ValueError

  # Compile OutputSpec.artifact_spec.additional_custom_parameters.
  for property_name, property_value in (
      output_channel.additional_custom_properties.items()):
    value_field = result.artifact_spec.additional_custom_properties[
        property_name].field_value
    try:
      data_types_utils.set_metadata_value(value_field, property_value)
    except ValueError:
      raise ValueError(
          f"Node {node_id} got unsupported parameter {property_name} with type "
          f"{type(property_value)}.") from ValueError

  # Compile OutputSpec.garbage_collection_policy
  # pylint: disable=protected-access
  if output_channel._garbage_collection_policy is not None:
    result.garbage_collection_policy.CopyFrom(
        output_channel._garbage_collection_policy)

  # Compile OutputSpec.external_artifacts_uris
  if output_channel._predefined_artifact_uris is not None:
    result.artifact_spec.external_artifact_uris.extend(
        output_channel._predefined_artifact_uris)

  return result


def _check_property_value_type(property_name: str,
                               property_value: types.Property,
                               artifact_type: metadata_store_pb2.ArtifactType):
  prop_value_type = data_types_utils.get_metadata_value_type(property_value)
  if prop_value_type != artifact_type.properties[property_name]:
    raise TypeError(
        "Unexpected value type of property '{}' in output artifact '{}': "
        "Expected {} but given {} (value:{!r})".format(
            property_name, artifact_type.name,
            metadata_store_pb2.PropertyType.Name(
                artifact_type.properties[property_name]),
            metadata_store_pb2.PropertyType.Name(prop_value_type),
            property_value))


class _PipelineEnd(base_node.BaseNode):
  """Virtual pipeline end node.

  While the pipeline end node does not exists nor accessible in DSL, having a
  PipelineEnd class helps generalizing the compilation.

  Supported features:
    - Node ID (which is "{pipeline_id}_end")
    - Node type name
    - Node inputs, which comes from the inner pipeline (i.e. the raw value
        used from Pipeline(outputs=raw_outputs))
    - Node outputs, which is the wrapped OutputChannel of the pipeline node
        that is visible from the outer pipeline

  Not yet supported:
    - upstream/downstream nodes relationship
  """

  def __init__(self, p: pipeline.Pipeline):
    super().__init__()
    self._pipeline = p
    self.with_id(pipeline_end_node_id(p))

  @property
  def type(self) -> str:
    return pipeline_end_node_type_name(self._pipeline)

  @property
  def inputs(self) -> Dict[str, channel_types.BaseChannel]:
    return {
        key: pipeline_output_channel.wrapped
        for key, pipeline_output_channel in self._pipeline.outputs.items()
    }

  @property
  def outputs(self) -> Dict[str, channel_types.BaseChannel]:
    return self._pipeline.outputs

  @property
  def exec_properties(self) -> Dict[str, Any]:
    return {}


def create_pipeline_end_node(p: pipeline.Pipeline) -> _PipelineEnd:
  """Create a dummy pipeline end node.

  pipeline end node does not appear in pipeline DSL but only in the pipeline IR.
  To generalizes compilation process for the pipeline end node, create a dummy
  BaseNode whose inputs are set as pipeline.outputs.

  Args:
    p: A Pipeline instance whose pipeline end node will be created.

  Returns:
    a pipeline end node.
  """
  with dsl_context_registry.use_registry(p.dsl_context_registry):
    return _PipelineEnd(p)
