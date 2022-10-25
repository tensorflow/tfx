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
"""Compiler submodule specialized for NodeInputs."""

from typing import Type, cast, Mapping

from tfx import types
from tfx.dsl.compiler import compiler_context
from tfx.dsl.compiler import compiler_utils
from tfx.dsl.compiler import constants
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import base_node
from tfx.dsl.experimental.conditionals import conditional
from tfx.dsl.input_resolution import resolver_op
from tfx.dsl.input_resolution.ops import ops
from tfx.dsl.placeholder import placeholder
from tfx.orchestration import data_types_utils
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import channel as channel_types
from tfx.types import resolved_channel
from tfx.types import value_artifact
from tfx.utils import deprecation_utils
from tfx.utils import name_utils
from tfx.utils import typing_utils


def _compile_input_graph(
    pipeline_ctx: compiler_context.PipelineContext,
    tfx_node: base_node.BaseNode,
    channel: resolved_channel.ResolvedChannel,
    result: pipeline_pb2.NodeInputs,
) -> str:
  """Compiles ResolvedChannel.output_node as InputGraph and returns its ID."""
  input_graph_id = pipeline_ctx.get_node_context(tfx_node).get_input_graph_key(
      channel.output_node, channel.for_each_context)
  if input_graph_id in result.input_graphs:
    return input_graph_id

  node_to_ids = {}
  input_graph = result.input_graphs[input_graph_id]

  def issue_node_id(prefix: str):
    return prefix + str(len(node_to_ids) + 1)

  def get_node_id(node: resolver_op.Node):
    if node in node_to_ids:
      return node_to_ids[node]

    if isinstance(node, resolver_op.InputNode):
      return compile_input_node(cast(resolver_op.InputNode, node))
    elif isinstance(node, resolver_op.DictNode):
      return compile_dict_node(cast(resolver_op.DictNode, node))
    elif isinstance(node, resolver_op.OpNode):
      return compile_op_node(cast(resolver_op.OpNode, node))
    else:
      raise NotImplementedError(
          'Expected `node` to be one of InputNode, DictNode, or OpNode '
          f'but got `{type(node).__name__}` type.')

  def compile_input_node(input_node: resolver_op.InputNode):
    # TODO(b/236140795): Remove once InputNode.wrapped is always BaseChannel.
    if typing_utils.is_compatible(
        input_node.wrapped, Mapping[str, channel_types.BaseChannel]):
      return get_node_id(resolver_op.DictNode({
          k: resolver_op.InputNode(
              v, output_data_type=resolver_op.DataType.ARTIFACT_LIST)
          for k, v in input_node.wrapped.items()
      }))

    node_id = issue_node_id(prefix='input_')
    node_to_ids[input_node] = node_id
    input_key = (
        pipeline_ctx.get_node_context(tfx_node)
        .get_input_key(input_node.wrapped))
    _compile_input_spec(
        pipeline_ctx=pipeline_ctx,
        tfx_node=tfx_node,
        input_key=input_key,
        channel=input_node.wrapped,
        hidden=True,
        min_count=0,
        result=result)
    input_graph.nodes[node_id].input_node.input_key = input_key
    input_graph.nodes[node_id].output_data_type = input_node.output_data_type
    return node_id

  def compile_dict_node(dict_node: resolver_op.DictNode):
    node_id = issue_node_id(prefix='dict_')
    node_to_ids[dict_node] = node_id
    input_graph.nodes[node_id].dict_node.node_ids.update({
        key: get_node_id(child_node)
        for key, child_node in dict_node.nodes.items()
    })
    input_graph.nodes[node_id].output_data_type = (
        pipeline_pb2.InputGraph.DataType.ARTIFACT_MULTIMAP)
    return node_id

  def compile_op_node(op_node: resolver_op.OpNode):
    node_id = issue_node_id(prefix='op_')
    node_to_ids[op_node] = node_id
    op_node_ir = input_graph.nodes[node_id].op_node
    if issubclass(op_node.op_type, resolver_op.ResolverOp):
      op_node_ir.op_type = op_node.op_type.canonical_name
    else:
      op_node_ir.op_type = name_utils.get_full_name(
          deprecation_utils.get_first_nondeprecated_class(op_node.op_type))
    for n in op_node.args:
      op_node_ir.args.add().node_id = get_node_id(n)
    for k, v in op_node.kwargs.items():
      data_types_utils.set_parameter_value(op_node_ir.kwargs[k].value, v)
    input_graph.nodes[node_id].output_data_type = op_node.output_data_type
    return node_id

  input_graph.result_node = get_node_id(channel.output_node)

  return input_graph_id


def _compile_channel_pb(
    artifact_type: Type[types.Artifact],
    pipeline_name: str,
    node_id: str,
    output_key: str,
    result: pipeline_pb2.InputSpec.Channel,
):
  """Compile InputSpec.Channel with an artifact type and context filters."""
  mlmd_artifact_type = artifact_type._get_artifact_type()  # pylint: disable=protected-access
  result.artifact_query.type.CopyFrom(mlmd_artifact_type)
  result.artifact_query.type.ClearField('properties')

  ctx = result.context_queries.add()
  ctx.type.name = constants.PIPELINE_CONTEXT_TYPE_NAME
  ctx.name.field_value.string_value = pipeline_name

  if node_id:
    ctx = result.context_queries.add()
    ctx.type.name = constants.NODE_CONTEXT_TYPE_NAME
    ctx.name.field_value.string_value = compiler_utils.node_context_name(
        pipeline_name, node_id)

  if output_key:
    result.output_key = output_key


def _convert_loop_var_to_resolved_channel(
    loop_var_channel: channel_types.LoopVarChannel,
) -> resolved_channel.ResolvedChannel:
  """Convert `LoopVarChannel` to equivalent `ResolvedChannel`."""
  input_node = resolver_op.InputNode(
      loop_var_channel.wrapped,
      output_data_type=resolver_op.DataType.ARTIFACT_LIST)
  dict_node = resolver_op.DictNode({'out': input_node})
  output_node = resolver_op.OpNode(
      op_type=ops.Unnest,
      output_data_type=resolver_op.DataType.ARTIFACT_MULTIMAP_LIST,
      args=[dict_node],
      kwargs={'key': 'out'})
  return resolved_channel.ResolvedChannel(
      artifact_type=loop_var_channel.type,
      output_node=output_node,
      output_key='out')


def _compile_input_spec(
    *,
    pipeline_ctx: compiler_context.PipelineContext,
    tfx_node: base_node.BaseNode,
    input_key: str,
    channel: channel_types.BaseChannel,
    hidden: bool,
    min_count: int,
    result: pipeline_pb2.NodeInputs,
) -> None:
  """Compiles `BaseChannel` into `InputSpec` at `result.inputs[input_key]`.

  Args:
    pipeline_ctx: A `PipelineContext`.
    tfx_node: A `BaseNode` instance from pipeline DSL.
    input_key: An input key that the compiled `InputSpec` would be stored with.
    channel: A `BaseChannel` instance to compile.
    hidden: If true, this sets `InputSpec.hidden = True`. If the channel is
        already compiled, then it has no effect.
    min_count: Minimum number of artifacts that should be resolved for this
        input key. If min_count is not met during the input resolution, it is
        considered as an error.
    result: A `NodeInputs` proto to which the compiled result would be written.
  """
  if input_key in result.inputs:
    # Already compiled. This can happen during compiling another input channel
    # from the same resolver function output.
    return

  if channel in pipeline_ctx.channels:
    # OutputChannel or PipelineInputChannel from the same pipeline has already
    # compiled IR in context.channels
    result.inputs[input_key].channels.append(pipeline_ctx.channels[channel])

  elif isinstance(channel, channel_types.PipelineOutputChannel):
    # This is the case when PipelineInputs uses pipeline.outputs where the
    # pipeline is external (i.e. not a parent or sibling pipeline) thus
    # pipeline run cannot be synced.
    channel = cast(channel_types.PipelineOutputChannel, channel)
    _compile_channel_pb(
        artifact_type=channel.type,
        pipeline_name=channel.pipeline.id,
        node_id=channel.wrapped.producer_component_id,
        output_key=channel.output_key,
        result=result.inputs[input_key].channels.add())

  elif isinstance(channel, channel_types.Channel):
    channel = cast(channel_types.Channel, channel)
    _compile_channel_pb(
        artifact_type=channel.type,
        pipeline_name=pipeline_ctx.pipeline_info.pipeline_name,
        node_id=channel.producer_component_id,
        output_key=channel.output_key,
        result=result.inputs[input_key].channels.add())

  elif isinstance(channel, channel_types.UnionChannel):
    channel = cast(channel_types.UnionChannel, channel)
    mixed_inputs = result.inputs[input_key].mixed_inputs
    mixed_inputs.method = pipeline_pb2.InputSpec.Mixed.Method.UNION
    for sub_channel in channel.channels:
      sub_key = (
          pipeline_ctx.get_node_context(tfx_node).get_input_key(sub_channel))
      mixed_inputs.input_keys.append(sub_key)
      _compile_input_spec(
          pipeline_ctx=pipeline_ctx,
          tfx_node=tfx_node,
          input_key=sub_key,
          channel=sub_channel,
          hidden=True,
          min_count=0,
          result=result)

  elif isinstance(channel, resolved_channel.ResolvedChannel):
    channel = cast(resolved_channel.ResolvedChannel, channel)
    input_graph_ref = result.inputs[input_key].input_graph_ref
    input_graph_ref.graph_id = _compile_input_graph(
        pipeline_ctx, tfx_node, channel, result)
    if channel.output_key:
      input_graph_ref.key = channel.output_key

  # TODO(b/239761275): Remove LoopVarChannel type.
  elif isinstance(channel, channel_types.LoopVarChannel):
    channel = _convert_loop_var_to_resolved_channel(
        cast(channel_types.LoopVarChannel, channel))
    _compile_input_spec(
        pipeline_ctx=pipeline_ctx,
        tfx_node=tfx_node,
        input_key=input_key,
        channel=channel,
        hidden=hidden,
        min_count=0,
        result=result)

  else:
    raise NotImplementedError(
        f'Node {tfx_node.id} got unsupported channel type {channel!r} for '
        f'inputs[{input_key!r}].')

  if hidden:
    result.inputs[input_key].hidden = True
  if min_count:
    result.inputs[input_key].min_count = min_count


def _compile_conditionals(
    context: compiler_context.PipelineContext,
    tfx_node: base_node.BaseNode,
    result: pipeline_pb2.NodeInputs,
) -> None:
  """Compiles conditionals attached to the BaseNode.

  It also compiles the channels that each conditional predicate depends on. If
  the channel already appears in the node inputs, reuses it. Otherwise, creates
  an implicit hidden input.

  Args:
    context: A `PipelineContext`.
    tfx_node: A `BaseNode` instance from pipeline DSL.
    result: A `NodeInputs` proto to which the compiled result would be written.
  """
  try:
    contexts = context.dsl_context_registry.get_contexts(tfx_node)
  except ValueError:
    return

  for dsl_context in contexts:
    if not isinstance(dsl_context, conditional.CondContext):
      continue
    cond_context = cast(conditional.CondContext, dsl_context)
    for channel in cond_context.predicate.dependent_channels():
      _compile_input_spec(
          pipeline_ctx=context,
          tfx_node=tfx_node,
          input_key=context.get_node_context(tfx_node).get_input_key(channel),
          channel=channel,
          hidden=False,
          min_count=1,
          result=result)
    cond_id = context.get_conditional_id(cond_context)
    expr = cond_context.predicate.encode_with_keys(
        context.get_node_context(tfx_node).get_input_key)
    result.conditionals[cond_id].placeholder_expression.CopyFrom(expr)


def _compile_inputs_for_dynamic_properties(
    context: compiler_context.PipelineContext,
    tfx_node: base_node.BaseNode,
    result: pipeline_pb2.NodeInputs,
) -> None:
  """Compiles additional InputSpecs used in dynamic properties.

  Dynamic properties are the execution properties whose value comes from the
  artifact value. Becauese of that, dynamic property resolution happens after
  the input resolution at orchestrator, so input resolution should include the
  resolved artifacts for the channel on which dynamic properties depend (thus
  `_compile_channel(hidden=False)`).

  Args:
    context: A `PipelineContext`.
    tfx_node: A `BaseNode` instance from pipeline DSL.
    result: A `NodeInputs` proto to which the compiled result would be written.
  """
  for exec_property in tfx_node.exec_properties.values():
    if not isinstance(exec_property, placeholder.ChannelWrappedPlaceholder):
      continue
    if not typing_utils.is_compatible(
        exec_property.channel.type, Type[value_artifact.ValueArtifact]):
      raise ValueError(
          'Dynamic execution property only supports ValueArtifact typed '
          f'channel. Got {exec_property.channel.type.TYPE_NAME}.')
    _compile_input_spec(
        pipeline_ctx=context,
        tfx_node=tfx_node,
        input_key=(
            context.get_node_context(tfx_node)
            .get_input_key(exec_property.channel)),
        channel=exec_property.channel,
        hidden=False,
        min_count=1,
        result=result)


def compile_node_inputs(
    context: compiler_context.PipelineContext,
    tfx_node: base_node.BaseNode,
    result: pipeline_pb2.NodeInputs,
) -> None:
  """Compile NodeInputs from BaseNode input channels."""
  # Compile DSL node inputs.
  for input_key, channel in tfx_node.inputs.items():
    if (compiler_utils.is_resolver(tfx_node) or
        (isinstance(tfx_node, base_component.BaseComponent) and
         tfx_node.spec.is_allow_empty_input(input_key))):
      min_count = 0
    else:
      min_count = 1
    _compile_input_spec(
        pipeline_ctx=context,
        tfx_node=tfx_node,
        input_key=input_key,
        channel=channel,
        hidden=False,
        min_count=min_count,
        result=result)

  # Add implicit input channels that are used in conditionals.
  _compile_conditionals(context, tfx_node, result)

  # Add implicit input channels that are used in dynamic properties.
  _compile_inputs_for_dynamic_properties(context, tfx_node, result)
