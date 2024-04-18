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
"""Compiles a TFX pipeline into a TFX DSL IR proto."""
import itertools
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Type, cast

from tfx import types
from tfx.dsl.compiler import compiler_context
from tfx.dsl.compiler import compiler_utils
from tfx.dsl.compiler import constants
from tfx.dsl.compiler import node_contexts_compiler
from tfx.dsl.compiler import node_execution_options_utils
from tfx.dsl.compiler import node_inputs_compiler
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import base_driver
from tfx.dsl.components.base import base_node
from tfx.dsl.experimental.node_execution_options import utils as execution_options_utils
from tfx.dsl.placeholder import placeholder
from tfx.orchestration import data_types
from tfx.orchestration import data_types_utils
from tfx.orchestration import pipeline
from tfx.proto.orchestration import executable_spec_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import channel as channel_types
from tfx.types import channel_utils
from tfx.utils import deprecation_utils
from tfx.utils import name_utils


class Compiler:
  """Compiles a TFX pipeline or a component into a uDSL IR proto."""

  def __init__(self, use_input_v2: bool = True):
    assert use_input_v2, "use_input_v2=False is no longer supported."

  def _compile_pipeline_begin_node(
      self, p: pipeline.Pipeline,
      pipeline_ctx: compiler_context.PipelineContext,
  ) -> pipeline_pb2.PipelineNode:
    """Compiles a PipelineBegin node for a composable pipeline."""
    node = pipeline_pb2.PipelineNode()

    # Step 1: Node info
    node.node_info.type.name = compiler_utils.pipeline_begin_node_type_name(p)
    node.node_info.id = compiler_utils.pipeline_begin_node_id(p)

    # Step 2: Node Context
    # Inner pipeline's contexts.
    node.contexts.CopyFrom(
        node_contexts_compiler.compile_node_contexts(
            pipeline_ctx,
            node.node_info.id,
        )
    )

    # Step 3: Node inputs
    # Pipeline node inputs are stored as the inputs of the PipelineBegin node.
    node_inputs_compiler.compile_node_inputs(
        pipeline_ctx.parent, p, node.inputs)

    # Step 4: Node outputs
    # PipeineBegin node's outputs are the same as its inputs,
    # i.e., the composable pipeline's inputs.
    internal_inputs = p._inputs.inputs if p._inputs else {}  # pylint: disable=protected-access
    _set_node_outputs(node, internal_inputs)
    pipeline_ctx.channels.update(
        _generate_input_spec_for_outputs(node, internal_inputs))

    # Pipeline Begin node has no parameters.

    # Step 5: Upstream/Downstream nodes
    # PipelineBegin node's upstreams nodes are the inner pipeline's upstream
    # nodes, i.e., the producer nodes of inner pipeline's inputs.
    # Outmost pipeline's PipelineBegin node does not has upstream nodes.
    if pipeline_ctx.is_subpipeline:
      upstreams = set(
          _find_runtime_upstream_node_ids(pipeline_ctx.parent, p))
      if _begin_node_is_upstream(p, pipeline_ctx.parent.pipeline):
        upstreams.add(
            compiler_utils.pipeline_begin_node_id(
                pipeline_ctx.parent.pipeline))
      # Sort node ids so that compiler generates consistent results.
      node.upstream_nodes.extend(sorted(upstreams))

    # Set node execution options based on pipeline-level NodeExecutionOptions.
    # Because _compile_pipeline_begin_node may be called if either the pipeline
    # is a subpipeline OR the pipeline has inputs we can *only* pass in the
    # parent context if this begin node is for a subpipeline.
    if pipeline_ctx.is_subpipeline:
      execution_options_ctx = pipeline_ctx.parent
    else:
      execution_options_ctx = pipeline_ctx
    _set_node_execution_options(node, p, execution_options_ctx, p.enable_cache)

    # PipelineBegin node's downstream nodes are the nodes in the inner pipeline
    # that consumes pipeline's input channels.
    result = set()
    for inner_node in p.components:
      if _begin_node_is_upstream(inner_node, p):
        result.add(inner_node.id)
    # Sort node ids so that compiler generates consistent results.
    node.downstream_nodes.extend(sorted(result))

    return node

  def _compile_pipeline_end_node(
      self, p: pipeline.Pipeline,
      pipeline_ctx: compiler_context.PipelineContext,
  ) -> pipeline_pb2.PipelineNode:
    """Compiles a PipelineEnd node for a composable pipeline."""
    node = pipeline_pb2.PipelineNode()

    # Step 1: Node info
    node.node_info.type.name = compiler_utils.pipeline_end_node_type_name(p)
    node.node_info.id = compiler_utils.pipeline_end_node_id(p)

    # Step 2: Node Context
    # Inner pipeline's contexts.
    node.contexts.CopyFrom(
        node_contexts_compiler.compile_node_contexts(
            pipeline_ctx,
            node.node_info.id,
        )
    )

    # Step 3: Node inputs
    node_inputs_compiler.compile_node_inputs(
        pipeline_ctx,
        compiler_utils.create_pipeline_end_node(p),
        node.inputs)

    # Step 4: Node outputs
    # PipeineEnd node's outputs are the same as inner pipeline's outputs.
    _set_node_outputs(node, p.outputs)
    if pipeline_ctx.is_subpipeline:
      pipeline_ctx.parent.channels.update(
          _generate_input_spec_for_pipeline_outputs(node, p))

    # PipelineEnd node does not have parameters.

    # Step 5: Upstream/Downstream nodes
    # PipelineEnd node's upstream nodes are the nodes in the inner pipeline
    # that produces pipeline's output channels.
    result = set()
    for pipeline_output in p.outputs.values():
      for inner_node in p.components:
        if pipeline_output.wrapped in inner_node.outputs.values():
          result.add(inner_node.id)
    # Sort node ids so that compiler generates consistent results.
    node.upstream_nodes.extend(sorted(result))

    # PipelineEnd node's downstream nodes are the inner pipeline's downstream
    # nodes, i.e., the consumer nodes of inner pipeline's outputs.
    downstreams = set(_find_runtime_downstream_node_ids(pipeline_ctx.parent, p))
    if pipeline_ctx.is_subpipeline and _end_node_is_downstream(
        p, pipeline_ctx.parent.pipeline):
      downstreams.add(
          compiler_utils.pipeline_end_node_id(pipeline_ctx.parent.pipeline))
    # Sort node ids so that compiler generates consistent results.
    node.downstream_nodes.extend(sorted(downstreams))

    return node

  def _compile_node(
      self,
      tfx_node: base_node.BaseNode,
      pipeline_ctx: compiler_context.PipelineContext,
      deployment_config: pipeline_pb2.IntermediateDeploymentConfig,
      enable_cache: bool,
  ) -> pipeline_pb2.PipelineNode:
    """Compiles an individual TFX node into a PipelineNode proto.

    Args:
      tfx_node: A TFX node.
      pipeline_ctx: Resources needed to compile the node.
      deployment_config: Intermediate deployment config to set. Will include
        related specs for executors, drivers and platform specific configs.
      enable_cache: whether cache is enabled

    Raises:
      TypeError: When supplied tfx_node has values of invalid type.

    Returns:
      A PipelineNode proto that encodes information of the node.
    """
    node = pipeline_pb2.PipelineNode()

    # Step 1: Node info
    node.node_info.type.name = tfx_node.type
    if isinstance(tfx_node,
                  base_component.BaseComponent) and tfx_node.type_annotation:
      node.node_info.type.base_type = (
          tfx_node.type_annotation.MLMD_SYSTEM_BASE_TYPE)
    node.node_info.id = tfx_node.id

    # Step 2: Node Context
    node.contexts.CopyFrom(
        node_contexts_compiler.compile_node_contexts(
            pipeline_ctx,
            node.node_info.id,
        )
    )

    # Step 3: Node inputs
    node_inputs_compiler.compile_node_inputs(
        pipeline_ctx, tfx_node, node.inputs)

    # Step 4: Node outputs
    if (isinstance(tfx_node, base_component.BaseComponent) or
        compiler_utils.is_importer(tfx_node)):
      _set_node_outputs(node, tfx_node.outputs)
    pipeline_ctx.channels.update(
        _generate_input_spec_for_outputs(node, tfx_node.outputs))

    # Step 5: Node parameters
    if not compiler_utils.is_resolver(tfx_node):
      _set_node_parameters(node, tfx_node)

    # Step 6: Executor spec and optional driver spec for components
    if isinstance(tfx_node, base_component.BaseComponent):
      executor_spec = tfx_node.executor_spec.encode(
          component_spec=tfx_node.spec)
      deployment_config.executor_specs[tfx_node.id].Pack(executor_spec)

      # TODO(b/163433174): Remove specialized logic once generalization of
      # driver spec is done.
      if tfx_node.driver_class != base_driver.BaseDriver:
        driver_class_path = _fully_qualified_name(tfx_node.driver_class)
        driver_spec = executable_spec_pb2.PythonClassExecutableSpec()
        driver_spec.class_path = driver_class_path
        deployment_config.custom_driver_specs[tfx_node.id].Pack(driver_spec)

    # Step 7: Upstream/Downstream nodes
    upstreams = set(_find_runtime_upstream_node_ids(pipeline_ctx, tfx_node))
    if _begin_node_is_upstream(tfx_node, pipeline_ctx.pipeline):
      upstreams.add(
          compiler_utils.pipeline_begin_node_id(pipeline_ctx.pipeline))
    # Sort node ids so that compiler generates consistent results.
    node.upstream_nodes.extend(sorted(upstreams))

    downstreams = set(
        _find_runtime_downstream_node_ids(pipeline_ctx, tfx_node))
    if _end_node_is_downstream(tfx_node, pipeline_ctx.pipeline):
      downstreams.add(
          compiler_utils.pipeline_end_node_id(pipeline_ctx.pipeline))
    # Sort node ids so that compiler generates consistent results.
    node.downstream_nodes.extend(sorted(downstreams))

    # Step 8: Node execution options and triggers.
    _set_node_execution_options(node, tfx_node, pipeline_ctx, enable_cache)

    # Step 9: Per-node platform config
    if isinstance(tfx_node, base_component.BaseComponent):
      tfx_component = cast(base_component.BaseComponent, tfx_node)
      if tfx_component.platform_config:
        deployment_config.node_level_platform_configs[tfx_node.id].Pack(
            tfx_component.platform_config)

    return node

  def compile(
      self,
      tfx_pipeline: pipeline.Pipeline,
      parent_pipeline_ctx: Optional[compiler_context.PipelineContext] = None,
  ) -> pipeline_pb2.Pipeline:
    """Compiles a tfx pipeline into uDSL proto.

    Args:
      tfx_pipeline: A TFX pipeline.
      parent_pipeline_ctx: Optional PipelineContext that includes info for
        the immediate parent pipeline. This is mainly used by a pipeline begin
        node get info for artifacts from its parent pipeline.

    Returns:
      A Pipeline proto that encodes all necessary information of the pipeline.
    """
    # Prepare pipeline compiler context.
    pipeline_ctx = compiler_context.PipelineContext(
        tfx_pipeline, parent_pipeline_ctx)

    tfx_pipeline.finalize()
    _validate_pipeline(tfx_pipeline, pipeline_ctx.parent_pipelines)

    pipeline_pb = pipeline_pb2.Pipeline()
    pipeline_pb.pipeline_info.id = pipeline_ctx.pipeline_info.pipeline_name
    pipeline_pb.execution_mode = pipeline_ctx.execution_mode

    if isinstance(
        pipeline_ctx.pipeline_info.pipeline_root, placeholder.Placeholder):
      pipeline_pb.runtime_spec.pipeline_root.placeholder.CopyFrom(
          pipeline_ctx.pipeline_info.pipeline_root.encode())
    else:
      # Unless an inner pipeline specified its own pipeline root, it inherits
      # from its closest parent pipeline that has a pipelie root defined.
      pipeline_root_str = ""
      if tfx_pipeline.pipeline_info.pipeline_root:
        pipeline_root_str = tfx_pipeline.pipeline_info.pipeline_root
      else:
        for parent_pipeline in reversed(pipeline_ctx.parent_pipelines):
          if parent_pipeline.pipeline_info.pipeline_root:
            pipeline_root_str = parent_pipeline.pipeline_info.pipeline_root
            break
      compiler_utils.set_runtime_parameter_pb(
          pipeline_pb.runtime_spec.pipeline_root.runtime_parameter,
          constants.PIPELINE_ROOT_PARAMETER_NAME, str, pipeline_root_str)

    if pipeline_pb.execution_mode == pipeline_pb2.Pipeline.ExecutionMode.SYNC:
      # TODO(kennethyang): Miragte all pipeline run ids to structural runtime
      # parameter. Currently only subpipelines use structural runtime
      # parameter for IR textproto compatibility.
      if not pipeline_ctx.is_root:
        compiler_utils.set_structural_runtime_parameter_pb(
            pipeline_pb.runtime_spec.pipeline_run_id
            .structural_runtime_parameter, [
                f"{pipeline_pb.pipeline_info.id}_",
                (constants.PIPELINE_RUN_ID_PARAMETER_NAME, str)
            ])
      else:
        compiler_utils.set_runtime_parameter_pb(
            pipeline_pb.runtime_spec.pipeline_run_id.runtime_parameter,
            constants.PIPELINE_RUN_ID_PARAMETER_NAME, str)

    deployment_config = pipeline_pb2.IntermediateDeploymentConfig()
    if tfx_pipeline.metadata_connection_config:
      deployment_config.metadata_connection_config.Pack(
          tfx_pipeline.metadata_connection_config)

    # Inner pipelines of a composable pipeline, or a outmost pipeline with
    # pipeline-level inputs have pipeline begin nodes.
    if not pipeline_ctx.is_root or tfx_pipeline._inputs:  # pylint: disable=protected-access
      pipeline_begin_node_pb = self._compile_pipeline_begin_node(
          tfx_pipeline, pipeline_ctx)
      pipeline_or_node = pipeline_pb.PipelineOrNode()
      pipeline_or_node.pipeline_node.CopyFrom(pipeline_begin_node_pb)
      pipeline_pb.nodes.append(pipeline_or_node)

    for node in tfx_pipeline.components:
      if isinstance(node, pipeline.Pipeline):
        pipeline_node_pb = self.compile(node, pipeline_ctx)
        pipeline_or_node = pipeline_pb.PipelineOrNode()
        pipeline_or_node.sub_pipeline.CopyFrom(pipeline_node_pb)

        # Set parent_ids of sub-pipelines, in the order of outer -> inner parent
        # pipelines.
        pipeline_or_node.sub_pipeline.pipeline_info.parent_ids.extend(
            parent_pipeline.pipeline_info.pipeline_name
            for parent_pipeline in pipeline_ctx.parent_pipelines
        )
        pipeline_or_node.sub_pipeline.pipeline_info.parent_ids.append(
            pipeline_ctx.pipeline_info.pipeline_name
        )

        pipeline_pb.nodes.append(pipeline_or_node)
      else:
        node_pb = self._compile_node(node, pipeline_ctx, deployment_config,
                                     tfx_pipeline.enable_cache)
        pipeline_or_node = pipeline_pb.PipelineOrNode()
        pipeline_or_node.pipeline_node.CopyFrom(node_pb)
        pipeline_pb.nodes.append(pipeline_or_node)

    # Inner pipelines of a composable pipeline, or a outmost pipeline with
    # pipeline-level outputs have pipeline end nodes.
    if not pipeline_ctx.is_root or tfx_pipeline._outputs:  # pylint: disable=protected-access
      pipeline_end_node_pb = self._compile_pipeline_end_node(
          tfx_pipeline, pipeline_ctx)
      pipeline_or_node = pipeline_pb.PipelineOrNode()
      pipeline_or_node.pipeline_node.CopyFrom(pipeline_end_node_pb)
      pipeline_pb.nodes.append(pipeline_or_node)

    if tfx_pipeline.platform_config:
      deployment_config.pipeline_level_platform_config.Pack(
          tfx_pipeline.platform_config)
    pipeline_pb.deployment_config.Pack(deployment_config)
    return pipeline_pb


def _fully_qualified_name(cls: Type[Any]):
  cls = deprecation_utils.get_first_nondeprecated_class(cls)
  return name_utils.get_full_name(cls)


def _validate_pipeline(tfx_pipeline: pipeline.Pipeline,
                       parent_pipelines: List[pipeline.Pipeline]):
  """Performs pre-compile validations."""
  if tfx_pipeline.execution_mode == pipeline.ExecutionMode.ASYNC:
    if compiler_utils.has_task_dependency(tfx_pipeline):
      raise ValueError("Task dependency is not supported in ASYNC mode.")

    if compiler_utils.has_resolver_node(tfx_pipeline):
      raise ValueError(
          "Resolver nodes can not be used in ASYNC mode. Use resolver "
          "functions instead."
      )

  if not compiler_utils.ensure_topological_order(tfx_pipeline.components):
    raise ValueError("Pipeline components are not topologically sorted.")

  if (
      parent_pipelines
      and tfx_pipeline.execution_mode != pipeline.ExecutionMode.SYNC
  ):
    raise ValueError("Subpipeline has to be Sync execution mode.")


def _set_node_outputs(node: pipeline_pb2.PipelineNode,
                      tfx_node_outputs: Dict[str, types.Channel]):
  """Compiles the outputs of a pipeline node."""
  for key, value in tfx_node_outputs.items():
    node.outputs.outputs[key].CopyFrom(
        compiler_utils.output_spec_from_channel(
            channel=value, node_id=node.node_info.id))


def _generate_input_spec_for_outputs(
    node: pipeline_pb2.PipelineNode,
    tfx_node_outputs: Dict[str, types.Channel],
    negative_context_filter: Callable[[pipeline_pb2.ContextSpec],
                                      bool] = lambda _: False
) -> Iterator[Tuple[types.Channel, pipeline_pb2.InputSpec.Channel]]:
  """Generates InputSpec in producer node, to be used by consumer node later."""
  for key, value in tfx_node_outputs.items():
    channel_pb = pipeline_pb2.InputSpec.Channel()
    channel_pb.producer_node_query.id = node.node_info.id
    for context in node.contexts.contexts:
      if negative_context_filter(context):
        continue
      context_query = channel_pb.context_queries.add()
      context_query.type.CopyFrom(context.type)
      context_query.name.CopyFrom(context.name)

    artifact_type = value.type._get_artifact_type()  # pylint: disable=protected-access
    channel_pb.artifact_query.type.CopyFrom(artifact_type)
    channel_pb.artifact_query.type.ClearField("properties")
    channel_pb.output_key = key
    yield value, channel_pb


def _generate_input_spec_for_pipeline_outputs(
    end_node: pipeline_pb2.PipelineNode, p: pipeline.Pipeline
) -> Iterator[Tuple[types.Channel, pipeline_pb2.InputSpec.Channel]]:
  """Generates InputSpec for pipeline outputs to be consumed later."""
  def remove_inner_pipeline_run_context(c: pipeline_pb2.ContextSpec) -> bool:
    return (c.type.name == constants.PIPELINE_RUN_CONTEXT_TYPE_NAME and
            c.name.HasField("structural_runtime_parameter"))

  yield from _generate_input_spec_for_outputs(
      end_node, p.outputs, remove_inner_pipeline_run_context)


def _set_node_parameters(node: pipeline_pb2.PipelineNode,
                         tfx_node: base_node.BaseNode):
  """Compiles exec properties of a pipeline node."""
  for key, value in tfx_node.exec_properties.items():
    if value is None:
      continue
    parameter_value = node.parameters.parameters[key]

    # Order matters, because runtime parameter can be in serialized string.
    if isinstance(value, data_types.RuntimeParameter):
      compiler_utils.set_runtime_parameter_pb(parameter_value.runtime_parameter,
                                              value.name, value.ptype,
                                              value.default)
    elif isinstance(value, placeholder.Placeholder):
      compiler_utils.validate_exec_property_placeholder(key, value)
      parameter_value.placeholder.CopyFrom(
          channel_utils.encode_placeholder_with_channels(
              value, compiler_utils.implicit_channel_key
          )
      )
    else:
      try:
        data_types_utils.set_parameter_value(parameter_value, value)
      except ValueError as e:
        raise ValueError(
            "Component {} got unsupported parameter {} with type {}.".format(
                tfx_node.id, key, type(value))) from e


def _set_node_execution_options(
    node: pipeline_pb2.PipelineNode,
    tfx_node: base_node.BaseNode,
    pipeline_ctx: compiler_context.PipelineContext,
    enable_cache: bool,
):
  """Compiles and sets NodeExecutionOptions of a pipeline node."""
  options_py = tfx_node.node_execution_options
  if options_py:
    assert isinstance(options_py, execution_options_utils.NodeExecutionOptions)
    if (
        options_py.trigger_strategy
        not in (
            pipeline_pb2.NodeExecutionOptions.TriggerStrategy.TRIGGER_STRATEGY_UNSPECIFIED,
            pipeline_pb2.NodeExecutionOptions.TriggerStrategy.ALL_UPSTREAM_NODES_SUCCEEDED,
        )
        or options_py.success_optional
        or options_py.lifetime_start
    ) and pipeline_ctx.is_async_mode:
      raise ValueError(
          "Node level triggering strategies, success optionality, and resource"
          " lifetimes are not allowed in ASYNC pipelines."
      )
    if (
        options_py.trigger_strategy
        == pipeline_pb2.NodeExecutionOptions.LIFETIME_END_WHEN_SUBGRAPH_CANNOT_PROGRESS
        and not options_py.lifetime_start
    ):
      raise ValueError(
          f"Node {node.node_info.id} has the trigger strategy"
          " LIFETIME_END_WHEN_SUBGRAPH_CANNOT_PROGRESS set but no"
          " lifetime_start. In order to use the trigger strategy the node"
          " must have a lifetime_start."
      )
    options_proto = node_execution_options_utils.compile_node_execution_options(
        options_py
    )
  else:
    options_proto = pipeline_pb2.NodeExecutionOptions()

  options_proto.caching_options.enable_cache = enable_cache
  node.execution_options.CopyFrom(options_proto)

  # TODO: b/310726801 - We should throw an error if this is an invalid
  # configuration.
  if pipeline_ctx.is_async_mode:
    input_triggers = node.execution_options.async_trigger.input_triggers
    for input_key, input_channel in tfx_node.inputs.items():
      if isinstance(input_channel.input_trigger, channel_types.NoTrigger):
        input_triggers[input_key].no_trigger = True
      if isinstance(
          input_channel.input_trigger, channel_types.TriggerByProperty
      ):
        input_triggers[input_key].trigger_by_property.property_keys.extend(
            input_channel.input_trigger.property_keys
        )


def _find_runtime_upstream_node_ids(
    pipeline_ctx: compiler_context.PipelineContext,
    here: base_node.BaseNode) -> List[str]:
  """Finds all upstream nodes that the current node depends on."""
  result = set()
  for up in itertools.chain(here.upstream_nodes,
                            pipeline_ctx.implicit_upstream_nodes(here)):
    if pipeline_ctx.is_async_mode and compiler_utils.is_resolver(up):
      result.update(_find_runtime_upstream_node_ids(pipeline_ctx, up))
    else:
      result.add(up.id)
  # Validate that upstream nodes are present in the pipeline.
  for up_id in result:
    if up_id not in pipeline_ctx.pipeline_node_ids:
      raise ValueError(f"Node {here.id} references upstream node {up_id} "
                       "which is not present in the pipeline.")
  # Sort result so that compiler generates consistent results.
  return sorted(result)


def _find_runtime_downstream_node_ids(context: compiler_context.PipelineContext,
                                      here: base_node.BaseNode) -> List[str]:
  """Finds all downstream nodes that depend on the current node."""
  result = set()
  if not context:
    return result
  for down in itertools.chain(here.downstream_nodes,
                              context.implicit_downstream_nodes(here)):
    if context.is_async_mode and compiler_utils.is_resolver(down):
      result.update(_find_runtime_downstream_node_ids(context, down))
    else:
      result.add(down.id)
  # Validate that downstream nodes are present in the pipeline.
  for down_id in result:
    if down_id not in context.pipeline_node_ids:
      raise ValueError(f"Node {here.id} references downstream node {down_id} "
                       "which is not present in the pipeline.")
  # Sort result so that compiler generates consistent results.
  return sorted(result)


def _begin_node_is_upstream(node: base_node.BaseNode,
                            tfx_pipeline: pipeline.Pipeline) -> bool:
  """Checks if a node needs to declare the begin node as its upstream node."""
  # Check if the PipelineInputChannel (whose dependent node ID is the pipeline
  # ID) is either directly or indirectly used for the node inputs.
  return tfx_pipeline.id in compiler_utils.get_data_dependent_node_ids(node)


def _end_node_is_downstream(node: base_node.BaseNode,
                            tfx_pipeline: pipeline.Pipeline) -> bool:
  """Checks if a node needs to declare the eng node as its downstream node."""
  # Given a node N inside a pipeline P, N needs to declare P_end as its
  # downstream node iff N produces at least a same output as P.
  pipeline_outputs_set = {c.wrapped for c in tfx_pipeline.outputs.values()}
  for node_output in node.outputs.values():
    if node_output in pipeline_outputs_set:
      return True
  return False
