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
import inspect
import itertools
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type, cast, Mapping

from tfx import types
from tfx.dsl.compiler import compiler_context
from tfx.dsl.compiler import compiler_utils
from tfx.dsl.compiler import constants
from tfx.dsl.compiler import node_inputs_compiler
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import base_driver
from tfx.dsl.components.base import base_node
from tfx.dsl.components.common import resolver
from tfx.dsl.control_flow import for_each
from tfx.dsl.experimental.conditionals import conditional
from tfx.dsl.input_resolution import resolver_function
from tfx.dsl.input_resolution import resolver_op
from tfx.dsl.input_resolution.ops import ops
from tfx.dsl.placeholder import placeholder
from tfx.orchestration import data_types
from tfx.orchestration import data_types_utils
from tfx.orchestration import pipeline
from tfx.proto.orchestration import executable_spec_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import channel as tfx_channel
from tfx.types import channel_utils
from tfx.types import resolved_channel
from tfx.types import value_artifact
from tfx.utils import deprecation_utils
from tfx.utils import json_utils
from tfx.utils import name_utils


class Compiler:
  """Compiles a TFX pipeline or a component into a uDSL IR proto."""

  def __init__(self, use_input_v2: bool = True):
    self._use_input_v2 = use_input_v2

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
    _set_node_context(node, pipeline_ctx)

    if self._use_input_v2:
      # Pipeline node inputs are stored as the inputs of the PipelineBegin node.
      node_inputs_compiler.compile_node_inputs(
          pipeline_ctx.parent, p, node.inputs)
    else:
      # Step 3: Node inputs
      # Composable pipeline's pipeline-level inputs are stored as the inputs of
      # PipelineBegin node.
      implicit_input_channels = {}

      # Step 3.1.1: Conditionals
      # Composable pipeline's conditional config is stored in PipelineBegin
      # node.
      implicit_input_channels.update(
          _set_conditionals(node, p, pipeline_ctx.parent, p.inputs))
      # Step 3.1.2: Add placeholder exec props to implicit_input_channels
      implicit_input_channels.update(
          _gather_implicit_inputs_from_exec_properties(p))

      # Step 3.2: Handle ForEach.
      # Similarly, pipeline level's foreach config is stored in PipelineBegin
      # node
      _set_for_each(node, p, pipeline_ctx.parent, p.inputs)

      # Step 3.3: Fill node inputs
      # Note here we use parent pipeline context, because a PipelineBegin node
      # uses output channels from its parent pipeline.
      _set_node_inputs(node, p, pipeline_ctx.parent, p.inputs,
                       implicit_input_channels)

    # Step 4: Node outputs
    # PipeineBegin node's outputs are the same as its inputs,
    # i.e., the composable pipeline's inputs.
    _set_node_outputs(node, p.inputs)
    internal_inputs = p._inputs.inputs if p._inputs else {}  # pylint: disable=protected-access
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
    _set_node_context(node, pipeline_ctx)

    if self._use_input_v2:
      node_inputs_compiler.compile_node_inputs(
          pipeline_ctx,
          compiler_utils.create_pipeline_end_node(p),
          node.inputs)
    else:
      # Step 3: Node inputs
      # Conditionals and Foreach do not apply to PipelineEnd node.
      # The inputs of a PipelineEnd node is the same as inner pipeline's
      # outputs.
      _set_node_inputs(node, p, pipeline_ctx,
                       {k: c.wrapped for k, c in p.outputs.items()}, {})

    # Step 4: Node outputs
    # PipeineEnd node's outputs are the same as inner pipeline's outputs.
    _set_node_outputs(node, p.outputs)
    if pipeline_ctx.is_subpipeline:
      pipeline_ctx.parent.channels.update(
          _generate_input_spec_for_outputs(node, p.outputs))

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
    _set_node_context(node, pipeline_ctx)

    # Pre Step 3: Alter graph topology if needed.
    if pipeline_ctx.is_async_mode:
      tfx_node_inputs = _embed_upstream_resolver_nodes(pipeline_ctx,
                                                       tfx_node, node)
    else:
      tfx_node_inputs = tfx_node.inputs

    # Step 3: Node inputs

    if self._use_input_v2:
      node_inputs_compiler.compile_node_inputs(
          pipeline_ctx, tfx_node, node.inputs)
    else:
      # Step 3.1: Generate implicit input channels
      implicit_input_channels = {}

      # Step 3.1.1: Conditionals
      implicit_input_channels.update(
          _set_conditionals(node, tfx_node, pipeline_ctx, tfx_node_inputs))

      # Step 3.1.2: Add placeholder exec props to implicit_input_channels
      implicit_input_channels.update(
          _gather_implicit_inputs_from_exec_properties(tfx_node))

      # Step 3.2: Handle ForEach.
      _set_for_each(node, tfx_node, pipeline_ctx, tfx_node_inputs)

      # Step 3.3: Fill node inputs
      _set_node_inputs(node, tfx_node, pipeline_ctx, tfx_node_inputs,
                       implicit_input_channels)

      # TODO(b/170694459): Refactor special nodes as plugins.
      # Step 3.4: Special treatment for Resolver node.
      if compiler_utils.is_resolver(tfx_node):
        assert pipeline_ctx.is_sync_mode

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

    # Step 8: Node execution options
    node.execution_options.caching_options.enable_cache = enable_cache
    # TODO(b/211890056): Support non default triggering strategies and node
    # success optionality with optional artifacts.
    node_execution_options = getattr(tfx_node, "_node_execution_options", None)
    if node_execution_options:
      if node_execution_options.get("trigger_strategy"):
        if not pipeline_ctx.is_sync_mode:
          raise ValueError("Node level triggering strategies are only used in "
                           "SYNC pipelines.")
        if tfx_node.inputs:
          raise NotImplementedError("Non default triggering strategies with "
                                    "data dependency are not yet supported.")
        node.execution_options.strategy = node_execution_options.get(
            "trigger_strategy")
      if node_execution_options.get("success_optional", None):
        if not pipeline_ctx.is_sync_mode:
          raise ValueError("Node level success optionality is only used in "
                           "SYNC pipelines.")
        if tfx_node.outputs:
          raise NotImplementedError("Node level success optionality with "
                                    "data dependency is not yet supported.")
        node.execution_options.node_success_optional = node_execution_options.get(
            "success_optional")

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
      # In ASYNC mode Resolver nodes are merged into the downstream node as a
      # ResolverConfig
      if compiler_utils.is_resolver(node) and pipeline_ctx.is_async_mode:
        continue

      if isinstance(node, pipeline.Pipeline):
        pipeline_node_pb = self.compile(node, pipeline_ctx)
        pipeline_or_node = pipeline_pb.PipelineOrNode()
        pipeline_or_node.sub_pipeline.CopyFrom(pipeline_node_pb)
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


def _compile_op_node(
    op_node: resolver_op.OpNode,) -> pipeline_pb2.ResolverConfig.ResolverStep:
  result = pipeline_pb2.ResolverConfig.ResolverStep()
  result.class_path = _fully_qualified_name(op_node.op_type)
  result.config_json = json_utils.dumps(op_node.kwargs)
  return result


def _compile_trace_result(
    trace_result: resolver_op.Node,
) -> List[pipeline_pb2.ResolverConfig.ResolverStep]:
  """Converts traced ResolverFunction output to corresponding ResolverSteps."""
  result = []
  op_node = trace_result
  while not isinstance(op_node, resolver_op.InputNode):
    assert isinstance(op_node, resolver_op.OpNode)
    result.append(_compile_op_node(op_node))
    if len(op_node.args) != 1:
      raise NotImplementedError(
          f"ResolverOp {op_node!r} has {len(op_node.args)} args, expected 1.")
    op_node = op_node.args[0]
  return list(reversed(result))


def _get_upstream_resolver_nodes(
    tfx_node: base_node.BaseNode) -> List[resolver.Resolver]:
  """Gets all transitive upstream resolver nodes in topological order."""
  result = []
  visit_queue = list(tfx_node.upstream_nodes)
  seen = set(node.id for node in visit_queue)
  while visit_queue:
    node = visit_queue.pop()
    if not compiler_utils.is_resolver(node):
      continue
    result.append(cast(resolver.Resolver, node))
    for upstream_node in node.upstream_nodes:
      if upstream_node.id not in seen:
        seen.add(node.id)
        visit_queue.append(upstream_node)
  return result


def _embed_upstream_resolver_nodes(
    pipeline_ctx: compiler_context.PipelineContext,
    tfx_node: base_node.BaseNode,
    node: pipeline_pb2.PipelineNode):
  """Embeds upstream Resolver nodes as a ResolverConfig.

  Iteratively reduces upstream resolver nodes into a resolver config of the
  current node until no upstream resolver node remains.
  Each iteration will consume one upstream resolver node, and convert it to
  the equivalent resolver steps and corresponding input channels.
  For example consider the following diagram:

  +--------------+  +------------+
  |  Upstream A  |  | Upstream B |
  +--------------+  +------------+
      a|    |b            |i  <-- output key
       |    |             |
      c|    |d            |
       v    v             |
  +----+----+----+        |
  | Resolver     |        |
  | cls=Foo      |   +----+
  +--------------+   |
      c|    |d <---- | ----- output key of the Resolver should be the same
       |    |        |       as the input key of the Current Node.
      c|    |d       |j  <-- input key
       v    v        v
      ++----+--------+-+
      | Current Node   |
      | ResolverSteps: |
      |   - ...        |
      +----------------+

  After one iteration, the Resolver node would be replaced by the resolver
  step of the downstream (current node).

  +--------------+  +------------+
  |  Upstream A  |  | Upstream B |
  +--------------+  +------------+
      a|    |b            |i
       |    |             |
      c|    |d            |j
       v    v             v
  +----+----+-------------+------+
  | Current Node                 |
  | ResolverSteps:               |
  |   - Foo()                    |
  |   - ...                      |
  +------------------------------+

  Following things are done for each reduction iteration:
   * Pick a upstream resolver node (in a reversed topological order).
   * Remove channels between resolver node and the current node.
   * Rewire resolver node input channels as those of the current node.
   * Convert the resolver node into corresponding resolver steps.

  This only applies to the ASYNC mode pipeline compilation.

  Args:
    pipeline_ctx: A pipeline context.
    tfx_node: A BaseNode instance.
    node: A PipelineNode IR to compile ResolverConfig into.

  Returns:
    a modified input channels of the given node.
  """
  # This input_channels dict will be updated in the middle as the resolver
  # nodes are reduced, and this updated input_channels should be used
  # afterwise instead of tfx_node.inputs.
  input_channels = dict(tfx_node.inputs)  # Shallow copy.
  resolver_steps = []
  resolver_nodes = _get_upstream_resolver_nodes(tfx_node)
  # Reduce each resolver node into resolver steps in reversed topological
  # order.
  for resolver_node in reversed(
      pipeline_ctx.topologically_sorted(resolver_nodes)):
    resolver_node = cast(resolver.Resolver, resolver_node)
    # TODO(b/169573945, lidanm): Properly handle channel.union() for resolver
    # node in async mode.
    resolver_channels = {
        input_key: chnl
        for input_key, chnl in input_channels.items()
        if chnl.producer_component_id == resolver_node.id
    }
    for input_key, chnl in resolver_channels.items():
      # CAVEAT: Currently resolver does not alter the input key, and we
      # require the output key of the resolver (which is the same as the
      # input key) to be consumed AS IS in the downstream node, whether it is
      # a resolver node or a TFX component node.
      # TODO(b/178452031): New Resolver should properly handle key mismatch.
      if input_key != chnl.output_key:
        raise ValueError(f"Downstream node input key ({input_key}) should be "
                         f"the same as the output key ({chnl.output_key}) "
                         "of the resolver node.")
      # Step 1.
      # Remove channel between parent resolver node and the tfx_node.
      del input_channels[input_key]
    # Step 2.
    # Rewire resolver node inputs to the tfx_node inputs.
    for parent_input_key, chnl in resolver_node.raw_inputs.items():
      if parent_input_key in input_channels:
        if chnl != input_channels[parent_input_key]:
          raise ValueError(
              f"Duplicated input key {parent_input_key} found while "
              f"compiling {tfx_node.type}#{tfx_node.id}.")
      else:
        input_channels[parent_input_key] = chnl
    # Step 3.
    # Convert resolver node into corresponding resolver steps.
    resolver_steps.extend(reversed(_compile_resolver_node(resolver_node)))

  if resolver_steps:
    node.inputs.resolver_config.resolver_steps.extend(reversed(resolver_steps))
  return input_channels


def _compile_resolver_node(
    resolver_node: resolver.Resolver,
) -> List[pipeline_pb2.ResolverConfig.ResolverStep]:
  """Converts Resolver node to a corresponding ResolverSteps."""
  result = _compile_trace_result(resolver_node.trace_result)
  for step in result:
    step.input_keys.extend(resolver_node.raw_inputs.keys())
  return result


def _compile_for_each_context(
    input_key: str) -> List[pipeline_pb2.ResolverConfig.ResolverStep]:
  """Generates ResolverSteps corresponding to ForEach context."""

  @resolver_function.resolver_function
  def impl(input_node):
    items = ops.Unnest(input_node, key=input_key)
    return ops.SkipIfEmpty(items)

  dummy_input_node = resolver_op.InputNode(
      None, resolver_op.DataType.ARTIFACT_MULTIMAP)
  return _compile_trace_result(impl.trace(dummy_input_node))


def _validate_pipeline(tfx_pipeline: pipeline.Pipeline,
                       parent_pipelines: List[pipeline.Pipeline]):
  """Performs pre-compile validations."""
  if (tfx_pipeline.execution_mode == pipeline.ExecutionMode.ASYNC and
      compiler_utils.has_task_dependency(tfx_pipeline)):
    raise ValueError("Task dependency is not supported in ASYNC mode.")

  if not compiler_utils.ensure_topological_order(tfx_pipeline.components):
    raise ValueError("Pipeline components are not topologically sorted.")

  if parent_pipelines and tfx_pipeline.execution_mode != pipeline.ExecutionMode.SYNC:
    raise ValueError("Subpipeline has to be Sync execution mode.")


def _set_node_context(node: pipeline_pb2.PipelineNode,
                      pipeline_ctx: compiler_context.PipelineContext):
  """Compiles the node contexts of a pipeline node."""
  # Context for the pipeline, across pipeline runs.
  pipeline_context_pb = node.contexts.contexts.add()
  pipeline_context_pb.type.name = constants.PIPELINE_CONTEXT_TYPE_NAME
  pipeline_context_pb.name.field_value.string_value = (
      pipeline_ctx.pipeline_info.pipeline_context_name)

  # Context for the current pipeline run.
  if pipeline_ctx.is_sync_mode:
    pipeline_run_context_pb = node.contexts.contexts.add()
    pipeline_run_context_pb.type.name = constants.PIPELINE_RUN_CONTEXT_TYPE_NAME
    # TODO(kennethyang): Miragte pipeline run id to structural_runtime_parameter
    # To keep existing IR textprotos used in tests unchanged, we only use
    # structural_runtime_parameter for subpipelines. After the subpipeline being
    # implemented, we will need to migrate normal pipelines to
    # structural_runtime_parameter as well for consistency. Similar for below.
    if pipeline_ctx.is_subpipeline:
      compiler_utils.set_structural_runtime_parameter_pb(
          pipeline_run_context_pb.name.structural_runtime_parameter, [
              f"{pipeline_ctx.pipeline_info.pipeline_context_name}_",
              (constants.PIPELINE_RUN_ID_PARAMETER_NAME, str)
          ])
    else:
      compiler_utils.set_runtime_parameter_pb(
          pipeline_run_context_pb.name.runtime_parameter,
          constants.PIPELINE_RUN_ID_PARAMETER_NAME, str)

  # Contexts inherited from the parent pipelines.
  for i, parent_pipeline in enumerate(pipeline_ctx.parent_pipelines[::-1]):
    parent_pipeline_context_pb = node.contexts.contexts.add()
    parent_pipeline_context_pb.type.name = constants.PIPELINE_CONTEXT_TYPE_NAME
    parent_pipeline_context_pb.name.field_value.string_value = (
        parent_pipeline.pipeline_info.pipeline_context_name)

    if pipeline_ctx.is_sync_mode:
      pipeline_run_context_pb = node.contexts.contexts.add()
      pipeline_run_context_pb.type.name = (
          constants.PIPELINE_RUN_CONTEXT_TYPE_NAME)

      # TODO(kennethyang): Miragte pipeline run id to structural runtime
      # parameter for the similar reason mentioned above.
      # Use structural runtime parameter to represent pipeline_run_id except
      # for the root level pipeline, for backward compatibility.
      if i == len(pipeline_ctx.parent_pipelines) - 1:
        compiler_utils.set_runtime_parameter_pb(
            pipeline_run_context_pb.name.runtime_parameter,
            constants.PIPELINE_RUN_ID_PARAMETER_NAME, str)
      else:
        compiler_utils.set_structural_runtime_parameter_pb(
            pipeline_run_context_pb.name.structural_runtime_parameter, [
                f"{parent_pipeline.pipeline_info.pipeline_context_name}_",
                (constants.PIPELINE_RUN_ID_PARAMETER_NAME, str)
            ])

  # Context for the node, across pipeline runs.
  node_context_pb = node.contexts.contexts.add()
  node_context_pb.type.name = constants.NODE_CONTEXT_TYPE_NAME
  node_context_pb.name.field_value.string_value = (
      compiler_utils.node_context_name(
          pipeline_ctx.pipeline_info.pipeline_context_name,
          node.node_info.id))


def _set_conditionals(
    node: pipeline_pb2.PipelineNode, tfx_node: base_node.BaseNode,
    pipeline_ctx: compiler_context.PipelineContext,
    tfx_node_inputs: Dict[str, types.Channel],
) -> Iterator[Tuple[str, types.Channel]]:
  """Compiles the conditionals for a pipeline node."""
  predicates = conditional.get_predicates(
      tfx_node, pipeline_ctx.dsl_context_registry)

  if predicates:
    implicit_keys_map = {}
    for key, chnl in tfx_node_inputs.items():
      if not isinstance(chnl, types.BaseChannel):
        raise ValueError(
            "Conditional only support using channel as a predicate.")
      implicit_keys_map[compiler_utils.implicit_channel_key(chnl)] = key
    encoded_predicates = []
    for predicate in predicates:
      for chnl in predicate.dependent_channels():
        implicit_key = compiler_utils.implicit_channel_key(chnl)
        if implicit_key not in implicit_keys_map:
          yield implicit_key, chnl
      encoded_predicates.append(
          predicate.encode_with_keys(
              compiler_utils.build_channel_to_key_fn(implicit_keys_map)))
    # In async pipeline, conditional resolver step should be the last step
    # in all resolver steps of a node.
    resolver_step = node.inputs.resolver_config.resolver_steps.add()
    resolver_step.class_path = constants.CONDITIONAL_RESOLVER_CLASS_PATH
    resolver_step.config_json = json_utils.dumps(
        {"predicates": encoded_predicates})


def _gather_implicit_inputs_from_exec_properties(
    tfx_node: base_node.BaseNode) -> Iterator[Tuple[str, types.Channel]]:
  for value in tfx_node.exec_properties.values():
    if isinstance(value, placeholder.ChannelWrappedPlaceholder):
      if not (inspect.isclass(value.channel.type) and
              issubclass(value.channel.type, value_artifact.ValueArtifact)):
        raise ValueError(
            "Dynamic execution property only supports ValueArtifact typed "
            f"channel. Got {value.channel.type.TYPE_NAME}.")
      yield compiler_utils.implicit_channel_key(value.channel), value.channel


def _set_for_each(node: pipeline_pb2.PipelineNode, tfx_node: base_node.BaseNode,
                  pipeline_ctx: compiler_context.PipelineContext,
                  tfx_node_inputs: Dict[str, Any]):
  """Compiles the ForEach configs for a pipeline node."""
  dsl_contexts = pipeline_ctx.dsl_context_registry.get_contexts(tfx_node)
  for dsl_context in dsl_contexts:
    if isinstance(dsl_context, for_each.ForEachContext):
      for input_key, channel in tfx_node_inputs.items():
        if (isinstance(channel, types.LoopVarChannel) and
            channel.wrapped is dsl_context.wrapped_channel):
          node.inputs.resolver_config.resolver_steps.extend(
              _compile_for_each_context(input_key))
          break
      else:
        # Ideally should not reach here as the same check is performed at
        # ForEachContext.will_add_node().
        raise ValueError(
            f"Unable to locate ForEach loop variable {dsl_context.channel} "
            f"from inputs of node {tfx_node.id}.")

  # Check loop variable is used outside the ForEach.
  for input_key, channel in tfx_node_inputs.items():
    if isinstance(channel, types.LoopVarChannel):
      if channel.for_each_context not in dsl_contexts:
        raise ValueError("Loop variable cannot be used outside the ForEach "
                         f"(node_id = {tfx_node.id}, input_key = {input_key}).")


def _set_node_inputs(node: pipeline_pb2.PipelineNode,
                     tfx_node: base_node.BaseNode,
                     pipeline_ctx: compiler_context.PipelineContext,
                     tfx_node_inputs: Dict[str, types.Channel],
                     implicit_input_channels: Dict[str, types.Channel]):
  """Compiles the inputs for a pipeline node."""
  all_inputs = {**tfx_node_inputs, **implicit_input_channels}
  resolved_channel_inputs = {}

  for key, input_channel in all_inputs.items():
    if isinstance(input_channel, resolved_channel.ResolvedChannel):
      resolved_channel_inputs[key] = input_channel

  if (len(resolved_channel_inputs) != len(all_inputs)
      and resolved_channel_inputs):
    bad_input_keys = list(set(all_inputs) - set(resolved_channel_inputs))
    # TODO(b/236140795): Migrate to InputGraph based IR.
    # This is the weird limitation we have using the ResolverConfig based IR.
    # New InputGraph based IR can solve the problem.
    raise ValueError(
        f"Node {tfx_node.id} inputs should all be coming from the same input "
        f"function output, while {bad_input_keys} are not.")

  if resolved_channel_inputs:
    _set_resolved_channel_inputs(
        node, tfx_node, pipeline_ctx, resolved_channel_inputs)
    return

  for key, value in all_inputs.items():
    input_spec = node.inputs.inputs[key]

    if isinstance(value, tfx_channel.PipelineInputChannel):
      channel_pb = input_spec.channels.add()

      if value in pipeline_ctx.channels:
        channel_pb.CopyFrom(pipeline_ctx.channels[value])
      else:
        raise ValueError(
            f"Failed to find producer info for the input channel '{key}' "
            f"of node {tfx_node.id}.")
      continue

    for input_channel in channel_utils.get_individual_channels(value):
      channel_pb = input_spec.channels.add()

      if isinstance(input_channel, types.OutputChannel):
        if pipeline_ctx and input_channel in pipeline_ctx.channels:
          channel_pb.CopyFrom(pipeline_ctx.channels[input_channel])
        elif isinstance(input_channel, tfx_channel.PipelineOutputChannel):
          # Add pipeline context query
          context_query = channel_pb.context_queries.add()
          context_query.type.name = constants.PIPELINE_CONTEXT_TYPE_NAME
          context_query.name.field_value.string_value = (
              input_channel.pipeline.pipeline_name)

          # Add node context query
          node_context_query = channel_pb.context_queries.add()
          node_context_query.type.name = constants.NODE_CONTEXT_TYPE_NAME
          node_context_query.name.field_value.string_value = (
              compiler_utils.node_context_name(
                  input_channel.pipeline.pipeline_name,
                  input_channel.wrapped.producer_component_id))

          artifact_type = value.type._get_artifact_type()  # pylint: disable=protected-access
          channel_pb.artifact_query.type.CopyFrom(artifact_type)
          channel_pb.artifact_query.type.ClearField("properties")
        else:
          raise ValueError(
              f"Failed to find producer info for the input channel '{key}' "
              f"of node {tfx_node.id}.")
      else:
        # If the node input is not an OutputChannel, fill the context queries
        # based on Channel info. We requires every channel to have pipeline
        # context and will fill it automatically.
        for context in node.contexts.contexts:
          if context.type.name == constants.PIPELINE_CONTEXT_TYPE_NAME:
            context_query = channel_pb.context_queries.add()
            context_query.type.CopyFrom(context.type)
            context_query.name.CopyFrom(context.name)

        if input_channel.producer_component_id:
          # Add node context query if `producer_component_id` is present.
          channel_pb.producer_node_query.id = input_channel.producer_component_id
          node_context_query = channel_pb.context_queries.add()
          node_context_query.type.name = constants.NODE_CONTEXT_TYPE_NAME
          node_context_query.name.field_value.string_value = "{}.{}".format(
              pipeline_ctx.pipeline_info.pipeline_context_name,
              input_channel.producer_component_id)

        artifact_type = input_channel.type._get_artifact_type()  # pylint: disable=protected-access
        channel_pb.artifact_query.type.CopyFrom(artifact_type)
        channel_pb.artifact_query.type.ClearField("properties")

        if input_channel.output_key:
          channel_pb.output_key = input_channel.output_key

    # Set NodeInputs.min_count.
    if isinstance(tfx_node, base_component.BaseComponent):
      if key in implicit_input_channels:
        # Mark all input channel as optional for implicit inputs
        # (e.g. conditionals). This is suboptimal, but still a safe guess to
        # avoid breaking the pipeline run.
        input_spec.min_count = 0
      else:
        try:
          # Calculating min_count from ComponentSpec.INPUTS.
          # Channel optional in spec, but provided to component.
          if tfx_node.spec.is_allow_empty_input(key):
            # Channel is defined as allow_empty in spec. It's okay to
            # trigger even if it has no resolved input artifacts.
            input_spec.min_count = 0
          else:
            # Channel is defined as NOT allow_empty in spec.
            # It must have resolved input artifacts before it triggers.
            input_spec.min_count = 1
        except KeyError:
          # Currently we can fall here if the upstream resolver node inputs
          # are embedded into the current node (in async mode). We always
          # regard resolver's inputs as optional.
          if pipeline_ctx.is_async_mode:
            input_spec.min_count = 0
          else:
            raise
    else:
      input_spec.min_count = 0


# TODO(b/236140795): Migrate to InputGraph based IR.
def _set_resolved_channel_inputs(
    node: pipeline_pb2.PipelineNode,
    tfx_node: base_node.BaseNode,
    pipeline_ctx: compiler_context.PipelineContext,
    inputs: Mapping[str, resolved_channel.ResolvedChannel]) -> None:
  """Set `node.inputs` where all `inputs` are ResolvedChannel.

  ResolvedChannel is a special BaseChannel that marks the output of the
  ResolverFunction. If a node has ResolvedChannel input, all resolver function
  definition should also be compiled as part of the node inputs.

  Args:
    node: A PipelineNode IR output.
    tfx_node: A compiling node.
    pipeline_ctx: A PipelineContext for the current pipeline.
    inputs: An input ResolvedChannels with keys.
  """
  for input_key, channel in inputs.items():
    if channel.output_key != input_key:
      # Resolved dict key should match the consumer input key. For example
      #    inputs = resolve_inputs(...)
      #    consumer_a = A(x=inputs['x'])  # valid
      #    consumer_b = B(y=inputs['x'])  # INVALID
      # This is the weird limitation we have using the ResolverConfig based IR.
      # New InputGraph based IR can solve the problem.
      raise ValueError(
          f"Node {tfx_node.id}.inputs[{input_key!r}] should have the same key "
          f"as the output key value but got {channel.output_key!r}.")
  output_nodes = {channel.output_node for channel in inputs.values()}
  if len(output_nodes) != 1:
    # This is the weird limitation we have using the ResolverConfig based IR.
    # New InputGraph based IR can solve the problem.
    raise ValueError(
        f"Node {tfx_node.id} should consume from single input function but "
        f"detected {len(output_nodes)}.")
  output_node = output_nodes.pop()
  input_nodes = set(resolver_function.get_input_nodes(output_node))
  if len(input_nodes) != 1:
    # This is the weird limitation we have using the ResolverConfig based IR.
    # New InputGraph based IR can solve the problem.
    raise ValueError(
        "Resolver function with multiple input nodes are not yet supported.")
  input_node: resolver_op.InputNode = input_nodes.pop()
  if any(isinstance(channel, resolved_channel.ResolvedChannel)
         for channel in input_node.wrapped.values()):
    # This is the weird limitation we have using the ResolverConfig based IR.
    # New InputGraph based IR can solve the problem.
    raise ValueError(
        "Cannot use input function output as an input to another input "
        "function.")
  _set_node_inputs(node, tfx_node, pipeline_ctx, input_node.wrapped, {})
  node.inputs.resolver_config.resolver_steps.extend(
      _compile_trace_result(output_node))


def _set_node_outputs(node: pipeline_pb2.PipelineNode,
                      tfx_node_outputs: Dict[str, types.Channel]):
  """Compiles the outputs of a pipeline node."""
  for key, value in tfx_node_outputs.items():
    node.outputs.outputs[key].CopyFrom(
        compiler_utils.output_spec_from_channel(
            channel=value, node_id=node.node_info.id))


def _generate_input_spec_for_outputs(
    node: pipeline_pb2.PipelineNode, tfx_node_outputs: Dict[str, types.Channel]
) -> Iterator[Tuple[types.Channel, pipeline_pb2.InputSpec.Channel]]:
  """Generates InputSpec in producer node, to be used by consumer node later."""
  for key, value in tfx_node_outputs.items():
    channel_pb = pipeline_pb2.InputSpec.Channel()
    channel_pb.producer_node_query.id = node.node_info.id
    for context in node.contexts.contexts:
      context_query = channel_pb.context_queries.add()
      context_query.type.CopyFrom(context.type)
      context_query.name.CopyFrom(context.name)

    artifact_type = value.type._get_artifact_type()  # pylint: disable=protected-access
    channel_pb.artifact_query.type.CopyFrom(artifact_type)
    channel_pb.artifact_query.type.ClearField("properties")
    channel_pb.output_key = key
    yield value, channel_pb


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
    # RuntimeInfoPlaceholder passes Execution parameters of Facade
    # components.
    elif isinstance(value, placeholder.RuntimeInfoPlaceholder):
      parameter_value.placeholder.CopyFrom(value.encode())
    # ChannelWrappedPlaceholder passes dynamic execution parameter.
    elif isinstance(value, placeholder.ChannelWrappedPlaceholder):
      compiler_utils.validate_dynamic_exec_ph_operator(value)
      parameter_value.placeholder.CopyFrom(
          value.encode_with_keys(compiler_utils.implicit_channel_key))
    else:
      try:
        data_types_utils.set_parameter_value(parameter_value, value)
      except ValueError:
        raise ValueError(
            "Component {} got unsupported parameter {} with type {}.".format(
                tfx_node.id, key, type(value))) from ValueError


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
  for input_chan in node.inputs.values():
    if tfx_pipeline.id in channel_utils.get_dependent_node_ids(input_chan):
      return True
  return False


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
