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
import collections
import inspect
import itertools
from typing import Any, Iterable, List, Type, cast

from tfx import types
from tfx.dsl.compiler import compiler_utils
from tfx.dsl.compiler import constants
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
from tfx.types import channel_utils
from tfx.types import value_artifact
from tfx.utils import deprecation_utils
from tfx.utils import json_utils
from tfx.utils import name_utils

from ml_metadata.proto import metadata_store_pb2


class _CompilerContext:
  """Encapsulates resources needed to compile a pipeline."""

  def __init__(self, tfx_pipeline: pipeline.Pipeline):
    self.pipeline_info = tfx_pipeline.pipeline_info
    self.execution_mode = compiler_utils.resolve_execution_mode(tfx_pipeline)
    self.node_pbs = {}
    self.dsl_context_registry = tfx_pipeline.dsl_context_registry
    self._pipeline_nodes_by_id = {}
    self._topological_order = {}
    self._implicit_upstream_nodes = collections.defaultdict(set)
    self._implicit_downstream_nodes = collections.defaultdict(set)
    for i, node in enumerate(tfx_pipeline.components, start=1):
      self._pipeline_nodes_by_id[node.id] = node
      self._topological_order[node.id] = i
      self._collect_conditional_dependency(node)

  def _add_implicit_dependency(self, parent_id: str, child_id: str) -> None:
    self._implicit_upstream_nodes[child_id].add(parent_id)
    self._implicit_downstream_nodes[parent_id].add(child_id)

  def _collect_conditional_dependency(self, here: base_node.BaseNode) -> None:
    for predicate in conditional.get_predicates(
        here, self.dsl_context_registry):
      for chnl in predicate.dependent_channels():
        self._add_implicit_dependency(chnl.producer_component_id, here.id)

  def topologically_sorted(self, tfx_nodes: Iterable[base_node.BaseNode]):
    return sorted(tfx_nodes, key=lambda node: self._topological_order[node.id])

  @property
  def is_sync_mode(self):
    return self.execution_mode == pipeline_pb2.Pipeline.SYNC

  @property
  def is_async_mode(self):
    return self.execution_mode == pipeline_pb2.Pipeline.ASYNC

  def implicit_upstream_nodes(
      self, here: base_node.BaseNode) -> List[base_node.BaseNode]:
    return [
        self._pipeline_nodes_by_id[node_id]
        for node_id in self._implicit_upstream_nodes[here.id]
    ]

  def implicit_downstream_nodes(
      self, here: base_node.BaseNode) -> List[base_node.BaseNode]:
    return [
        self._pipeline_nodes_by_id[node_id]
        for node_id in self._implicit_downstream_nodes[here.id]
    ]


class Compiler:
  """Compiles a TFX pipeline or a component into a uDSL IR proto."""

  def _compile_node_outputs(self, tfx_node: base_node.BaseNode,
                            node_pb: pipeline_pb2.PipelineNode):
    """Compiles the outputs of a node/component."""
    for key, value in tfx_node.outputs.items():
      output_spec = node_pb.outputs.outputs[key]
      artifact_type = value.type._get_artifact_type()  # pylint: disable=protected-access
      output_spec.artifact_spec.type.CopyFrom(artifact_type)

      # Attach additional properties for artifacts produced by importer nodes.
      for property_name, property_value in value.additional_properties.items():
        _check_property_value_type(property_name, property_value, artifact_type)
        value_field = output_spec.artifact_spec.additional_properties[
            property_name].field_value
        try:
          data_types_utils.set_metadata_value(value_field, property_value)
        except ValueError:
          raise ValueError(
              "Component {} got unsupported parameter {} with type {}.".format(
                  tfx_node.id, property_name,
                  type(property_value))) from ValueError

      for property_name, property_value in (
          value.additional_custom_properties.items()):
        value_field = output_spec.artifact_spec.additional_custom_properties[
            property_name].field_value
        try:
          data_types_utils.set_metadata_value(value_field, property_value)
        except ValueError:
          raise ValueError(
              "Component {} got unsupported parameter {} with type {}.".format(
                  tfx_node.id, property_name,
                  type(property_value))) from ValueError

  def _compile_node(
      self,
      tfx_node: base_node.BaseNode,
      compile_context: _CompilerContext,
      deployment_config: pipeline_pb2.IntermediateDeploymentConfig,
      enable_cache: bool,
  ) -> pipeline_pb2.PipelineNode:
    """Compiles an individual TFX node into a PipelineNode proto.

    Args:
      tfx_node: A TFX node.
      compile_context: Resources needed to compile the node.
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
    # Context for the pipeline, across pipeline runs.
    pipeline_context_pb = node.contexts.contexts.add()
    pipeline_context_pb.type.name = constants.PIPELINE_CONTEXT_TYPE_NAME
    pipeline_context_pb.name.field_value.string_value = compile_context.pipeline_info.pipeline_context_name

    # Context for the current pipeline run.
    if compile_context.is_sync_mode:
      pipeline_run_context_pb = node.contexts.contexts.add()
      pipeline_run_context_pb.type.name = constants.PIPELINE_RUN_CONTEXT_TYPE_NAME
      compiler_utils.set_runtime_parameter_pb(
          pipeline_run_context_pb.name.runtime_parameter,
          constants.PIPELINE_RUN_ID_PARAMETER_NAME, str)

    # Context for the node, across pipeline runs.
    node_context_pb = node.contexts.contexts.add()
    node_context_pb.type.name = constants.NODE_CONTEXT_TYPE_NAME
    node_context_pb.name.field_value.string_value = (
        compiler_utils.node_context_name(
            compile_context.pipeline_info.pipeline_context_name,
            node.node_info.id))

    # Pre Step 3: Alter graph topology if needed.
    if compile_context.is_async_mode:
      tfx_node_inputs = self._embed_upstream_resolver_nodes(
          compile_context, tfx_node, node)
    else:
      tfx_node_inputs = tfx_node.inputs

    # Step 3: Node inputs

    # Step 3.1: Generate implicit input channels
    # Step 3.1.1: Conditionals
    implicit_input_channels = {}
    predicates = conditional.get_predicates(
        tfx_node, compile_context.dsl_context_registry)
    if predicates:
      implicit_keys_map = {}
      for key, chnl in tfx_node_inputs.items():
        if not isinstance(chnl, types.Channel):
          raise ValueError(
              "Conditional only support using channel as a predicate.")
        implicit_keys_map[compiler_utils.implicit_channel_key(chnl)] = key
      encoded_predicates = []
      for predicate in predicates:
        for chnl in predicate.dependent_channels():
          implicit_key = compiler_utils.implicit_channel_key(chnl)
          if implicit_key not in implicit_keys_map:
            # Store this channel and add it to the node inputs later.
            implicit_input_channels[implicit_key] = chnl
        encoded_predicates.append(
            predicate.encode_with_keys(
                compiler_utils.build_channel_to_key_fn(implicit_keys_map)))
      # In async pipeline, conditional resolver step should be the last step
      # in all resolver steps of a node.
      resolver_step = node.inputs.resolver_config.resolver_steps.add()
      resolver_step.class_path = constants.CONDITIONAL_RESOLVER_CLASS_PATH
      resolver_step.config_json = json_utils.dumps(
          {"predicates": encoded_predicates})

    # Step 3.1.2: Add placeholder exec props to implicit_input_channels
    for key, value in tfx_node.exec_properties.items():
      if isinstance(value, placeholder.ChannelWrappedPlaceholder):
        if not (inspect.isclass(value.channel.type) and
                issubclass(value.channel.type, value_artifact.ValueArtifact)):
          raise ValueError("output channel to dynamic exec properties is not "
                           "ValueArtifact")
        implicit_key = compiler_utils.implicit_channel_key(value.channel)
        implicit_input_channels[implicit_key] = value.channel

    # Step 3.2: Handle ForEach.
    dsl_contexts = compile_context.dsl_context_registry.get_contexts(tfx_node)
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
          raise ValueError(
              "Loop variable cannot be used outside the ForEach "
              f"(node_id = {tfx_node.id}, input_key = {input_key}).")

    # Step 3.3: Fill node inputs
    for key, value in itertools.chain(tfx_node_inputs.items(),
                                      implicit_input_channels.items()):
      input_spec = node.inputs.inputs[key]
      for input_channel in channel_utils.get_individual_channels(value):
        chnl = input_spec.channels.add()

        # If the node is an OutputChannel, fill the context queries with the
        # producer node's contexts.
        if isinstance(input_channel, types.OutputChannel):
          chnl.producer_node_query.id = input_channel.producer_component_id

          # Here we rely on pipeline.components to be topologically sorted.
          assert input_channel.producer_component_id in compile_context.node_pbs, (
              f"producer component {input_channel.producer_component_id} "
              f"should have been compiled before {tfx_node.id} "
              f"(while compiling {input_channel} in inputs['{key}']).")
          producer_pb = compile_context.node_pbs[
              input_channel.producer_component_id]
          for producer_context in producer_pb.contexts.contexts:
            context_query = chnl.context_queries.add()
            context_query.type.CopyFrom(producer_context.type)
            context_query.name.CopyFrom(producer_context.name)
        elif isinstance(input_channel, types.SourceChannel):
          # Add pipeline context query.
          if input_channel.pipeline_name:
            node_context_query = chnl.context_queries.add()
            node_context_query.type.name = constants.PIPELINE_CONTEXT_TYPE_NAME
            node_context_query.name.field_value.string_value = (
                input_channel.pipeline_name)
          # Add node context query.
          if input_channel.producer_component_id:
            # Add node context query if `producer_component_id` is present.
            chnl.producer_node_query.id = input_channel.producer_component_id
            node_context_query = chnl.context_queries.add()
            node_context_query.type.name = constants.NODE_CONTEXT_TYPE_NAME
            node_context_query.name.field_value.string_value = "{}.{}".format(
                input_channel.pipeline_name,
                input_channel.producer_component_id)
          # Add pipeline run context query.
          if input_channel.pipeline_run_id:
            node_context_query = chnl.context_queries.add()
            node_context_query.type.name = (
                constants.PIPELINE_RUN_CONTEXT_TYPE_NAME)
            node_context_query.name.field_value.string_value = (
                input_channel.pipeline_run_id)
        # If the node input is Channel, fill the context queries
        # based on Channel info. We requires every channel to have pipeline
        # context and will fill it automatically.
        else:
          # Add pipeline context query.
          context_query = chnl.context_queries.add()
          context_query.type.CopyFrom(pipeline_context_pb.type)
          context_query.name.CopyFrom(pipeline_context_pb.name)

          # Optionally add node context query.
          if input_channel.producer_component_id:
            # Add node context query if `producer_component_id` is present.
            chnl.producer_node_query.id = input_channel.producer_component_id
            node_context_query = chnl.context_queries.add()
            node_context_query.type.name = constants.NODE_CONTEXT_TYPE_NAME
            node_context_query.name.field_value.string_value = "{}.{}".format(
                compile_context.pipeline_info.pipeline_context_name,
                input_channel.producer_component_id)

        artifact_type = input_channel.type._get_artifact_type()  # pylint: disable=protected-access
        chnl.artifact_query.type.CopyFrom(artifact_type)
        chnl.artifact_query.type.ClearField("properties")

        if input_channel.output_key:
          chnl.output_key = input_channel.output_key

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
              if tfx_node.spec.is_optional_input(key):
                input_spec.min_count = 0
              else:
                input_spec.min_count = 1
            except KeyError:
              # Currently we can fall here if the upstream resolver node inputs
              # are embedded into the current node (in async mode). We always
              # regard resolver's inputs as optional.
              if compile_context.is_async_mode:
                input_spec.min_count = 0
              else:
                raise

    # TODO(b/170694459): Refactor special nodes as plugins.
    # Step 3.4: Special treatment for Resolver node.
    if compiler_utils.is_resolver(tfx_node):
      assert compile_context.is_sync_mode
      node.inputs.resolver_config.resolver_steps.extend(
          _compile_resolver_node(tfx_node))

    # Step 4: Node outputs
    if (isinstance(tfx_node, base_component.BaseComponent) or
        compiler_utils.is_importer(tfx_node)):
      self._compile_node_outputs(tfx_node, node)

    # Step 5: Node parameters
    if not compiler_utils.is_resolver(tfx_node):
      for key, value in tfx_node.exec_properties.items():
        if value is None:
          continue
        parameter_value = node.parameters.parameters[key]

        # Order matters, because runtime parameter can be in serialized string.
        if isinstance(value, data_types.RuntimeParameter):
          compiler_utils.set_runtime_parameter_pb(
              parameter_value.runtime_parameter, value.name, value.ptype,
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
                "Component {} got unsupported parameter {} with type {}."
                .format(tfx_node.id, key, type(value))) from ValueError

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
    node.upstream_nodes.extend(
        self._find_runtime_upstream_node_ids(compile_context, tfx_node))
    node.downstream_nodes.extend(
        self._find_runtime_downstream_node_ids(compile_context, tfx_node))

    # Step 8: Node execution options
    node.execution_options.caching_options.enable_cache = enable_cache

    # Step 9: Per-node platform config
    if isinstance(tfx_node, base_component.BaseComponent):
      tfx_component = cast(base_component.BaseComponent, tfx_node)
      if tfx_component.platform_config:
        deployment_config.node_level_platform_configs[tfx_node.id].Pack(
            tfx_component.platform_config)

    return node

  def _find_runtime_upstream_node_ids(self, context: _CompilerContext,
                                      here: base_node.BaseNode) -> List[str]:
    """Finds all upstream nodes that the current node depends on."""
    result = set()
    for up in itertools.chain(here.upstream_nodes,
                              context.implicit_upstream_nodes(here)):
      if context.is_async_mode and compiler_utils.is_resolver(up):
        result.update(self._find_runtime_upstream_node_ids(context, up))
      else:
        result.add(up.id)
    # Sort result so that compiler generates consistent results.
    return sorted(result)

  def _find_runtime_downstream_node_ids(self, context: _CompilerContext,
                                        here: base_node.BaseNode) -> List[str]:
    """Finds all downstream nodes that depend on the current node."""
    result = set()
    for down in itertools.chain(here.downstream_nodes,
                                context.implicit_downstream_nodes(here)):
      if context.is_async_mode and compiler_utils.is_resolver(down):
        result.update(self._find_runtime_downstream_node_ids(context, down))
      else:
        result.add(down.id)
    # Sort result so that compiler generates consistent results.
    return sorted(result)

  def _embed_upstream_resolver_nodes(self, context: _CompilerContext,
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
      context: A compiler context.
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
    resolver_nodes = self._get_upstream_resolver_nodes(tfx_node)
    # Reduce each resolver node into resolver steps in reversed topological
    # order.
    for resolver_node in reversed(context.topologically_sorted(resolver_nodes)):
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
      for parent_input_key, chnl in resolver_node.inputs.items():
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
      node.inputs.resolver_config.resolver_steps.extend(
          reversed(resolver_steps))
    return input_channels

  def _get_upstream_resolver_nodes(
      self, tfx_node: base_node.BaseNode) -> List[base_node.BaseNode]:
    """Gets all transitive upstream resolver nodes in topological order."""
    result = []
    visit_queue = list(tfx_node.upstream_nodes)
    seen = set(node.id for node in visit_queue)
    while visit_queue:
      node = visit_queue.pop()
      if not compiler_utils.is_resolver(node):
        continue
      result.append(node)
      for upstream_node in node.upstream_nodes:
        if upstream_node.id not in seen:
          seen.add(node.id)
          visit_queue.append(upstream_node)
    return result

  def compile(self, tfx_pipeline: pipeline.Pipeline) -> pipeline_pb2.Pipeline:
    """Compiles a tfx pipeline into uDSL proto.

    Args:
      tfx_pipeline: A TFX pipeline.

    Returns:
      A Pipeline proto that encodes all necessary information of the pipeline.
    """
    tfx_pipeline.finalize()
    _validate_pipeline(tfx_pipeline)
    context = _CompilerContext(tfx_pipeline)
    pipeline_pb = pipeline_pb2.Pipeline()
    pipeline_pb.pipeline_info.id = context.pipeline_info.pipeline_name
    pipeline_pb.execution_mode = context.execution_mode
    if isinstance(context.pipeline_info.pipeline_root, placeholder.Placeholder):
      pipeline_pb.runtime_spec.pipeline_root.placeholder.CopyFrom(
          context.pipeline_info.pipeline_root.encode())
    else:
      compiler_utils.set_runtime_parameter_pb(
          pipeline_pb.runtime_spec.pipeline_root.runtime_parameter,
          constants.PIPELINE_ROOT_PARAMETER_NAME, str,
          context.pipeline_info.pipeline_root)
    if pipeline_pb.execution_mode == pipeline_pb2.Pipeline.ExecutionMode.SYNC:
      compiler_utils.set_runtime_parameter_pb(
          pipeline_pb.runtime_spec.pipeline_run_id.runtime_parameter,
          constants.PIPELINE_RUN_ID_PARAMETER_NAME, str)

    deployment_config = pipeline_pb2.IntermediateDeploymentConfig()
    if tfx_pipeline.metadata_connection_config:
      deployment_config.metadata_connection_config.Pack(
          tfx_pipeline.metadata_connection_config)
    for node in tfx_pipeline.components:
      # In ASYNC mode Resolver nodes are merged into the downstream node as a
      # ResolverConfig
      if compiler_utils.is_resolver(node) and context.is_async_mode:
        continue
      node_pb = self._compile_node(node, context, deployment_config,
                                   tfx_pipeline.enable_cache)
      pipeline_or_node = pipeline_pb.PipelineOrNode()
      pipeline_or_node.pipeline_node.CopyFrom(node_pb)
      # TODO(b/158713812): Support sub-pipeline.
      pipeline_pb.nodes.append(pipeline_or_node)
      context.node_pbs[node.id] = node_pb

    if tfx_pipeline.platform_config:
      deployment_config.pipeline_level_platform_config.Pack(
          tfx_pipeline.platform_config)
    pipeline_pb.deployment_config.Pack(deployment_config)
    return pipeline_pb


def _fully_qualified_name(cls: Type[Any]):
  cls = deprecation_utils.get_first_nondeprecated_class(cls)
  return name_utils.get_full_name(cls)


def _compile_resolver_op(
    op_node: resolver_op.OpNode,) -> pipeline_pb2.ResolverConfig.ResolverStep:
  result = pipeline_pb2.ResolverConfig.ResolverStep()
  result.class_path = _fully_qualified_name(op_node.op_type)
  result.config_json = json_utils.dumps(op_node.kwargs)
  return result


def _compile_resolver_function(
    f: resolver_function.ResolverFunction,
) -> List[pipeline_pb2.ResolverConfig.ResolverStep]:
  """Converts ResolverFunction to corresponding ResolverSteps."""
  result = []
  op_node = f.trace()
  while not op_node.is_input_node:
    result.append(_compile_resolver_op(op_node))
    op_node = op_node.arg
  return list(reversed(result))


def _compile_resolver_node(
    resolver_node: base_node.BaseNode,
) -> List[pipeline_pb2.ResolverConfig.ResolverStep]:
  """Converts Resolver node to a corresponding ResolverSteps."""
  assert compiler_utils.is_resolver(resolver_node)
  resolver_node = cast(resolver.Resolver, resolver_node)
  result = _compile_resolver_function(resolver_node.resolver_function)
  for step in result:
    step.input_keys.extend(resolver_node.inputs.keys())
  return result


def _compile_for_each_context(
    input_key: str) -> List[pipeline_pb2.ResolverConfig.ResolverStep]:
  """Generates ResolverSteps corresponding to ForEach context."""

  @resolver_function.resolver_function
  def impl(input_node):
    items = ops.Unnest(input_node, key=input_key)
    return ops.SkipIfEmpty(items)

  return _compile_resolver_function(impl)


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


def _validate_pipeline(tfx_pipeline: pipeline.Pipeline):
  """Performs pre-compile validations."""
  if (tfx_pipeline.execution_mode == pipeline.ExecutionMode.ASYNC and
      compiler_utils.has_task_dependency(tfx_pipeline)):
    raise ValueError("Task dependency is not supported in ASYNC mode.")

  if not compiler_utils.ensure_topological_order(tfx_pipeline.components):
    raise ValueError("Pipeline components are not topologically sorted.")
