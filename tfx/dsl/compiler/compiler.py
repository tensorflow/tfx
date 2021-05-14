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
from typing import cast, Iterable, List, Mapping

from tfx import types
from tfx.dsl.compiler import compiler_utils
from tfx.dsl.compiler import constants
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import base_driver
from tfx.dsl.components.base import base_node
from tfx.dsl.components.common import resolver
from tfx.orchestration import data_types
from tfx.orchestration import data_types_utils
from tfx.orchestration import pipeline
from tfx.proto.orchestration import executable_spec_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import deprecation_utils
from tfx.utils import json_utils
from ml_metadata.proto import metadata_store_pb2


class _CompilerContext(object):
  """Encapsulates resources needed to compile a pipeline."""

  def __init__(self, pipeline_info: data_types.PipelineInfo,
               execution_mode: pipeline_pb2.Pipeline.ExecutionMode,
               topological_order: Mapping[str, int]):
    self.pipeline_info = pipeline_info
    self.execution_mode = execution_mode
    self.node_pbs = {}
    self.node_outputs = set()
    self._topological_order = topological_order

  @classmethod
  def from_tfx_pipeline(cls, tfx_pipeline: pipeline.Pipeline):
    topological_order = {}
    for i, node in enumerate(tfx_pipeline.components, start=1):
      topological_order[node.id] = i
    return cls(
        pipeline_info=tfx_pipeline.pipeline_info,
        execution_mode=compiler_utils.resolve_execution_mode(tfx_pipeline),
        topological_order=topological_order)

  def topologically_sorted(self, tfx_nodes: Iterable[base_node.BaseNode]):
    return sorted(tfx_nodes, key=lambda node: self._topological_order[node.id])

  @property
  def is_sync_mode(self):
    return self.execution_mode == pipeline_pb2.Pipeline.SYNC

  @property
  def is_async_mode(self):
    return self.execution_mode == pipeline_pb2.Pipeline.ASYNC


class Compiler(object):
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
                  tfx_node.id, property_name, type(property_value)))

      for property_name, property_value in (
          value.additional_custom_properties.items()):
        value_field = output_spec.artifact_spec.additional_custom_properties[
            property_name].field_value
        try:
          data_types_utils.set_metadata_value(value_field, property_value)
        except ValueError:
          raise ValueError(
              "Component {} got unsupported parameter {} with type {}.".format(
                  tfx_node.id, property_name, type(property_value)))

  def _compile_node(
      self, tfx_node: base_node.BaseNode, compile_context: _CompilerContext,
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
    node_context_pb.name.field_value.string_value = "{}.{}".format(
        compile_context.pipeline_info.pipeline_context_name, node.node_info.id)

    # Pre Step 3: Alter graph topology if needed.
    if compile_context.is_async_mode:
      tfx_node_inputs = self._compile_resolver_config(
          compile_context, tfx_node, node)
    else:
      tfx_node_inputs = tfx_node.inputs

    # Step 3: Node inputs
    for key, value in tfx_node_inputs.items():
      input_spec = node.inputs.inputs[key]
      channel = input_spec.channels.add()

      # If the node input comes from another node's output, fill the context
      # queries with the producer node's contexts.
      if value in compile_context.node_outputs:
        channel.producer_node_query.id = value.producer_component_id

        # Here we rely on pipeline.components to be topologically sorted.
        assert value.producer_component_id in compile_context.node_pbs, (
            "producer component should have already been compiled.")
        producer_pb = compile_context.node_pbs[value.producer_component_id]
        for producer_context in producer_pb.contexts.contexts:
          context_query = channel.context_queries.add()
          context_query.type.CopyFrom(producer_context.type)
          context_query.name.CopyFrom(producer_context.name)

      # If the node input does not come from another node's output, fill the
      # context queries based on Channel info. We requires every channel to have
      # pipeline context and will fill it automatically.
      else:
        # Add pipeline context query.
        context_query = channel.context_queries.add()
        context_query.type.CopyFrom(pipeline_context_pb.type)
        context_query.name.CopyFrom(pipeline_context_pb.name)

        # Optionally add node context query.
        if value.producer_component_id:
          # Add node context query if `producer_component_id` is present.
          channel.producer_node_query.id = value.producer_component_id
          node_context_query = channel.context_queries.add()
          node_context_query.type.name = constants.NODE_CONTEXT_TYPE_NAME
          node_context_query.name.field_value.string_value = "{}.{}".format(
              compile_context.pipeline_info.pipeline_context_name,
              value.producer_component_id)

      artifact_type = value.type._get_artifact_type()  # pylint: disable=protected-access
      channel.artifact_query.type.CopyFrom(artifact_type)
      channel.artifact_query.type.ClearField("properties")

      if value.output_key:
        channel.output_key = value.output_key

      # TODO(b/158712886): Calculate min_count based on if inputs are optional.
      # min_count = 0 stands for optional input and 1 stands for required input.

    # TODO(b/170694459): Refactor special nodes as plugins.
    # Step 3.1: Special treatment for Resolver node.
    if compiler_utils.is_resolver(tfx_node):
      assert compile_context.is_sync_mode
      node.inputs.resolver_config.resolver_steps.extend(
          _convert_to_resolver_steps(tfx_node))

    # Step 4: Node outputs
    for key, value in tfx_node.outputs.items():
      # Register the output in the context.
      compile_context.node_outputs.add(value)
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
        else:
          try:
            data_types_utils.set_metadata_value(parameter_value.field_value,
                                                value)
          except ValueError:
            raise ValueError(
                "Component {} got unsupported parameter {} with type {}."
                .format(tfx_node.id, key, type(value)))

    # Step 6: Executor spec and optional driver spec for components
    if isinstance(tfx_node, base_component.BaseComponent):
      executor_spec = tfx_node.executor_spec.encode(
          component_spec=tfx_node.spec)
      deployment_config.executor_specs[tfx_node.id].Pack(executor_spec)

      # TODO(b/163433174): Remove specialized logic once generalization of
      # driver spec is done.
      if tfx_node.driver_class != base_driver.BaseDriver:
        driver_class_path = "{}.{}".format(tfx_node.driver_class.__module__,
                                           tfx_node.driver_class.__name__)
        driver_spec = executable_spec_pb2.PythonClassExecutableSpec()
        driver_spec.class_path = driver_class_path
        deployment_config.custom_driver_specs[tfx_node.id].Pack(driver_spec)

    # Step 7: Upstream/Downstream nodes
    # Note: the order of tfx_node.upstream_nodes is inconsistent from
    # run to run. We sort them so that compiler generates consistent results.
    # For ASYNC mode upstream/downstream node information is not set as
    # compiled IR graph topology can be different from that on pipeline
    # authoring time; for example Resolver nodes are removed.
    if compile_context.is_sync_mode:
      node.upstream_nodes.extend(
          sorted(node.id for node in tfx_node.upstream_nodes))
      node.downstream_nodes.extend(
          sorted(node.id for node in tfx_node.downstream_nodes))

    # Step 8: Node execution options
    node.execution_options.caching_options.enable_cache = enable_cache

    # Step 9: Per-node platform config
    if isinstance(tfx_node, base_component.BaseComponent):
      tfx_component = cast(base_component.BaseComponent, tfx_node)
      if tfx_component.platform_config:
        deployment_config.node_level_platform_configs[tfx_node.id].Pack(
            tfx_component.platform_config)

    return node

  def _compile_resolver_config(self, context: _CompilerContext,
                               tfx_node: base_node.BaseNode,
                               node: pipeline_pb2.PipelineNode):
    """Compiles upstream Resolver nodes as a ResolverConfig.

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
    input_channels = dict(tfx_node.inputs.get_all())  # Shallow copy.
    resolver_steps = []
    resolver_nodes = self._get_upstream_resolver_nodes(tfx_node)
    # Reduce each resolver node into resolver steps in reversed topological
    # order.
    for resolver_node in reversed(context.topologically_sorted(resolver_nodes)):
      resolver_channels = {
          input_key: channel
          for input_key, channel in input_channels.items()
          if channel.producer_component_id == resolver_node.id
      }
      for input_key, channel in resolver_channels.items():
        # CAVEAT: Currently resolver does not alter the input key, and we
        # require the output key of the resolver (which is the same as the
        # input key) to be consumed AS IS in the downstream node, whether it is
        # a resolver node or a TFX component node.
        # TODO(b/178452031): New Resolver should properly handle key mismatch.
        if input_key != channel.output_key:
          raise ValueError(f"Downstream node input key ({input_key}) should be "
                           f"the same as the output key ({channel.output_key}) "
                           "of the resolver node.")
        # Step 1.
        # Remove channel between parent resolver node and the tfx_node.
        del input_channels[input_key]
      # Step 2.
      # Rewire resolver node inputs to the tfx_node inputs.
      for parent_input_key, channel in resolver_node.inputs.items():
        if parent_input_key in input_channels:
          if channel != input_channels[parent_input_key]:
            raise ValueError(
                f"Duplicated input key {parent_input_key} found while "
                f"compiling {tfx_node.type}#{tfx_node.id}.")
        else:
          input_channels[parent_input_key] = channel
      # Step 3.
      # Convert resolver node into corresponding resolver steps.
      resolver_steps.extend(
          reversed(_convert_to_resolver_steps(resolver_node)))

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
    _validate_pipeline(tfx_pipeline)
    context = _CompilerContext.from_tfx_pipeline(tfx_pipeline)
    pipeline_pb = pipeline_pb2.Pipeline()
    pipeline_pb.pipeline_info.id = context.pipeline_info.pipeline_name
    pipeline_pb.execution_mode = context.execution_mode
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


def _convert_to_resolver_steps(resolver_node: base_node.BaseNode):
  """Converts Resolver node to a corresponding ResolverSteps."""
  assert compiler_utils.is_resolver(resolver_node)
  resolver_node = cast(resolver.Resolver, resolver_node)
  result = []
  for strategy_cls, config in resolver_node.strategy_class_and_configs:
    strategy_cls = deprecation_utils.get_first_nondeprecated_class(strategy_cls)
    step = pipeline_pb2.ResolverConfig.ResolverStep()
    step.class_path = (
        f"{strategy_cls.__module__}.{strategy_cls.__name__}")
    step.config_json = json_utils.dumps(config)
    step.input_keys.extend(resolver_node.inputs.keys())
    result.append(step)
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


def _validate_pipeline(tfx_pipeline: pipeline.Pipeline):
  """Performs pre-compile validations."""
  if (tfx_pipeline.execution_mode == pipeline.ExecutionMode.ASYNC and
      compiler_utils.has_task_dependency(tfx_pipeline)):
    raise ValueError("Task dependency is not supported in ASYNC mode.")

  if not compiler_utils.ensure_topological_order(tfx_pipeline.components):
    raise ValueError("Pipeline components are not topologically sorted.")
