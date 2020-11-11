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
import json
import re

from typing import cast

from tfx.components.common_nodes import importer_node
from tfx.components.common_nodes import resolver_node
from tfx.dsl.compiler import compiler_utils
from tfx.dsl.compiler import constants
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import base_driver
from tfx.dsl.components.base import base_node
from tfx.dsl.experimental import latest_artifacts_resolver
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.orchestration import data_types
from tfx.orchestration import pipeline
from tfx.proto.orchestration import executable_spec_pb2
from tfx.proto.orchestration import pipeline_pb2


class _CompilerContext(object):
  """Encapsulates resources needed to compile a pipeline."""

  def __init__(self, pipeline_info: data_types.PipelineInfo,
               execution_mode: pipeline_pb2.Pipeline.ExecutionMode):
    self.pipeline_info = pipeline_info
    self.execution_mode = execution_mode
    self.node_pbs = {}


class Compiler(object):
  """Compiles a TFX pipeline or a component into a uDSL IR proto."""

  def __init__(self):
    pass

  def _compile_importer_node_outputs(self, tfx_node: base_node.BaseNode,
                                     node_pb: pipeline_pb2.PipelineNode):
    """Compiles the outputs of an importer node."""
    for key, value in tfx_node.outputs.items():
      output_spec = node_pb.outputs.outputs[key]
      artifact_type = value.type._get_artifact_type()  # pylint: disable=protected-access
      output_spec.artifact_spec.type.CopyFrom(artifact_type)

      # Attach additional properties for artifacts produced by importer nodes.
      for property_name, property_value in tfx_node.exec_properties[
          importer_node.PROPERTIES_KEY].items():
        value_field = output_spec.artifact_spec.additional_properties[
            property_name].field_value
        try:
          compiler_utils.set_field_value_pb(value_field, property_value)
        except ValueError:
          raise ValueError(
              "Component {} got unsupported parameter {} with type {}.".format(
                  tfx_node.id, property_name, type(property_value)))

      for property_name, property_value in tfx_node.exec_properties[
          importer_node.CUSTOM_PROPERTIES_KEY].items():
        value_field = output_spec.artifact_spec.additional_custom_properties[
            property_name].field_value
        try:
          compiler_utils.set_field_value_pb(value_field, property_value)
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
    if (compile_context.execution_mode ==
        pipeline_pb2.Pipeline.ExecutionMode.SYNC):
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

    # Step 3: Node inputs
    for key, value in tfx_node.inputs.items():
      input_spec = node.inputs.inputs[key]
      channel = input_spec.channels.add()
      if value.producer_component_id:
        channel.producer_node_query.id = value.producer_component_id

        # Here we rely on pipeline.components to be topologically sorted.
        assert value.producer_component_id in compile_context.node_pbs, (
            "producer component should have already been compiled.")
        producer_pb = compile_context.node_pbs[value.producer_component_id]
        for producer_context in producer_pb.contexts.contexts:
          if (not compiler_utils.is_resolver(tfx_node) or
              producer_context.name.runtime_parameter.name !=
              constants.PIPELINE_RUN_CONTEXT_TYPE_NAME):
            context_query = channel.context_queries.add()
            context_query.type.CopyFrom(producer_context.type)
            context_query.name.CopyFrom(producer_context.name)
      else:
        # Caveat: portable core requires every channel to have at least one
        # Contex. But For cases like system nodes and producer-consumer
        # pipelines, a channel may not have contexts at all. In these cases,
        # we want to use the pipeline level context as the input channel
        # context.
        context_query = channel.context_queries.add()
        context_query.type.CopyFrom(pipeline_context_pb.type)
        context_query.name.CopyFrom(pipeline_context_pb.name)

      artifact_type = value.type._get_artifact_type()  # pylint: disable=protected-access
      channel.artifact_query.type.CopyFrom(artifact_type)
      channel.artifact_query.type.ClearField("properties")

      if value.output_key:
        channel.output_key = value.output_key

      # TODO(b/158712886): Calculate min_count based on if inputs are optional.
      # min_count = 0 stands for optional input and 1 stands for required input.

    # Step 3.1: Special treatment for Resolver node
    if compiler_utils.is_resolver(tfx_node):
      resolver = tfx_node.exec_properties[resolver_node.RESOLVER_CLASS]
      if resolver == latest_artifacts_resolver.LatestArtifactsResolver:
        node.inputs.resolver_config.resolver_policy = (
            pipeline_pb2.ResolverConfig.ResolverPolicy.LATEST_ARTIFACT)
      elif resolver == latest_blessed_model_resolver.LatestBlessedModelResolver:
        node.inputs.resolver_config.resolver_policy = (
            pipeline_pb2.ResolverConfig.ResolverPolicy.LATEST_BLESSED_MODEL)
      else:
        raise ValueError("Got unsupported resolver policy: {}".format(
            resolver.type))

    # Step 4: Node outputs
    if isinstance(tfx_node, base_component.BaseComponent):
      for key, value in tfx_node.outputs.items():
        output_spec = node.outputs.outputs[key]
        artifact_type = value.type._get_artifact_type()  # pylint: disable=protected-access
        output_spec.artifact_spec.type.CopyFrom(artifact_type)

    # TODO(b/170694459): Refactor special nodes as plugins.
    # Step 4.1: Special treament for Importer node
    if compiler_utils.is_importer(tfx_node):
      self._compile_importer_node_outputs(tfx_node, node)

    # Step 5: Node parameters
    if not compiler_utils.is_resolver(tfx_node):
      for key, value in tfx_node.exec_properties.items():
        if value is None:
          continue
        # Ignore following two properties for a importer node, because they are
        # already attached to the artifacts produced by the importer node.
        if compiler_utils.is_importer(tfx_node) and (
            key == importer_node.PROPERTIES_KEY or
            key == importer_node.CUSTOM_PROPERTIES_KEY):
          continue
        parameter_value = node.parameters.parameters[key]

        # Order matters, because runtime parameter can be in serialized string.
        if isinstance(value, data_types.RuntimeParameter):
          compiler_utils.set_runtime_parameter_pb(
              parameter_value.runtime_parameter, value.name, value.ptype,
              value.default)
        elif isinstance(value, str) and re.search(
            data_types.RUNTIME_PARAMETER_PATTERN, value):
          runtime_param = json.loads(value)
          compiler_utils.set_runtime_parameter_pb(
              parameter_value.runtime_parameter, runtime_param.name,
              runtime_param.ptype, runtime_param.default)
        else:
          try:
            compiler_utils.set_field_value_pb(parameter_value.field_value,
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
    node.upstream_nodes.extend(
        sorted([
            upstream_component.id
            for upstream_component in tfx_node.upstream_nodes
        ]))
    node.downstream_nodes.extend(
        sorted([
            downstream_component.id
            for downstream_component in tfx_node.downstream_nodes
        ]))

    # Step 8: Node execution options
    node.execution_options.caching_options.enable_cache = enable_cache

    # Step 9: Per-node platform config
    if isinstance(tfx_node, base_component.BaseComponent):
      tfx_component = cast(base_component.BaseComponent, tfx_node)
      if tfx_component.platform_config:
        deployment_config.node_level_platform_configs[tfx_node.id].Pack(
            tfx_component.platform_config)

    return node

  def compile(self, tfx_pipeline: pipeline.Pipeline) -> pipeline_pb2.Pipeline:
    """Compiles a tfx pipeline into uDSL proto.

    Args:
      tfx_pipeline: A TFX pipeline.

    Returns:
      A Pipeline proto that encodes all necessary information of the pipeline.
    """
    context = _CompilerContext(
        tfx_pipeline.pipeline_info,
        compiler_utils.resolve_execution_mode(tfx_pipeline))
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

    assert compiler_utils.ensure_topological_order(tfx_pipeline.components), (
        "Pipeline components are not topologically sorted.")
    deployment_config = pipeline_pb2.IntermediateDeploymentConfig()
    if tfx_pipeline.metadata_connection_config:
      deployment_config.metadata_connection_config.Pack(
          tfx_pipeline.metadata_connection_config)
    for node in tfx_pipeline.components:
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
