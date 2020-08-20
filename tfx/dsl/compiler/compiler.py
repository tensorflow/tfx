# Lint as: python2, python3
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
"""Compiles a TFX pipeline or a component into a uDSL IR proto."""

# TODO(b/149535307): Remove __future__ imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import re

from tfx.components.base import base_driver
from tfx.components.base import base_node
from tfx.components.common_nodes import resolver_node
from tfx.dsl.compiler import compiler_utils
from tfx.dsl.compiler import constants
from tfx.dsl.experimental import latest_artifacts_resolver
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.orchestration import data_types
from tfx.orchestration import pipeline
from tfx.proto.orchestration import local_deployment_config_pb2
from tfx.proto.orchestration import pipeline_pb2


class _CompilerContext(object):
  """Encapsulates resources needed to compile a pipeline."""

  def __init__(self, pipeline_info: data_types.PipelineInfo):
    self.pipeline_info = pipeline_info
    self.node_pbs = {}


class Compiler(object):
  """Compiles a TFX pipeline or a component into a uDSL IR proto."""

  def __init__(self):
    pass

  def _compile_node(
      self, tfx_node: base_node.BaseNode, compile_context: _CompilerContext,
      deployment_config: pipeline_pb2.IntermediateDeploymentConfig
  ) -> pipeline_pb2.PipelineNode:
    """Compiles an individual TFX node into a PipelineNode proto.

    Args:
      tfx_node: A TFX node.
      compile_context: Resources needed to compile the node.
      deployment_config: Intermediate deployment config to set. Will include
        related specs for executors, drivers and platform specific configs.

    Returns:
      A PipelineNode proto that encodes information of the node.
    """
    node = pipeline_pb2.PipelineNode()

    # Step 1: Node info
    node.node_info.type.name = tfx_node.type
    node.node_info.id = tfx_node.id

    # Step 2: Node Context
    pipeline_context_pb = node.contexts.contexts.add()
    pipeline_context_pb.type.name = constants.PIPELINE_CONTEXT_TYPE_NAME
    pipeline_context_pb.name.field_value.string_value = compile_context.pipeline_info.pipeline_context_name

    pipeline_run_context_pb = node.contexts.contexts.add()
    pipeline_run_context_pb.type.name = constants.PIPELINE_RUN_CONTEXT_TYPE_NAME
    compiler_utils.set_runtime_parameter_pb(
        pipeline_run_context_pb.name.runtime_parameter,
        constants.PIPELINE_RUN_CONTEXT_TYPE_NAME, str)

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
    if compiler_utils.is_component(tfx_node):
      for key, value in tfx_node.outputs.items():
        output_spec = node.outputs.outputs[key]
        artifact_type = value.type._get_artifact_type()  # pylint: disable=protected-access
        output_spec.artifact_spec.type.CopyFrom(artifact_type)

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
        elif isinstance(value, str) and re.search(
            data_types.RUNTIME_PARAMETER_PATTERN, value):
          runtime_param = json.loads(value)
          compiler_utils.set_runtime_parameter_pb(
              parameter_value.runtime_parameter, runtime_param.name,
              runtime_param.ptype, runtime_param.default)
        elif isinstance(value, str):
          parameter_value.field_value.string_value = value
        elif isinstance(value, int):
          parameter_value.field_value.int_value = value
        elif isinstance(value, float):
          parameter_value.field_value.double_value = value
        else:
          raise ValueError(
              "Component {} got unsupported parameter {} with type {}.".format(
                  tfx_node.id, key, type(value)))

    # Step 6: Executor spec and optional driver spec for components
    if compiler_utils.is_component(tfx_node):
      executor_spec = tfx_node.executor_spec.encode()
      deployment_config.executor_specs[tfx_node.id].Pack(executor_spec)

      # TODO(b/163433174): Remove specialized logic once generalization of
      # driver spec is done.
      if tfx_node.driver_class != base_driver.BaseDriver:
        driver_class_path = "{}.{}".format(tfx_node.driver_class.__module__,
                                           tfx_node.driver_class.__name__)
        driver_spec = local_deployment_config_pb2.ExecutableSpec()
        driver_spec.python_class_executable_spec.class_path = driver_class_path
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
    # TODO(kennethyang): Add support for node execution options.

    return node

  def compile(self, tfx_pipeline: pipeline.Pipeline) -> pipeline_pb2.Pipeline:
    """Compiles a tfx pipeline into uDSL proto.

    Args:
      tfx_pipeline: A TFX pipeline.

    Returns:
      A Pipeline proto that encodes all necessary information of the pipeline.
    """
    context = _CompilerContext(tfx_pipeline.pipeline_info)
    pipeline_pb = pipeline_pb2.Pipeline()
    pipeline_pb.pipeline_info.id = context.pipeline_info.pipeline_name
    compiler_utils.set_runtime_parameter_pb(
        pipeline_pb.runtime_spec.pipeline_root.runtime_parameter,
        constants.PIPELINE_ROOT_PARAMETER_NAME, str,
        context.pipeline_info.pipeline_root)
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
      node_pb = self._compile_node(node, context, deployment_config)
      pipeline_or_node = pipeline_pb.PipelineOrNode()
      pipeline_or_node.pipeline_node.CopyFrom(node_pb)
      # TODO(b/158713812): Support sub-pipeline.
      pipeline_pb.nodes.append(pipeline_or_node)
      context.node_pbs[node.id] = node_pb

    pipeline_pb.deployment_config.Pack(deployment_config)
    # Currently only synchronous mode is supported
    pipeline_pb.execution_mode = pipeline_pb2.Pipeline.ExecutionMode.SYNC
    return pipeline_pb
