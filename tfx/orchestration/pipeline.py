# Copyright 2019 Google LLC. All Rights Reserved.
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
"""Definition and related classes for TFX pipeline."""

import enum
from typing import List, Optional, Union, cast

from tfx.dsl.compiler import constants
from tfx.dsl.components.base import base_node
from tfx.dsl.components.base import executor_spec
from tfx.dsl.placeholder import placeholder as ph
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.types import channel_utils
from tfx.utils import topsort

from google.protobuf import message

# Argo's workflow name cannot exceed 63 chars:
# see https://github.com/argoproj/argo/issues/1324.
# MySQL's database name cannot exceed 64 chars:
# https://dev.mysql.com/doc/refman/5.6/en/identifiers.html
_MAX_PIPELINE_NAME_LENGTH = 63

# Pipeline root is by default specified as a RuntimeParameter when runnning on
# KubeflowDagRunner. This constant offers users an easy access to the pipeline
# root placeholder when defining a pipeline. For example,
#
# pusher = Pusher(
#     model=trainer.outputs['model'],
#     model_blessing=evaluator.outputs['blessing'],
#     push_destination=pusher_pb2.PushDestination(
#         filesystem=pusher_pb2.PushDestination.Filesystem(
#             base_directory=os.path.join(
#                 str(pipeline.ROOT_PARAMETER), 'model_serving'))))
ROOT_PARAMETER = data_types.RuntimeParameter(
    name=constants.PIPELINE_ROOT_PARAMETER_NAME, ptype=str)


class ExecutionMode(enum.Enum):
  """Execution mode of a pipeline.

  Please see this
  [RFC](https://github.com/tensorflow/community/blob/master/rfcs/20200601-tfx-udsl-semantics.md)
  for more details.
  """
  SYNC = 1
  ASYNC = 2


def add_beam_pipeline_args_to_component(component, beam_pipeline_args):
  if isinstance(component.executor_spec, executor_spec.BeamExecutorSpec):
    # Prepend pipeline-level beam_pipeline_args in front of component specific
    # ones to make component-level override pipeline-level args.
    cast(
        executor_spec.BeamExecutorSpec,
        component.executor_spec).beam_pipeline_args = beam_pipeline_args + cast(
            executor_spec.BeamExecutorSpec,
            component.executor_spec).beam_pipeline_args


class Pipeline:
  """Logical TFX pipeline object.

  Pipeline object represents the DAG of TFX components, which can be run using
  one of the pipeline orchestration systems that TFX supports. For details,
  please refer to the
  [guide](https://github.com/tensorflow/tfx/blob/master/docs/guide/build_tfx_pipeline.md).

  Attributes:
    components: A deterministic list of logical components of this pipeline,
      which are deduped and topologically sorted.
    enable_cache: Whether or not cache is enabled for this run.
    metadata_connection_config: The config to connect to ML metadata.
    execution_mode: Execution mode of the pipeline. Currently only support
      synchronous execution mode.
    beam_pipeline_args: Pipeline arguments for Beam powered Components. Use
      `with_beam_pipeline_args` to set component level Beam args.
    platform_config: Pipeline level platform config, in proto form.
  """

  def __init__(self,
               pipeline_name: str,
               pipeline_root: Union[str, ph.Placeholder],
               metadata_connection_config: Optional[
                   metadata.ConnectionConfigType] = None,
               components: Optional[List[base_node.BaseNode]] = None,
               enable_cache: Optional[bool] = False,
               beam_pipeline_args: Optional[List[str]] = None,
               platform_config: Optional[message.Message] = None,
               execution_mode: Optional[ExecutionMode] = ExecutionMode.SYNC,
               **kwargs):
    """Initialize pipeline.

    Args:
      pipeline_name: Name of the pipeline;
      pipeline_root: Path to root directory of the pipeline. This will most
        often be just a string. Some orchestrators may have limited support for
        constructing this from a Placeholder, e.g. a RuntimeInfoPlaceholder that
        refers to fields from the platform config.
      metadata_connection_config: The config to connect to ML metadata.
      components: Optional list of components to construct the pipeline.
      enable_cache: Whether or not cache is enabled for this run.
      beam_pipeline_args: Pipeline arguments for Beam powered Components.
      platform_config: Pipeline level platform config, in proto form.
      execution_mode: The execution mode of the pipeline, can be SYNC or ASYNC.
      **kwargs: Additional kwargs forwarded as pipeline args.
    """
    if len(pipeline_name) > _MAX_PIPELINE_NAME_LENGTH:
      raise ValueError(
          f'pipeline {pipeline_name} exceeds maximum allowed length: {_MAX_PIPELINE_NAME_LENGTH}.'
      )

    # TODO(b/183621450): deprecate PipelineInfo.
    self.pipeline_info = data_types.PipelineInfo(  # pylint: disable=g-missing-from-attributes
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root)
    self.enable_cache = enable_cache
    self.metadata_connection_config = metadata_connection_config
    self.execution_mode = execution_mode

    self._beam_pipeline_args = beam_pipeline_args or []

    self.platform_config = platform_config

    self.additional_pipeline_args = kwargs.get(  # pylint: disable=g-missing-from-attributes
        'additional_pipeline_args', {})

    # Calls property setter.
    self.components = components or []

  @property
  def beam_pipeline_args(self):
    """Beam pipeline args used for all components in the pipeline."""
    return self._beam_pipeline_args

  @property
  def components(self):
    """A deterministic list of logical components that are deduped and topologically sorted."""
    return self._components

  @components.setter
  def components(self, components: List[base_node.BaseNode]):
    deduped_components = set(components)
    node_by_id = {}
    # TODO(b/202822834): Use better distinction for bound channels.
    # bound_channels stores the exhaustive list of all nodes' output channels,
    # which is implicitly *bound* to the single pipeline run, as opposed to
    # manually constructed channels to fetch artifacts beyond the current
    # pipeline run.
    bound_channels = set()

    # Fills in producer map.
    for component in deduped_components:
      # Checks every node has an unique id.
      if component.id in node_by_id:
        raise RuntimeError(
            f'Duplicated node_id {component.id} for component type'
            f'{component.type}.')
      node_by_id[component.id] = component
      for key, output_channel in component.outputs.items():
        if (output_channel.producer_component_id is not None and
            output_channel.producer_component_id != component.id and
            output_channel.output_key != key):
          raise AssertionError(
              f'{output_channel} is produced more than once: '
              f'{output_channel.producer_id}[{output_channel.output_key}], '
              f'{component.id}[{key}]')
        output_channel.producer_component_id = component.id
        output_channel.output_key = key
        bound_channels.add(output_channel)

    # Connects nodes based on producer map.
    for component in deduped_components:
      for input_channel in component.inputs.values():
        for ch in (
            channel_utils.get_individual_channels(input_channel)):
          if ch not in bound_channels:
            continue
          upstream_node = node_by_id.get(ch.producer_component_id)
          if upstream_node:
            component.add_upstream_node(upstream_node)
            upstream_node.add_downstream_node(component)

    layers = topsort.topsorted_layers(
        list(deduped_components),
        get_node_id_fn=lambda c: c.id,
        get_parent_nodes=lambda c: c.upstream_nodes,
        get_child_nodes=lambda c: c.downstream_nodes)
    self._components = []
    for layer in layers:
      for component in layer:
        self._components.append(component)

    if self.beam_pipeline_args:
      for component in self._components:
        add_beam_pipeline_args_to_component(component, self.beam_pipeline_args)
