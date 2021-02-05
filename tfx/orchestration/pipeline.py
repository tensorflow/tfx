# Lint as: python2, python3
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import enum
import json
import os
from typing import List, Optional, Text, cast

from tfx.dsl.components.base import base_node
from tfx.dsl.components.base import executor_spec
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.utils import topsort

from google.protobuf import message

# Argo's workflow name cannot exceed 63 chars:
# see https://github.com/argoproj/argo/issues/1324.
# MySQL's database name cannot exceed 64 chars:
# https://dev.mysql.com/doc/refman/5.6/en/identifiers.html
MAX_PIPELINE_NAME_LENGTH = 63

# Name of pipeline_root parameter.
_PIPELINE_ROOT = 'pipeline-root'

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
ROOT_PARAMETER = data_types.RuntimeParameter(name=_PIPELINE_ROOT, ptype=Text)


class ExecutionMode(enum.Enum):
  """Execution mode of a pipeline.

  Please see `this RFC
  <https://github.com/tensorflow/community/blob/master/rfcs/20200601-tfx-udsl-semantics.md>`
  for more details.
  """
  SYNC = 1
  ASYNC = 2


class Pipeline(object):
  """Logical TFX pipeline object.

  Attributes:
    pipeline_args: Kwargs used to create real pipeline implementation. This is
      forwarded to PipelineRunners instead of consumed in this class. This
      should include:
      - pipeline_name: Required. The unique name of this pipeline.
      - pipeline_root: Required. The root of the pipeline outputs.
    components: Logical components of this pipeline. When read, this is in
      topologically sorted order.
    pipeline_info: An instance of data_types.PipelineInfo that contains basic
      properties of the pipeline.
    enable_cache: Whether or not cache is enabled for this run.
    metadata_connection_config: The config to connect to ML metadata.
    execution_mode: Execution mode of the pipeline. Currently only support
      synchronous execution mode.
    beam_pipeline_args: Pipeline arguments for Beam powered Components.
    platform_config: Pipeline level platform config, in proto form.
    additional_pipeline_args: Other pipeline args.
  """

  def __init__(self,
               pipeline_name: Text,
               pipeline_root: Text,
               metadata_connection_config: Optional[
                   metadata.ConnectionConfigType] = None,
               components: Optional[List[base_node.BaseNode]] = None,
               enable_cache: Optional[bool] = False,
               beam_pipeline_args: Optional[List[Text]] = None,
               platform_config: Optional[message.Message] = None,
               execution_mode: Optional[ExecutionMode] = ExecutionMode.SYNC,
               **kwargs):
    """Initialize pipeline.

    Args:
      pipeline_name: Name of the pipeline;
      pipeline_root: Path to root directory of the pipeline;
      metadata_connection_config: The config to connect to ML metadata.
      components: A list of components in the pipeline (optional only for
        backward compatible purpose to be used with deprecated
        PipelineDecorator).
      enable_cache: Whether or not cache is enabled for this run.
      beam_pipeline_args: Pipeline arguments for Beam powered Components.
      platform_config: Pipeline level platform config, in proto form.
      execution_mode: The execution mode of the pipeline, can be SYNC or ASYNC.
      **kwargs: Additional kwargs forwarded as pipeline args.
    """
    if len(pipeline_name) > MAX_PIPELINE_NAME_LENGTH:
      raise ValueError('pipeline name %s exceeds maximum allowed lenght' %
                       pipeline_name)
    pipeline_args = dict(kwargs)

    self.pipeline_info = data_types.PipelineInfo(
        pipeline_name=pipeline_name, pipeline_root=pipeline_root)
    self.enable_cache = enable_cache
    self.metadata_connection_config = metadata_connection_config
    self.execution_mode = execution_mode

    self.beam_pipeline_args = beam_pipeline_args or []

    self.platform_config = platform_config

    self.additional_pipeline_args = pipeline_args.get(
        'additional_pipeline_args', {})

    # Store pipeline_args in a json file only when temp file exists.
    pipeline_args.update({
        'pipeline_name': pipeline_name,
        'pipeline_root': pipeline_root,
    })
    if 'TFX_JSON_EXPORT_PIPELINE_ARGS_PATH' in os.environ:
      pipeline_args_path = os.environ.get('TFX_JSON_EXPORT_PIPELINE_ARGS_PATH')
      with open(pipeline_args_path, 'w') as f:
        json.dump(pipeline_args, f)

    # Calls property setter.
    self.components = components or []

    # TODO(b/156000550): Currently `beam_pipeline_args` is set at pipeline
    # level in the SDK. However it is subject to change, to move to per-node
    # configuration. We will need to change the following logic accordingly.
    if not self.beam_pipeline_args:
      return
    for component in components:
      if isinstance(component.executor_spec, executor_spec.ExecutorClassSpec):
        cast(executor_spec.ExecutorClassSpec,
             component.executor_spec).extra_flags.extend(beam_pipeline_args)

  @property
  def components(self):
    """A deterministic list of logical components that are deduped and topologically sorted."""
    return self._components

  @components.setter
  def components(self, components: List[base_node.BaseNode]):
    deduped_components = set(components)
    producer_map = {}
    instances_per_component_type = collections.defaultdict(set)

    # Fills in producer map.
    for component in deduped_components:
      # Guarantees every component of a component type has unique component_id.
      if component.id in instances_per_component_type[component.type]:
        raise RuntimeError('Duplicated component_id %s for component type %s' %
                           (component.id, component.type))
      instances_per_component_type[component.type].add(component.id)
      for key, output_channel in component.outputs.items():
        assert not producer_map.get(
            output_channel), '{} produced more than once'.format(output_channel)
        producer_map[output_channel] = component
        output_channel.producer_component_id = component.id
        output_channel.output_key = key

    # Connects nodes based on producer map.
    for component in deduped_components:
      for i in component.inputs.values():
        if producer_map.get(i):
          component.add_upstream_node(producer_map[i])
          producer_map[i].add_downstream_node(component)

    layers = topsort.topsorted_layers(
        list(deduped_components),
        get_node_id_fn=lambda c: c.id,
        get_parent_nodes=lambda c: c.upstream_nodes,
        get_child_nodes=lambda c: c.downstream_nodes)
    self._components = []
    for layer in layers:
      for component in layer:
        self._components.append(component)
