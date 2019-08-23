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
import functools
import json
import os

import tensorflow as tf

from typing import List, Optional, Text
from ml_metadata.proto import metadata_store_pb2
from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import
from tfx.components.base import base_component
from tfx.orchestration import data_types
from tfx.orchestration import metadata

# Argo's workflow name cannot exceed 63 chars:
# see https://github.com/argoproj/argo/issues/1324.
# MySQL's database name cannot exceed 64 chars:
# https://dev.mysql.com/doc/refman/5.6/en/identifiers.html
MAX_PIPELINE_NAME_LENGTH = 63


@deprecation.deprecated(
    None,
    'PipelineDecorator is no longer needed. Please construct a pipeline '
    'directly from a list of components  using the constructor call to '
    'pipeline.Pipeline.',
)
class PipelineDecorator(object):
  """Pipeline decorator that has pipeline-level specification."""

  def __init__(self, **kwargs):
    self._pipeline = self._new_pipeline(**kwargs)

  # TODO(b/126411144): Come up with a better style to construct TFX pipeline.
  def __call__(self, func):

    @functools.wraps(func)
    def decorated():
      self._pipeline.components = func()
      return self._pipeline

    return decorated

  def _new_pipeline(self, **kwargs):
    return Pipeline(**kwargs)


class Pipeline(object):
  """Logical TFX pipeline object.

  Attributes:
    pipeline_args: kwargs used to create real pipeline implementation. This is
      forwarded to PipelineRunners instead of consumed in this class. This
      should include:
      - pipeline_name: Required. The unique name of this pipeline.
      - pipeline_root: Required. The root of the pipeline outputs.
    components: logical components of this pipeline.
    pipeline_info: An instance of data_types.PipelineInfo that contains basic
      properties of the pipeline.
    enable_cache: whether or not cache is enabled for this run.
    metadata_connection_config: the config to connect to ML metadata.
    additional_pipeline_args: other pipeline args. e.g. beam runner args.
  """

  def __init__(self,
               pipeline_name: Text,
               pipeline_root: Text,
               metadata_connection_config: Optional[
                   metadata_store_pb2.ConnectionConfig] = None,
               components: Optional[List[base_component.BaseComponent]] = None,
               enable_cache: Optional[bool] = False,
               metadata_db_root: Optional[Text] = None,
               **kwargs):
    """Initialize pipeline.

    Args:
      pipeline_name: name of the pipeline;
      pipeline_root: path to root directory of the pipeline;
      metadata_connection_config: the config to connect to ML metadata.
      components: a list of components in the pipeline (optional only for
        backward compatible purpose to be used with deprecated
        PipelineDecorator).
      enable_cache: whether or not cache is enabled for this run.
      metadata_db_root: Deprecated. the uri to the metadata database root.
        Deprecated and will be removed in future version.
        Please use metadata_connection_config instead.
      **kwargs: additional kwargs forwarded as pipeline args.
        - beam_pipeline_args: Beam pipeline args for beam jobs within executor.
          Executor will use beam DirectRunner as Default.
    """
    if len(pipeline_name) > MAX_PIPELINE_NAME_LENGTH:
      raise ValueError('pipeline name %s exceeds maximum allowed lenght' %
                       pipeline_name)
    # TODO(b/138406006): Deprecate pipeline args after 0.14 release.
    self.pipeline_args = dict(kwargs)
    self.pipeline_args.update({
        'pipeline_name': pipeline_name,
        'pipeline_root': pipeline_root,
    })

    self.pipeline_info = data_types.PipelineInfo(
        pipeline_name=pipeline_name, pipeline_root=pipeline_root)
    self.enable_cache = enable_cache
    if metadata_connection_config:
      self.metadata_connection_config = metadata_connection_config
      assert not metadata_db_root, ('At most one of metadata_connection_config '
                                    'and metadata_db_root should be set')
    else:
      # TODO(b/138406006): Drop metadata_db_root support after 0.14 release.
      # We also need to make metadata_connection_config required.
      tf.logging.warning(
          'metadata_db_root is deprecated, metadata_connection_config will be required in next release'
      )
      if metadata_db_root:
        self.metadata_connection_config = metadata.sqlite_metadata_connection_config(
            metadata_db_root)
      else:
        self.metadata_connection_config = None

    self.additional_pipeline_args = self.pipeline_args.get(
        'additional_pipeline_args', {})

    # Store pipeline_args in a json file only when temp file exists.
    if 'TFX_JSON_EXPORT_PIPELINE_ARGS_PATH' in os.environ:
      pipeline_args_path = os.environ.get('TFX_JSON_EXPORT_PIPELINE_ARGS_PATH')
      with open(pipeline_args_path, 'w') as f:
        json.dump(self.pipeline_args, f)

    # Calls property setter.
    self.components = components or []

  @property
  def components(self):
    """A list of logical components that are deduped and topological sorted."""
    return self._components

  @components.setter
  def components(self, components: List[base_component.BaseComponent]):
    deduped_components = set(components)
    producer_map = {}
    instances_per_component_type = collections.defaultdict(set)

    # Fills in producer map.
    for component in deduped_components:
      # Guarantees every component of a component type has unique component_id.
      if component.component_id in instances_per_component_type[
          component.component_type]:
        raise RuntimeError('Duplicated component_id %s for component type %s' %
                           (component.component_id, component.component_type))
      instances_per_component_type[component.component_type].add(
          component.component_id)
      for key, output_channel in component.outputs.get_all().items():
        assert not producer_map.get(
            output_channel), '{} produced more than once'.format(output_channel)
        producer_map[output_channel] = component
        # Fill in detailed artifact properties.
        for artifact in output_channel.get():
          artifact.name = key
          artifact.pipeline_name = self.pipeline_info.pipeline_name
          artifact.producer_component = component.component_id

    # Connects nodes based on producer map.
    for component in deduped_components:
      for i in component.inputs.get_all().values():
        if producer_map.get(i):
          component.add_upstream_node(producer_map[i])
          producer_map[i].add_downstream_node(component)

    self._components = []
    visited = set()

    # Finds the nodes with indegree 0.
    current_layer = [c for c in deduped_components if not c.upstream_nodes]
    # Sorts component in topological order.
    while current_layer:
      next_layer = []
      for component in current_layer:
        self._components.append(component)
        visited.add(component)
        for downstream_node in component.downstream_nodes:
          if downstream_node.upstream_nodes.issubset(visited):
            next_layer.append(downstream_node)
      current_layer = next_layer
    # If there is a cycle in the graph, upon visiting the cycle, no node will be
    # ready to be processed because it is impossible to find a single node that
    # has all its dependencies visited.
    if len(self._components) < len(deduped_components):
      raise RuntimeError('There is a cycle in the pipeline')
