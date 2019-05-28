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

import functools
from typing import List, Optional, Text

from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import
from tfx.components.base import base_component


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

  Args:
    pipeline_name: name of the pipeline;
    pipeline_root: path to root directory of the pipeline;
    components: a list of components in the pipeline (optional only for backward
      compatible purpose to be used with deprecated PipelineDecorator).
    kwargs: additional kwargs forwarded as pipeline args.
  Attributes:
    pipeline_args: kwargs used to create real pipeline implementation. This is
      forwarded to PipelineRunners instead of consumed in this class. This
      should include:
      - pipeline_name: Required. The unique name of this pipeline.
      - pipeline_root: Required. The root of the pipeline outputs.
    components: logical components of this pipeline.
  """

  def __init__(self,
               pipeline_name: Text,
               pipeline_root: Text,
               components: Optional[List[base_component.BaseComponent]] = None,
               **kwargs):
    # TODO(b/126565661): Add more documentation on this.
    self.pipeline_args = kwargs
    self.pipeline_args.update({
        'pipeline_name': pipeline_name,
        'pipeline_root': pipeline_root
    })
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

    # Fills in producer map.
    for component in deduped_components:
      for o in component.outputs.get_all().values():
        assert not producer_map.get(0), '{} produced more than once'.format(o)
        producer_map[o] = component

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
