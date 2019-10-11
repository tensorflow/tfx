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
"""TFX Resolver definition."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, Text, Type

from tfx import types
from tfx.components.base import base_driver
from tfx.components.base import base_node
from tfx.dsl.resolvers import base_resolver
from tfx.orchestration import data_types
from tfx.types import node_common
from tfx.utils import json_utils

# Constant to access resolver class from resolver exec_properties.
RESOLVER_CLASS = 'resolver_class'
# Constant to access resolver config from resolver exec_properties.
RESOLVER_CONFIGS = 'source_uri'


class ResolverDriver(base_driver.BaseDriver):
  """Driver for Resolver.

  Constructs an instance of the resolver_class specified by user with configs
  passed in by user and marks the resolved artifacts as the output of the
  ResolverNode.
  """

  def pre_execution(
      self,
      input_dict: Dict[Text, types.Channel],
      output_dict: Dict[Text, types.Channel],
      exec_properties: Dict[Text, Any],
      driver_args: data_types.DriverArgs,
      pipeline_info: data_types.PipelineInfo,
      component_info: data_types.ComponentInfo,
  ) -> data_types.ExecutionDecision:
    resolver_class = exec_properties[RESOLVER_CLASS]
    if exec_properties[RESOLVER_CONFIGS]:
      resolver = resolver_class(**exec_properties[RESOLVER_CONFIGS])
    else:
      resolver = resolver_class()
    resolve_result = resolver.resolve(
        metadata_handler=self._metadata_handler,
        source_channels=input_dict.copy())
    if not resolve_result.has_complete_result:
      raise RuntimeError('Cannot resolve all artifacts as needed.')

    return data_types.ExecutionDecision(
        input_dict={},
        output_dict=resolve_result.per_key_resolve_result,
        exec_properties=exec_properties,
        execution_id=self._register_execution(
            exec_properties={},
            pipeline_info=pipeline_info,
            component_info=component_info),
        use_cached_results=True)


class ResolverNode(base_node.BaseNode):
  """Definition for TFX ResolverNode.

  ResolverNode is a special TFX node which handles special artifact resolution
  logics that will be used as inputs for downstream nodes.

  To use ResolverNode, pass the followings to the ResolverNode constructor:
    a. name of the ResolverNode instance
    g. a subclass of BaseResolver
    c. the configs that will be used to construct an instance of (a)
    d. channels to resolve with their tag, in the form of kwargs
  Here is an example:

  ...
  example_gen = ImportExampleGen(...)
  latest_five_examples_resolver = ResolverNode(
      instance_name='latest_five_examples_resolver',
      resolver_class=latest_artifacts_resolver.LatestArtifactsResolver,
      resolver_config={'desired_num_of_artifacts' : 5},
      examples=example_gen.outputs['examples'])
  trainer = MyTrainer(
      examples=latest_model_resolver.outputs['examples'],
      user_module=...)
  ...

  Attributes:
    _resolver_class: the class of the resolver.
    _resolver_configs: the configs that will be used to construct an instance of
      _resolver_class.
  """

  DRIVER_CLASS = ResolverDriver

  def __init__(self,
               instance_name: Text,
               resolver_class: Type[base_resolver.BaseResolver],
               resolver_configs: Dict[Text, json_utils.JsonableType] = None,
               **kwargs: types.Channel):
    """Init function for ResolverNode.

    Args:
      instance_name: the name of the ResolverNode instance.
      resolver_class: the URI to the resource that needs to be registered.
      resolver_configs: a dict of key to JsonableType representing configs that
        will be used to construct the resolver.
      **kwargs: a key -> Channel dict, describing what are the Channels to be
        resolved. This is set by user through keyword args.
    """
    self._resolver_class = resolver_class
    self._resolver_configs = resolver_configs or {}
    self._input_dict = kwargs
    self._output_dict = kwargs.copy()
    super(ResolverNode, self).__init__(instance_name=instance_name)

  @property
  def inputs(self) -> node_common._PropertyDictWrapper:
    return node_common._PropertyDictWrapper(self._input_dict)  # pylint: disable=protected-access

  @property
  def outputs(self) -> node_common._PropertyDictWrapper:
    return node_common._PropertyDictWrapper(self._output_dict)  # pylint: disable=protected-access

  @property
  def exec_properties(self) -> Dict[Text, Any]:
    return {
        RESOLVER_CLASS: self._resolver_class,
        RESOLVER_CONFIGS: self._resolver_configs
    }
