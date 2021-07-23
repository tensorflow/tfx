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

import abc
from typing import Any, Dict, List, Optional, Text, Type, Sequence, Mapping

from tfx import types
from tfx.dsl.components.base import base_driver
from tfx.dsl.components.base import base_node
from tfx.dsl.input_resolution import resolver_function
from tfx.dsl.input_resolution import resolver_op
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.utils import deprecation_utils
from tfx.utils import doc_controls
from tfx.utils import json_utils

import ml_metadata as mlmd

# Constant to access resolver class from resolver exec_properties.
RESOLVER_STRATEGY_CLASS = 'resolver_class'
# Constant to access resolver config from resolver exec_properties.
RESOLVER_CONFIG = 'source_uri'


class ResolveResult:
  """The data structure to hold results from Resolver.

  Attributes:
    per_key_resolve_result: a key -> List[Artifact] dict containing the resolved
      artifacts for each source channel with the key as tag.
    per_key_resolve_state: a key -> bool dict containing whether or not the
      resolved artifacts for the channel are considered complete.
    has_complete_result: bool value indicating whether all desired artifacts
      have been resolved.
  """

  def __init__(self, per_key_resolve_result: Dict[str, List[types.Artifact]],
               per_key_resolve_state: Dict[str, bool]):
    self.per_key_resolve_result = per_key_resolve_result
    self.per_key_resolve_state = per_key_resolve_state
    self.has_complete_result = all(s for s in per_key_resolve_state.values())


class ResolverStrategy(abc.ABC):
  """Base resolver strategy class.

  A resolver strategy defines a type behavior used for input selection. A
  resolver strategy subclass must override the resolve_artifacts() function
  which takes a dict of <str, List<types.Artifact>> as parameters and return
  the resolved dict.
  """

  @doc_controls.do_not_generate_docs
  @classmethod
  def as_resolver_op(cls, input_node: resolver_op.OpNode, **kwargs):
    """ResolverOp-like usage inside resolver_function."""
    return resolver_op.OpNode(
        op_type=cls,
        arg=input_node,
        kwargs=kwargs)

  @deprecation_utils.deprecated(
      date='2020-09-24',
      instructions='Please switch to the `resolve_artifacts`.')
  @doc_controls.do_not_generate_docs
  def resolve(
      self,
      pipeline_info: data_types.PipelineInfo,
      metadata_handler: metadata.Metadata,
      source_channels: Dict[str, types.Channel],
  ) -> ResolveResult:
    """Resolves artifacts from channels by querying MLMD.

    Args:
      pipeline_info: PipelineInfo of the current pipeline. We do not want to
        query artifacts across pipeline boundary.
      metadata_handler: a read-only handler to query MLMD.
      source_channels: a key -> channel dict which contains the info of the
        source channels.

    Returns:
      a ResolveResult instance.

    Raises:
      DeprecationWarning: when it is called.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def resolve_artifacts(
      self, store: mlmd.MetadataStore,
      input_dict: Dict[str, List[types.Artifact]]
  ) -> Optional[Dict[str, List[types.Artifact]]]:
    """Resolves artifacts from channels, optionally querying MLMD if needed.

    In asynchronous execution mode, resolver classes may composed in sequence
    where the resolve_artifacts() result from the previous resolver instance
    would be passed to the next resolver instance's resolve_artifacts() inputs.

    If resolve_artifacts() returns None, it is considered as "no inputs
    available", and the remaining resolvers will not be executed.

    Also if resolve_artifacts() omits any key from the input_dict it will not
    be available from the downstream resolver instances. General recommendation
    is to preserve all keys in the input_dict unless you have specific reason.

    Args:
      store: An MLMD MetadataStore.
      input_dict: The input_dict to resolve from.

    Returns:
      If all entries has enough data after the resolving, returns the resolved
      input_dict. Otherise, return None.
    """

# Lazily register valid op_type for OpNode to avoid circular import.
resolver_op.OpNode.register_valid_op_type(ResolverStrategy)


class _ResolverDriver(base_driver.BaseDriver):
  """Driver for Resolver.

  Constructs an instance of the resolver_class specified by user with configs
  passed in by user and marks the resolved artifacts as the output of the
  Resolver.
  """

  # TODO(ruoyu): We need a better approach to let the Resolver fail on
  # incomplete data.
  def pre_execution(
      self,
      input_dict: Dict[str, types.Channel],
      output_dict: Dict[str, types.Channel],
      exec_properties: Dict[str, Any],
      driver_args: data_types.DriverArgs,
      pipeline_info: data_types.PipelineInfo,
      component_info: data_types.ComponentInfo,
  ) -> data_types.ExecutionDecision:
    # Registers contexts and execution
    contexts = self._metadata_handler.register_pipeline_contexts_if_not_exists(
        pipeline_info)
    execution = self._metadata_handler.register_execution(
        exec_properties=exec_properties,
        pipeline_info=pipeline_info,
        component_info=component_info,
        contexts=contexts)
    # Gets resolved artifacts.
    resolver_class = exec_properties[RESOLVER_STRATEGY_CLASS]
    if exec_properties[RESOLVER_CONFIG]:
      resolver = resolver_class(**exec_properties[RESOLVER_CONFIG])
    else:
      resolver = resolver_class()
    resolve_result = resolver.resolve(
        pipeline_info=pipeline_info,
        metadata_handler=self._metadata_handler,
        source_channels=input_dict.copy())
    # TODO(b/148828122): This is a temporary workaround for interactive mode.
    for k, c in output_dict.items():
      output_dict[k] = types.Channel(type=c.type).set_artifacts(
          resolve_result.per_key_resolve_result[k])
    # Updates execution to reflect artifact resolution results and mark
    # as cached.
    self._metadata_handler.update_execution(
        execution=execution,
        component_info=component_info,
        output_artifacts=resolve_result.per_key_resolve_result,
        execution_state=metadata.EXECUTION_STATE_CACHED,
        contexts=contexts)

    return data_types.ExecutionDecision(
        input_dict={},
        output_dict=resolve_result.per_key_resolve_result,
        exec_properties=exec_properties,
        execution_id=execution.id,
        use_cached_results=True)


class Resolver(base_node.BaseNode):
  """Definition for TFX Resolver.

  Resolver is a special TFX node which handles special artifact resolution
  logics that will be used as inputs for downstream nodes.

  To use Resolver, pass the followings to the Resolver constructor:

  * Name of the Resolver instance
  * A subclass of ResolverStrategy
  * Configs that will be used to construct an instance of ResolverStrategy
  * Channels to resolve with their tag, in the form of kwargs

  Here is an example:

  ```
  example_gen = ImportExampleGen(...)
  examples_resolver = Resolver(
        strategy_class=SpanRangeStrategy,
        config={'range_config': range_config},
        examples=Channel(type=Examples, producer_component_id=example_gen.id)
        ).with_id('Resolver.span_resolver')
  trainer = Trainer(
      examples=examples_resolver.outputs['examples'],
      ...)
  ```
  """

  def __init__(self,
               strategy_class: Optional[Type[ResolverStrategy]] = None,
               config: Optional[Dict[str, json_utils.JsonableType]] = None,
               function: Optional[resolver_function.ResolverFunction] = None,
               **channels: types.Channel):
    """Init function for Resolver.

    Args:
      strategy_class: Optional `ResolverStrategy` which contains the artifact
          resolution logic. One of `strategy_class` or `function`
          argument should be set.
      config: Optional dict of key to Jsonable type for constructing
          resolver_strategy.
      function: Optional `ResolverFunction` which contains the artifact
          resolution logic. User should not use this parameter directly but use
          `@resolver_stem` decorated function instead. One of `strategy_class`
          or `function` argument should be set.
      **channels: Input channels to the Resolver node as keyword arguments.
    """
    if (strategy_class is not None) + (function is not None) != 1:
      raise ValueError('Exactly one of strategy_class= or function= argument '
                       'should be given.')
    if (strategy_class is not None and
        not issubclass(strategy_class, ResolverStrategy)):
      raise TypeError('strategy_class should be ResolverStrategy, but got '
                      f'{strategy_class} instead.')
    if (function is not None and
        not isinstance(function, resolver_function.ResolverFunction)):
      raise TypeError(f'function should be ResolverFunction, but got '
                      f'{function} instead.')
    self._strategy_class = strategy_class
    self._config = config or {}
    if function is not None:
      self._resolver_function = function
    else:
      self._resolver_function = convert_strategy_to_resolver_function(
          [self._strategy_class], [self._config])
    self._input_dict = channels
    self._output_dict = {}
    for k, c in self._input_dict.items():
      if not isinstance(c, types.Channel):
        raise ValueError(
            f'Expected extra kwarg {k!r} to be of type `tfx.types.Channel` '
            f'but got {c!r} instead.')
      # TODO(b/161490287): remove static artifacts.
      self._output_dict[k] = (
          types.Channel(type=c.type).set_artifacts([c.type()]))
    super().__init__(driver_class=_ResolverDriver)

  @doc_controls.do_not_generate_docs
  def trace(
      self, input_node: resolver_op.OpNode) -> resolver_op.OpNode:
    """Get ResolverFunction's output OpNode."""
    return self._resolver_function.trace(input_node)

  @property
  @doc_controls.do_not_generate_docs
  def inputs(self) -> Dict[Text, Any]:
    return self._input_dict

  @property
  def outputs(self) -> Dict[Text, Any]:
    """Output Channel dict that contains resolved artifacts."""
    return self._output_dict

  @property
  @doc_controls.do_not_generate_docs
  def exec_properties(self) -> Dict[str, Any]:
    if self._strategy_class is None:
      return {}
    return {
        RESOLVER_STRATEGY_CLASS: self._strategy_class,
        RESOLVER_CONFIG: self._config
    }


@doc_controls.do_not_generate_docs
def convert_strategy_to_resolver_function(
    strategy_class_list: Sequence[Type[ResolverStrategy]],
    config_list: Sequence[Mapping[str, json_utils.JsonableType]],
) -> resolver_function.ResolverFunction:
  """Creates ResolverFunction that runs a list of ResolverStrategy."""

  @resolver_function.resolver_function
  def impl(input_node):
    result = input_node
    for strategy_cls, config in zip(strategy_class_list, config_list):
      result = strategy_cls.as_resolver_op(input_node, **config)
    return result

  return impl
