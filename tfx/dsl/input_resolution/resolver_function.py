# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Module for ResolverFunction."""
import functools
from typing import Callable, Type, Union, Mapping, Any, Optional, cast, Iterable

from tfx.dsl.control_flow import for_each_internal
from tfx.dsl.input_resolution import resolver_op
from tfx.types import artifact
from tfx.types import channel
from tfx.types import resolved_channel
from tfx.utils import doc_controls
from tfx.utils import typing_utils

_ArtifactType = Type[artifact.Artifact]
_ArtifactTypeMap = Mapping[str, Type[artifact.Artifact]]
_TypeHint = Union[_ArtifactType, _ArtifactTypeMap]
_TypeInferrer = Callable[..., Optional[_TypeHint]]


def _default_type_inferrer(*args: Any, **kwargs: Any) -> Optional[_TypeHint]:
  """Default _TypeInferrer that mirrors args[0] type."""
  del kwargs
  if len(args) == 1:
    only_arg = args[0]
    if typing_utils.is_compatible(
        only_arg, Mapping[str, channel.BaseChannel]):
      return {k: v.type for k, v in only_arg.items()}
    if isinstance(only_arg, channel.BaseChannel):
      return only_arg.type
  return None


@doc_controls.do_not_generate_docs
class ResolverFunction:
  """ResolverFunction represents a traceable function of resolver operators.

  Resolver function returns some form of channel depending on the function
  definition.

  It can return a single channel:

      trainer = Trainer(
          examples=latest_created(example_gen.outputs['examples']))

  or a dictionary of channels:

      k_fold_inputs = k_fold(example_gen.outputs['examples'], splits=5)
      trainer = Trainer(
          examples=k_fold_inputs['train'],
      )
      evaluator = Evaluator(
          examples=k_fold_inputs['eval'],
          model=trainer.outputs['model'],
      )

  or a ForEach-loopable channels:

    with ForEach(
        tfx.dsl.inputs.sequential_rolling_range(
            example_gens.outputs['examples'], n=3)) as train_window:
      trainer = Trainer(
          examples=train_window['examples'])
  """

  def __init__(
      self, f: Callable[..., resolver_op.Node],
      *,
      output_type: Optional[_TypeHint] = None,
      output_type_inferrer: _TypeInferrer = _default_type_inferrer,
      unwrap_dict_key: Optional[str] = None):
    """Constructor.

    Args:
      f: A python function consists of ResolverOp invocations.
      output_type: Static output type, either a single ArtifactType or a
          dict[str, ArtifactType]. If output_type is not given,
          output_type_inferrer will be used to infer the output type.
      output_type_inferrer: An output type inferrer function, which takes the
          same arguments as the resolver function and returns the output_type.
          If not given, default inferrer (which mirrors the args[0] type) would
          be used.
      unwrap_dict_key: If the resolver function returns ARTIFACT_MULTIMAP_LIST,
          resolver function can optionally specify unwrap_dict_key so that the
          returning Loopable is an unwrapped channel instead of a dict of
          channels. This key must be be the valid string key of the output_type.
    """
    self._function = f
    self._output_type = output_type
    if output_type is not None and (
        not typing_utils.is_compatible(output_type, _TypeHint)):
      raise ValueError(
          f'Invalid output_type: {output_type}, should be {_TypeHint}.')
    self._output_type_inferrer = output_type_inferrer
    self._unwrap_dict_key = unwrap_dict_key

  def with_output_type(self, output_type: _TypeHint):
    """Statically set output type of the resolver function.

    Use this if type inferrer cannot deterministically infer the output type
    but only caller knows the real output type.

    Examples:

        @resolver_function
        def my_resolver_function():
          ...

        examples = my_resolver_function.with_output_type(
            standard_artifacts.Examples)()

    Args:
      output_type: A static output type, either an ArtifactType or a
          dict[str, ArtifactType].

    Returns:
      A new resolver function instance with the static output type.
    """
    return ResolverFunction(
        self._function, output_type=output_type,
        output_type_inferrer=self._output_type_inferrer)

  @staticmethod
  def _try_convert_to_node(value: Any) -> Any:
    """Try converting python value to resolver_op.Node."""
    if isinstance(value, channel.BaseChannel):
      return resolver_op.InputNode(value, resolver_op.DataType.ARTIFACT_LIST)
    if typing_utils.is_compatible(value, Mapping[str, channel.BaseChannel]):
      # TODO(b/236140795): Change it to DictNode of InputNodes.
      return resolver_op.InputNode(
          value, resolver_op.DataType.ARTIFACT_MULTIMAP)
    return value

  def output_type_inferrer(self, f: _TypeInferrer) -> _TypeInferrer:
    """Decorator to register resolver function type inferrer.

    Usage:
      @resolver_function
      def latest(channel):
        ...

      @latest.output_type_inferrer
      def latest_type(channel):
        return channel.type

    Args:
      f: A type inference function to decorate.

    Returns:
      The given function.
    """
    self._output_type_inferrer = f
    return f

  def __call__(self, *args, **kwargs):
    """Invoke a resolver function.

    This would trace the @resolver_function with given arguments. BaseChannel
    argument is converted to InputNode for tracing. Return value depends on the
    actual return value of the @resolver_function.

    * If the function returns ARTIFACT_LIST type, __call__ returns a BaseChannel
    instance that can be used as a component inputs.
    * If the function returns ARTIFACT_MULTIMAP type, __call__ returns a
    Mapping[str, BaseChannel].
    * If the function returns ARTIFACT_MULTIMAP_LIST, then __call__ returns a
    intermediate object that can be unwrapped to Mapping[str, BaseChannel] with
    ForEach context manager.

    Args:
      *args: Arguments to the wrapped function.
      **kwargs: Keyword arguments to the wrapped function.

    Raises:
      RuntimeError: if output_type is invalid or unset.

    Returns:
      Resolver function result as a BaseChannels.
    """
    output_type = self._output_type or (
        self._output_type_inferrer(*args, **kwargs))
    if output_type is None:
      raise RuntimeError(
          'Unable to infer output type. Please use '
          'resolver_function.with_output_type()')

    args = [self._try_convert_to_node(v) for v in args]
    kwargs = {k: self._try_convert_to_node(v) for k, v in kwargs.items()}
    out = self.trace(*args, **kwargs)

    if out.output_data_type == resolver_op.DataType.ARTIFACT_LIST:
      if not typing_utils.is_compatible(output_type, _ArtifactType):
        raise RuntimeError(
            f'Invalid output_type {output_type}. Expected {_ArtifactType}')
      output_type = cast(_ArtifactType, output_type)
      return resolved_channel.ResolvedChannel(output_type, out)
    if out.output_data_type == resolver_op.DataType.ARTIFACT_MULTIMAP:
      if not typing_utils.is_compatible(output_type, _ArtifactTypeMap):
        raise RuntimeError(
            f'Invalid output_type {output_type}. Expected {_ArtifactTypeMap}')
      output_type = cast(_ArtifactTypeMap, output_type)
      return {
          key: resolved_channel.ResolvedChannel(artifact_type, out, key)
          for key, artifact_type in output_type.items()
      }
    if out.output_data_type == resolver_op.DataType.ARTIFACT_MULTIMAP_LIST:
      if not typing_utils.is_compatible(output_type, _ArtifactTypeMap):
        raise RuntimeError(
            f'Invalid output_type {output_type}. Expected {_ArtifactTypeMap}')
      if self._unwrap_dict_key and self._unwrap_dict_key not in output_type:
        raise RuntimeError(
            f'unwrap_dict_key {self._unwrap_dict_key} does not exist in the '
            f'output type keys: {list(output_type)}.')

      def loop_var_factory(context: for_each_internal.ForEachContext):
        result = {
            key: resolved_channel.ResolvedChannel(
                artifact_type, out, key, context)
            for key, artifact_type in output_type.items()
        }
        if self._unwrap_dict_key:
          result = result[self._unwrap_dict_key]
        return result

      return for_each_internal.Loopable(loop_var_factory)

  # TODO(b/236140660): Make trace() private and only use __call__.
  def trace(
      self,
      *args: resolver_op.Node,
      **kwargs: Any) -> resolver_op.Node:
    """Trace resolver function with given node arguments.

    Do not call this function directly; Use __call__ only.

    Tracing happens by substituting the input arguments (from BaseChannel to
    InputNode) and calling the inner python function. Traced result is the
    return value of the inner python function. Since ResolverOp invocation
    stores all the input arguments (which originated from InputNode), we can
    analyze the full ResolverOp invocation graph from the return value.

    Trace happens only once during the resolver function invocation. Traced
    resolver function (which is a resolver_op.Node) is serialized to the
    pipeline IR during compilation (i.e. inner python function is serialized),
    and the inner python function is not invoked again on IR interpretation.

    Args:
      *args: Substituted arguments to the resolver function.
      **kwargs: Substitued keyword arguments to the resolver function.

    Raises:
      RuntimeError: if the tracing fails.

    Returns:
      A traced result, which is a resolver_op.Node.
    """
    # TODO(b/188023509): Better debug support & error message.
    result = self._function(*args, **kwargs)
    if typing_utils.is_compatible(result, Mapping[str, resolver_op.Node]):
      result = resolver_op.DictNode(
          cast(Mapping[str, resolver_op.Node], result))
    if not isinstance(result, resolver_op.Node):
      raise RuntimeError(
          f'Invalid resolver function trace result {result}. Expected to '
          'return an output of ResolverOp or a dict of outputs.')
    return result


def resolver_function(
    f: Optional[Callable[..., resolver_op.OpNode]] = None, *,
    output_type: Optional[_TypeHint] = None,
    unwrap_dict_key: Optional[str] = None):
  """Decorator for the resolver function."""
  if f is None:
    def decorator(f):
      return ResolverFunction(
          f, output_type=output_type, unwrap_dict_key=unwrap_dict_key)
    return decorator
  else:
    return ResolverFunction(f)


def _deduplicate(f: Callable[..., Iterable[Any]]):
  """A decorator that removes duplicative element from iterables."""

  @functools.wraps(f)
  def wrapped(*args, **kwargs):
    seen = set()
    result = []
    for item in f(*args, **kwargs):
      if id(item) not in seen:
        seen.add(id(item))
        result.append(item)
    return result

  return wrapped


@doc_controls.do_not_generate_docs
@_deduplicate
def get_dependent_channels(
    node: resolver_op.Node) -> Iterable[channel.BaseChannel]:
  """Get a list of BaseChannels that the given node depends on."""
  for input_node in get_input_nodes(node):
    if isinstance(input_node.wrapped, channel.BaseChannel):
      yield input_node.wrapped
    elif isinstance(input_node.wrapped, dict):
      yield from input_node.wrapped.values()


@doc_controls.do_not_generate_docs
@_deduplicate
def get_input_nodes(node: resolver_op.Node) -> Iterable[resolver_op.InputNode]:
  """Get a list of input nodes that given node depends on."""
  if isinstance(node, resolver_op.InputNode):
    yield node
  elif isinstance(node, resolver_op.OpNode):
    for arg_node in node.args:
      yield from get_input_nodes(arg_node)
  elif isinstance(node, resolver_op.DictNode):
    for wrapped in node.nodes.values():
      yield from get_input_nodes(wrapped)
