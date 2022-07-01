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

from tfx.dsl.input_resolution import resolver_op
from tfx.types import artifact
from tfx.types import channel
from tfx.types import resolved_channel
from tfx.utils import doc_controls
from tfx.utils import typing_utils

_ArtifactType = Type[artifact.Artifact]
_ArtifactTypeMap = Mapping[str, Type[artifact.Artifact]]
_TypeHint = Union[_ArtifactType, _ArtifactTypeMap]


@doc_controls.do_not_generate_docs
class ResolverFunction:
  """ResolverFunction represents a traceable function of resolver operators.

  ResolverFunction as a whole, takes an ArtifactMultiMap as an argument and
  returns an ArtifactMultiMap.

  Usage:
      @resolver_function
      def trainer_resolver_fn(root):
        result = FooOp(root, foo=1)
        result = BarOp(result, bar='x')
        return result

      trainer_resolver = Resolver(
          function=trainer_resolver_fn,
          examples=example_gen.outputs['examples'],
          ...)
  """

  def __init__(
      self, f: Callable[..., resolver_op.Node],
      type_hint: Optional[_TypeHint] = None):
    self._function = f
    self._type_hint = type_hint
    if type_hint is not None and (
        not typing_utils.is_compatible(type_hint, _TypeHint)):
      raise ValueError(
          f'Invalid type_hint: {type_hint}, should be {_TypeHint}.')

  def with_type_hint(self, type_hint: _TypeHint):
    return ResolverFunction(self._function, type_hint)

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

  @staticmethod
  def _try_infer_type(args) -> Optional[_TypeHint]:
    if len(args) == 1:
      only_arg = args[0]
      if typing_utils.is_compatible(
          only_arg, Mapping[str, channel.BaseChannel]):
        return {k: v.type for k, v in only_arg.items()}
    return None

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
      RuntimeError: if type_hint is invalid or unset.

    Returns:
      Resolver function result as a BaseChannels.
    """
    type_hint = self._type_hint or self._try_infer_type(args)
    args = [self._try_convert_to_node(v) for v in args]
    kwargs = {k: self._try_convert_to_node(v) for k, v in kwargs.items()}
    out = self.trace(*args, **kwargs)
    if type_hint is None:
      raise RuntimeError(
          'type_hint not set. Please use resolver_function.with_type_hint()')
    if out.output_data_type == resolver_op.DataType.ARTIFACT_LIST:
      if not typing_utils.is_compatible(type_hint, _ArtifactType):
        raise RuntimeError(
            f'Invalid type_hint {type_hint}. Expected {_ArtifactType}')
      type_hint = cast(_ArtifactType, type_hint)
      return resolved_channel.ResolvedChannel(type_hint, out)
    if out.output_data_type == resolver_op.DataType.ARTIFACT_MULTIMAP:
      if not typing_utils.is_compatible(type_hint, _ArtifactTypeMap):
        raise RuntimeError(
            f'Invalid type_hint {type_hint}. Expected {_ArtifactTypeMap}')
      type_hint = cast(_ArtifactTypeMap, type_hint)
      return {
          key: resolved_channel.ResolvedChannel(artifact_type, out, key)
          for key, artifact_type in type_hint.items()
      }
    if out.output_data_type == resolver_op.DataType.ARTIFACT_MULTIMAP_LIST:
      if not typing_utils.is_compatible(type_hint, _ArtifactTypeMap):
        raise RuntimeError(
            f'Invalid type_hint {type_hint}. Expected {_ArtifactTypeMap}')
      # TODO(b/237363715): Return ForEach-able value.
      raise NotImplementedError(
          'ARTIFACT_MULTIMAP_LIST return value is not yet supported.')

  # TODO(b/236140660): Make trace() private and only use __call__.
  def trace(
      self,
      *args: resolver_op.Node,
      **kwargs: Any) -> resolver_op.Node:
    """Trace resolver function with given node arguments."""
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
    type_hint: Optional[_TypeHint] = None):
  """Decorator for the resolver function."""
  if type_hint:
    if not typing_utils.is_compatible(type_hint, _TypeHint):
      raise ValueError(f'Invalid type_hint {type_hint}. Expected {_TypeHint}')
    def decorator(f):
      return ResolverFunction(f, type_hint)
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
