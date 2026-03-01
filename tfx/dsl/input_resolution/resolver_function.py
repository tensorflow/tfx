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
import contextlib
import inspect
from typing import Callable, Type, Union, Mapping, Any, Optional, Sequence, cast, overload

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
_TypeInferFn = Callable[..., Optional[_TypeHint]]
_LoopableTransformFn = Callable[[Mapping[str, channel.BaseChannel]], Any]


def _default_type_inferrer(*args: Any, **kwargs: Any) -> Optional[_TypeHint]:
  """Default _TypeInferrer that mirrors args[0] type."""
  del kwargs
  if len(args) == 1:
    only_arg = args[0]
    if typing_utils.is_compatible(only_arg, Mapping[str, channel.BaseChannel]):
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
      self,
      f: Callable[..., resolver_op.Node],
      *,
      output_type: Optional[_TypeHint] = None,
      output_type_inferrer: _TypeInferFn = _default_type_inferrer,
      loopable_transform: Optional[_LoopableTransformFn] = None,
  ):
    """Constructor.

    Args:
      f: A python function consists of ResolverOp invocations.
      output_type: Static output type, either a single ArtifactType or a
        dict[str, ArtifactType]. If output_type is not given,
        output_type_inferrer will be used to infer the output type.
      output_type_inferrer: An output type inferrer function, which takes the
        same arguments as the resolver function and returns the output_type. If
        not given, default inferrer (which mirrors the args[0] type) would be
        used.
      loopable_transform: If the resolver function returns
        ARTIFACT_MULTIMAP_LIST, resolver function invocation returns a loopable
        value (the type that can be used with ForEach). This transform function
        is applied to the `ForEach` with clause target value (often called loop
        variable) so that the `as` variables are more easy-to-use. For example,
        one can automatically unwrap the dict key if the dict key is
        internal-only value and would not be exposed to the user side, using
        `loopable_transform = lambda d: d[key]`.
    """
    # This instance is a decorated callable so it should reuse the decorated
    # function's system attributes.
    if hasattr(f, '__name__'):
      self.__name__ = f.__name__
    if hasattr(f, '__qualname__'):
      self.__qualname__ = f.__qualname__
    if hasattr(f, '__module__'):
      self.__module__ = f.__module__
    if hasattr(f, '__doc__'):
      self.__doc__ = f.__doc__
    self.__signature__ = inspect.signature(f)
    self.__wrapped__ = f

    self._output_type = output_type
    self._output_type_inferrer = output_type_inferrer
    self._loopable_transform = loopable_transform
    self._invocation = None

  @contextlib.contextmanager
  def given_output_type(self, output_type: _TypeHint) -> 'ResolverFunction':
    """Temporarily patches output_type."""
    if not typing_utils.is_compatible(output_type, _TypeHint):
      raise ValueError(
          f'Invalid output_type: {output_type}, should be {_TypeHint}.'
      )
    original = self._output_type
    try:
      self._output_type = output_type
      yield self
    finally:
      self._output_type = original

  @contextlib.contextmanager
  def given_invocation(
      self,
      f: Callable[..., Any],
      *,
      args: Sequence[Any],
      kwargs: Mapping[str, Any],
  ) -> 'ResolverFunction':
    """Temporarily patches Invocation."""
    invocation = resolved_channel.Invocation(
        function=f, args=args, kwargs=kwargs
    )
    if self._invocation is not None:
      raise RuntimeError(f'{self.__name__} has already given an invocation.')
    self._invocation = invocation
    try:
      yield self
    finally:
      self._invocation = None

  @staticmethod
  def _try_convert_to_node(value: Any) -> Any:
    """Try converting python value to resolver_op.Node."""
    if isinstance(value, channel.BaseChannel):
      return resolver_op.InputNode(value)
    if typing_utils.is_compatible(value, Mapping[str, channel.BaseChannel]):
      return resolver_op.DictNode(
          {
              input_key: resolver_op.InputNode(input_channel)
              for input_key, input_channel in value.items()
          }
      )
    return value

  def output_type_inferrer(self, f: _TypeInferFn) -> _TypeInferFn:
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
        self._output_type_inferrer(*args, **kwargs)
    )
    if output_type is None:
      raise RuntimeError(
          'Unable to infer output type. Please use '
          'resolver_function.with_output_type()'
      )

    args = [self._try_convert_to_node(v) for v in args]
    kwargs = {k: self._try_convert_to_node(v) for k, v in kwargs.items()}
    out = self.trace(*args, **kwargs)

    invocation = self._invocation or resolved_channel.Invocation(
        function=self, args=args, kwargs=kwargs
    )

    if out.output_data_type == resolver_op.DataType.ARTIFACT_LIST:
      if self._loopable_transform is not None:
        raise TypeError(
            'loopable_transform is not applicable for ARTIFACT_LIST output'
        )
      if not typing_utils.is_compatible(output_type, _ArtifactType):
        raise RuntimeError(
            f'Invalid output_type {output_type}. Expected {_ArtifactType}'
        )
      output_type = cast(_ArtifactType, output_type)
      return resolved_channel.ResolvedChannel(
          artifact_type=output_type, output_node=out, invocation=invocation
      )
    if out.output_data_type == resolver_op.DataType.ARTIFACT_MULTIMAP:
      if self._loopable_transform is not None:
        raise TypeError(
            'loopable_transform is not applicable for ARTIFACT_MULTIMAP output'
        )
      if not typing_utils.is_compatible(output_type, _ArtifactTypeMap):
        raise RuntimeError(
            f'Invalid output_type {output_type}. Expected {_ArtifactTypeMap}'
        )
      output_type = cast(_ArtifactTypeMap, output_type)
      result = {}
      for key, artifact_type in output_type.items():
        result[key] = resolved_channel.ResolvedChannel(
            artifact_type=artifact_type,
            output_node=out,
            output_key=key,
            invocation=invocation,
        )
      return result
    if out.output_data_type == resolver_op.DataType.ARTIFACT_MULTIMAP_LIST:
      if not typing_utils.is_compatible(output_type, _ArtifactTypeMap):
        raise RuntimeError(
            f'Invalid output_type {output_type}. Expected {_ArtifactTypeMap}'
        )

      def loop_var_factory(for_each_context: for_each_internal.ForEachContext):
        result = {}
        for key, artifact_type in output_type.items():
          result[key] = resolved_channel.ResolvedChannel(
              artifact_type=artifact_type,
              output_node=out,
              output_key=key,
              invocation=invocation,
              for_each_context=for_each_context,
          )
        if self._loopable_transform:
          result = self._loopable_transform(result)
        return result

      return for_each_internal.Loopable(loop_var_factory)

  # TODO(b/236140660): Make trace() private and only use __call__.
  def trace(self, *args: resolver_op.Node, **kwargs: Any) -> resolver_op.Node:
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
    result = self.__wrapped__(*args, **kwargs)
    if typing_utils.is_compatible(result, Mapping[str, resolver_op.Node]):
      result = resolver_op.DictNode(
          cast(Mapping[str, resolver_op.Node], result)
      )
    if not isinstance(result, resolver_op.Node):
      raise RuntimeError(
          f'Invalid resolver function trace result {result}. Expected to '
          'return an output of ResolverOp or a dict of outputs.'
      )
    return result


@overload
def resolver_function(
    f: Callable[..., resolver_op.OpNode],
) -> ResolverFunction:
  ...


@overload
def resolver_function(
    *,
    output_type: Optional[_TypeHint] = None,
    unwrap_dict_key: Optional[Union[str, Sequence[str]]] = None,
) -> Callable[..., ResolverFunction]:
  ...


def resolver_function(
    f: Optional[Callable[..., resolver_op.OpNode]] = None,
    *,
    output_type: Optional[_TypeHint] = None,
    unwrap_dict_key: Optional[Union[str, Sequence[str]]] = None,
):
  """Decorator for the resolver function.

  Args:
    f: Python function to decorate. See the usage at canned_resolver_function.py
    output_type: Optional static output type hint.
    unwrap_dict_key: If present, it will add loopable transform that unwraps
      dictionary key(s) so that `ForEach` captured value is a single channel, or
      a tuple of channels. This is only valid if the resolver function return
      value is ARTIFACT_MULTIMAP_LIST type.

  Returns:
    A ResolverFunction, or a decorator to create ResolverFunction.
  """
  if f is not None:
    return ResolverFunction(f)

  loopable_transform = None
  if unwrap_dict_key:
    if isinstance(unwrap_dict_key, str):
      key = cast(str, unwrap_dict_key)
      loopable_transform = lambda d: d[key]
    elif typing_utils.is_compatible(unwrap_dict_key, Sequence[str]):
      keys = cast(Sequence[str], unwrap_dict_key)
      loopable_transform = lambda d: tuple(d[key] for key in keys)
    else:
      raise ValueError(
          'Invalid unwrap_dict_key: Expected str or Sequence[str] but got '
          f'{unwrap_dict_key}'
      )

  def decorator(f):
    return ResolverFunction(
        f,
        output_type=output_type,
        loopable_transform=loopable_transform,
    )

  return decorator
