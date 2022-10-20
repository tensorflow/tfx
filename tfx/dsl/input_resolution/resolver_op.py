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
"""Module for ResolverOp and its related definitions."""
from __future__ import annotations

import abc
from typing import Any, Generic, Mapping, Type, TypeVar, Union, Sequence, Optional

import attr
from tfx import types
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import json_utils
from tfx.utils import typing_utils
import typing_extensions

import ml_metadata as mlmd


# Mark frozen as context instance may be used across multiple operator
# invocations.
@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class Context:
  """Context for running ResolverOp."""
  # MetadataStore for MLMD read access.
  store: mlmd.MetadataStore
  # TODO(jjong): Add more context such as current pipeline, current pipeline
  # run, and current running node information.


# Note that to use DataType as a generic type parameter (e.g.
# `Sequence[DataType]`) you need either `from __future__ import annotations`
# or quote the enum parameter (e.g. `Sequence['DataType']`).
# go/pytype-faq#annotating-with-a-proto-enum-type-caused-a-runtime-error
DataType = pipeline_pb2.InputGraph.DataType
_ValidDataType = typing_extensions.Literal[  # pytype: disable=invalid-annotation
    DataType.ARTIFACT_LIST,
    DataType.ARTIFACT_MULTIMAP,
    DataType.ARTIFACT_MULTIMAP_LIST,
]
# Actual data corresponding to ARTIFACT_LIST, ARTIFACT_MULTIMAP, and
# ARTIFACT_MULTIMAP_LIST.
_Data = Union[
    Sequence[types.Artifact],
    typing_utils.ArtifactMultiMap,
    Sequence[typing_utils.ArtifactMultiMap],
]


class _ResolverOpMeta(abc.ABCMeta):
  """Metaclass for ResolverOp.

  Features:
  1. ResolverOp cannot be instantiated directly. Calling ResolverOp() returns
     OpNode instance which is used for function tracing. Instead _ResolverOpMeta
     exposes create() classmethod to actually create a ResolverOp isntance.
  2. _ResolverOpMeta is aware of Property, and keyword argument values
     given from the ResolverOp invocation is validated without creating a
     ResolverOp instance.
  3. It inherits ABCMeta, so @abstractmethod works.
  """
  # Below pylint rule gives false alarm for the first argument of the metaclass
  # method (cls).
  # pylint: disable=no-value-for-parameter

  def __new__(cls, name, bases, attrs, **kwargs):
    # pylint: disable=too-many-function-args
    return super().__new__(cls, name, bases, attrs)

  def __init__(
      cls, name, bases, attrs,
      *,
      canonical_name: Optional[str] = None,
      arg_data_types: Sequence[DataType] = (DataType.ARTIFACT_MULTIMAP,),
      return_data_type: DataType = DataType.ARTIFACT_MULTIMAP):
    cls._props_by_name = {
        prop.name: prop
        for prop in attrs.values()
        if isinstance(prop, Property)
    }
    cls._canonical_name = canonical_name
    if not typing_utils.is_compatible(arg_data_types, Sequence[_ValidDataType]):
      raise ValueError(
          f'Invalid arg_data_types = {arg_data_types}. '
          'Expected Sequence[DataType].')
    cls._arg_data_types = arg_data_types
    if not typing_utils.is_compatible(return_data_type, _ValidDataType):
      raise ValueError(
          f'Invalid return_data_type = {return_data_type}. Expected DataType.')
    cls._return_data_type = return_data_type
    super().__init__(name, bases, attrs)

  @property
  def canonical_name(cls):
    return cls._canonical_name or cls.__name__

  def __call__(cls, *args: Union['Node', Mapping[str, 'Node']], **kwargs: Any):
    """Fake instantiation of the ResolverOp class.

    Original implementation of metaclass.__call__ method is to instantiate
    (__new__) and initialize (__init__) the class instance. For ResolverOp(),
    we don't actually create the ResolverOp instance but returns an OpNode
    which represents a ResolverOp invocation.

    In order to actually create the ResolverOp instance, use ResolverOp.create()
    classmethod instead.

    Args:
      *args: Input arguments for the operator.
      **kwargs: Property values for the ResolverOp.

    Returns:
      An OpNode instance that represents the operator call.
    """
    args = cls._check_and_transform_args(args)
    cls._check_kwargs(kwargs)
    return OpNode(
        op_type=cls,
        args=args,
        output_data_type=cls._return_data_type,
        kwargs=kwargs)

  def _check_and_transform_args(cls, args: Sequence[Any]) -> Sequence['Node']:
    """Static check against ResolverOp positional arguments."""
    if len(args) != len(cls._arg_data_types):
      raise ValueError(f'{cls.__name__} expects {len(cls._arg_data_types)} '
                       f'arguments but got {len(args)}.')
    transformed_args = []
    for arg, arg_data_type in zip(args, cls._arg_data_types):
      if (arg_data_type == DataType.ARTIFACT_MULTIMAP and
          isinstance(arg, dict)):
        arg = DictNode(arg)
      if not isinstance(arg, Node):
        raise ValueError('Cannot directly call ResolverOp with real values. '
                         'Use output of another operator as an argument.')
      if arg.output_data_type != arg_data_type:
        raise TypeError(
            f'{cls.__name__} takes {DataType.Name(arg_data_type)} type '
            f'but got {DataType.Name(arg.output_data_type)} instead.')
      transformed_args.append(arg)
    return transformed_args

  def _check_kwargs(cls, kwargs: Mapping[str, Any]):
    """Static check against ResolverOp keyword arguments."""
    for name, prop in cls._props_by_name.items():
      if prop.required and name not in kwargs:
        raise ValueError(f'Required property {name} is missing.')
    for name, value in kwargs.items():
      if name not in cls._props_by_name:
        raise KeyError(f'Unknown property {name}.')
      prop = cls._props_by_name[name]
      prop.validate(value)

  def create(cls, **props: Any) -> 'ResolverOp':
    """Actually create a ResolverOp instance.

    Note: Normal class call (e.g. MyResolver()) does not create a MyResolver
    instance but an OpNode(op_type=MyResolver). This classmethod is a hidden
    way of creating an actual MyResolver instance.

    Args:
      **props: Property values for the ResolverOp.
    Returns:
      A ResolverOp instance.
    """
    real_instance = super().__call__()
    cls._check_kwargs(props)
    for name, value in props.items():
      setattr(real_instance, name, value)
    return real_instance


class _Empty:
  """Sentinel class for empty value (!= None)."""
  pass

_EMPTY = _Empty()
_T = TypeVar('_T', bound=json_utils.JsonableType)


class Property(Generic[_T]):
  """Property descriptor for ResolverOp.

  Usage:
    class FooOp(ResolverOp):
      foo = Property(type=int, default=42)

      def apply(self, input_dict):
        whatever(self.foo, input_dict)  # Can be accessed from self.
        ...

    @resolver_stem
    def my_resolver_stem(input_dict):
      result = FooOp(input_dict, foo=123)
  """

  def __init__(
      self, *,
      type: Type[_T],  # pylint: disable=redefined-builtin
      default: Union[_T, _Empty] = _EMPTY):
    self._type = type
    self._required = default is _EMPTY
    if default is not _EMPTY and not typing_utils.is_compatible(default, type):
      raise TypeError(f'Default value {default!r} is not {type} type.')
    self._default = default
    self._name = ''  # Will be filled by __set_name__.
    self._private_name = ''  # Will be filled by __set_name__.

  @property
  def name(self) -> str:
    return self._name

  @property
  def type(self) -> Type[_T]:
    return self._type

  @property
  def required(self) -> bool:
    return self._required

  def validate(self, value: Any):
    if not typing_utils.is_compatible(value, self._type):
      raise TypeError(f'{self._name} should be {self._type} but got {value!r}.')

  def __set_name__(self, owner, name):
    if name == 'context':
      raise NameError(
          'Property name "context" is reserved. Please use other name.')
    self._name = name
    self._private_name = f'_prop_{name}'

  def __set__(self, obj, value):
    self.validate(value)
    setattr(obj, self._private_name, value)

  def __get__(self, obj, objtype=None) -> _T:
    if hasattr(obj, self._private_name):
      return getattr(obj, self._private_name)
    elif self._required:
      raise ValueError(f'Required property {self._name} not set.')
    else:
      return self._default  # pytype: disable=bad-return-type


class ResolverOp(metaclass=_ResolverOpMeta):
  """ResolverOp is the building block of input resolution logic.

  Usage:
    output_dict = FooOp(input_dict, foo=123)

  Note that output_dict in the example above is not an FooOp instance, but an
  OpNode representing the operator call Foo(input_dict, foo=123). In order to
  actually create the FooOp instance, use FooOp.create().
  """

  def __init__(self, *unused_args, **unused_kwargs):
    """Dummy constructor to bypass false negative pytype alarm."""

  @abc.abstractmethod
  def apply(self, *args: _Data) -> _Data:
    """Implementation of the operator."""

  def set_context(self, context: Context):
    """Set Context to be used when applying the operator."""
    self.context = context


class Node:
  output_data_type: DataType

  def __eq__(self, other):
    if not isinstance(other, Node):
      return NotImplemented
    return self is other

  def __hash__(self):
    return hash(id(self))


@attr.s(kw_only=True, repr=False, eq=False)
class OpNode(Node):
  """Node that represents a ResolverOp invocation and its result."""
  # ResolverOp class that is used for the Node.
  op_type = attr.ib()
  # Output data type of ResolverOp.
  output_data_type = attr.ib(
      type=DataType,
      default=DataType.ARTIFACT_MULTIMAP)
  # Arguments of the ResolverOp.
  args = attr.ib(type=Sequence[Node], default=())
  # Property for the ResolverOp, given as keyword arguments.
  kwargs = attr.ib(type=Mapping[str, Any], factory=dict)

  @args.validator
  def _validate_args(self, attribute, value):
    del attribute  # Unused.
    if not typing_utils.is_compatible(value, Sequence[Node]):
      raise TypeError(f'`args` should be a Sequence[Node] but got {value!r}.')

  def __repr__(self):
    all_args = [repr(arg) for arg in self.args]
    all_args.extend(f'{k}={repr(v)}' for k, v in self.kwargs.items())
    return f'{self.op_type.__qualname__}({", ".join(all_args)})'


class InputNode(Node):
  """Node that represents the input arguments of the resolver function."""

  def __init__(self, wrapped: Any, output_data_type: DataType):
    # TODO(b/236140795): Allow only BaseChannel as a wrapped and make
    # output_data_type always be the ARTIFACT_LIST.
    self.wrapped = wrapped
    self.output_data_type = output_data_type

  def __repr__(self) -> str:
    return 'Input()'

  def __eq__(self, others):
    if not isinstance(others, InputNode):
      return NotImplemented
    return self.wrapped == others.wrapped

  def __hash__(self):
    if isinstance(self.wrapped, dict):
      return hash(tuple(sorted(self.wrapped.items())))
    return hash(self.wrapped)


class DictNode(Node):
  """Node that represents a dict of Node values."""
  output_data_type = DataType.ARTIFACT_MULTIMAP

  def __init__(self, nodes: Mapping[str, Node]):
    if not typing_utils.is_compatible(nodes, Mapping[str, Node]) or any(
        v.output_data_type != DataType.ARTIFACT_LIST for v in nodes.values()):
      raise ValueError(
          'Expected dict[str, Node] s.t. all node.output_data_type == '
          f'ARTIFACT_LIST, but got {nodes}.')
    self.nodes = nodes

  def __eq__(self, other):
    if not isinstance(other, DictNode):
      return NotImplemented
    return self.nodes == other.nodes

  def __hash__(self):
    return hash(tuple(sorted(self.nodes.items())))

  def __repr__(self) -> str:
    args = [f'{k}={v!r}' for k, v in self.nodes.items()]
    return f'Dict({", ".join(args)})'
