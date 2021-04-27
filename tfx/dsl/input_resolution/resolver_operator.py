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
import abc
from typing import Dict, List, Any, Sequence, Mapping, Generic, TypeVar, Type, Union, ClassVar

import attr

import tfx.types
from tfx.utils import json_utils


# TODO(jjong): Consider interoperability with Mapping[str, Sequence[Artifact]].
ArtifactMultimap = Dict[str, List[tfx.types.Artifact]]


class _ResolverOpMeta(abc.ABCMeta):
  """Metaclass for ResolverOp.

  Features:
  1. ResolverOp cannot be instantiated directly. Calling ResolverOp() returns
     OpNode instance which is used for function tracing. Instead _ResolverOpMeta
     exposes create() classmethod to actually create a ResolverOp isntance.
  2. _ResolverOpMeta is aware of ResolverOpProperty, and keyword argument values
     given from the ResolverOp invocation is validated without creating an
     ResolverOp instance.
  3. It inherits ABCMeta, so @abstractmethod works.
  """
  # Below pylint rule gives false alarm for the first argument of the metaclass
  # method (cls).
  # pylint: disable=no-value-for-parameter

  def __init__(cls, name, bases, attrs):
    cls._props_by_name = {
        prop.name: prop
        for prop in attrs.values()
        if isinstance(prop, ResolverOpProperty)
    }
    super().__init__(name, bases, attrs)

  def __call__(cls, *args: 'OpNode', **kwargs: Any):
    cls._check_args(args)
    cls._check_kwargs(kwargs)
    return OpNode(
        op_type=cls,
        args=args,
        kwargs=kwargs)

  def _check_args(cls, args: Sequence['OpNode']):
    if len(args) != 1:
      raise ValueError('ResolverOp takes one positional argument.')
    if not isinstance(args[0], OpNode):
      raise ValueError('Cannot directly call ResolverOp with real values. Use '
                       'output of other operator as an argument.')

  def _check_kwargs(cls, kwargs: Mapping[str, Any]):
    for name, prop in cls._props_by_name.items():
      if prop.required and name not in kwargs:
        raise ValueError(f'Required property {name} is missing.')
    for name, value in kwargs.items():
      if name not in cls._props_by_name:
        raise KeyError(f'Unknown property {name}.')
      prop = cls._props_by_name[name]
      prop.validate(value)

  def create(cls, **props: Any):
    result = super().__call__()
    cls._check_kwargs(props)
    for name, value in props.items():
      setattr(result, name, value)
    return result

  def create_from_json(cls, config_json: str):
    return cls.create(**json_utils.loads(config_json))


class _Empty:
  """Sentinel class for empty value (!= None)."""
  pass

_EMPTY = _Empty()
_T = TypeVar('_T')


class ResolverOpProperty(Generic[_T]):
  """Property descriptor for ResolverOp.

  Usage:
    class FooOp(ResolverOp):
      foo = ResolverOpProperty(type=int, default=42)

      def apply(self, input_dict):
        whatever(self.foo, input_dict)  # Can be accessed from self.
        ...

    @resolver
    def MyResolver(input_dict):
      result = FooOp(input_dict, foo=123)
  """

  def __init__(
      self, *,
      type: Type[_T],  # pylint: disable=redefined-builtin
      default: Union[_T, _Empty] = _EMPTY):
    self._type = type
    self._required = default is _EMPTY
    if default is not _EMPTY and not self._isinstance(default, type):
      raise TypeError(f'Default value {default!r} is not {type} type.')
    self._default = default

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
    if not self._isinstance(value, self._type):
      raise TypeError(f'{self._name} should be {self._type} but got {value!r}.')

  def _isinstance(self, value: Any, typ: Any):
    """Custom isinstance() that supports Generic types as well."""
    typ_args = getattr(typ, '__args__', ())
    if hasattr(typ, '__origin__'):
      # Drop subscripted extra type parameters from generic type.
      # (e.g. Dict[str, str].__origin__ == dict)
      # See https://www.python.org/dev/peps/pep-0585 for more information.
      typ = typ.__origin__
    if typ == Union:
      return any(self._isinstance(value, t) for t in typ_args)
    else:
      return isinstance(value, typ)

  def __set_name__(self, owner, name):
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
      return self._default


class ResolverOp(metaclass=_ResolverOpMeta):
  """ResolverOp is the building block of input resolution logic.

  Currently ResolverOp signature is limited to:
      (ArtifactMultimap) -> ArtifactMultimap
  but the constraints may be relaxed in the future.

  Usage:
    output_dict = FooOp(input_dict, foo=123)

  Note that output_dict in the example above is not an FooOp instance, but an
  OpNode representing the operator call Foo(input_dict, foo=123). In order to
  actually create the FooOp instance, use FooOp.create().
  """

  @abc.abstractmethod
  def apply(self, input_dict: ArtifactMultimap) -> ArtifactMultimap:
    """Implementation of the operator."""


@attr.s(kw_only=True, repr=False)
class OpNode:
  """Node representation for ResolverOp invocation."""
  # Singleton OpNode instance representing the input node.
  INPUT_NODE: ClassVar['OpNode']

  # ResolverOp class that is used for the Node.
  op_type = attr.ib()
  # Arguments to the ResolverOp.
  args = attr.ib(factory=tuple)
  # OpProperty for the ResolverOp, given as a keyword arguments.
  kwargs = attr.ib(factory=dict)

  @op_type.validator
  def validate_op_type(self, attribute, value):
    if not issubclass(value, ResolverOp):
      raise TypeError(f'op_type {value} is not a ResolverOp.')

  @args.validator
  def validate_args(self, attribute, value):
    if not isinstance(value, Sequence):
      raise TypeError(f'Invalid args: {value!r}.')
    for v in value:
      if not isinstance(v, OpNode):
        raise TypeError(f'Invalid args: {value!r}.')

  def __repr__(self):
    if self.is_input_node:
      return 'INPUT_NODE'
    else:
      all_args = [repr(arg) for arg in self.args]
      all_args.extend(f'{k}={repr(v)}' for k, v in self.kwargs.items())
      return f'{self.op_type.__qualname__}({", ".join(all_args)})'

  @property
  def is_input_node(self):
    return self is OpNode.INPUT_NODE

attr.set_run_validators(False)
OpNode.INPUT_NODE = OpNode(op_type=None)
attr.set_run_validators(True)
