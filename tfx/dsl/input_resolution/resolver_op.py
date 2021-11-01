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
import enum
from typing import Any, ClassVar, Generic, Mapping, Type, TypeVar, Union, Sequence

import attr
import tfx.types
from tfx.utils import json_utils
from tfx.utils import typing_utils

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


class DataTypes(enum.Enum):
  """Supported data types for ResolverOps input/outputs."""
  ARTIFACT_LIST = Sequence[tfx.types.Artifact]
  ARTIFACT_MULTIMAP = typing_utils.ArtifactMultiMap
  ARTIFACT_MULTIMAP_LIST = Sequence[typing_utils.ArtifactMultiMap]

  def is_acceptable(self, value: Any) -> bool:
    """Check the value is instance of the data type."""
    if self == self.ARTIFACT_LIST:
      return typing_utils.is_homogeneous_artifact_list(value)
    elif self == self.ARTIFACT_MULTIMAP:
      return typing_utils.is_artifact_multimap(value)
    elif self == self.ARTIFACT_MULTIMAP_LIST:
      return typing_utils.is_list_of_artifact_multimap(value)
    raise NotImplementedError(f'Cannot check type for {self}.')


class _ResolverOpMeta(abc.ABCMeta):
  """Metaclass for ResolverOp.

  Features:
  1. ResolverOp cannot be instantiated directly. Calling ResolverOp() returns
     OpNode instance which is used for function tracing. Instead _ResolverOpMeta
     exposes create() classmethod to actually create a ResolverOp isntance.
  2. _ResolverOpMeta is aware of ResolverOpProperty, and keyword argument values
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
      arg_data_types: Sequence[DataTypes] = (DataTypes.ARTIFACT_MULTIMAP,),
      return_data_type: DataTypes = DataTypes.ARTIFACT_MULTIMAP):
    cls._props_by_name = {
        prop.name: prop
        for prop in attrs.values()
        if isinstance(prop, ResolverOpProperty)
    }
    if len(arg_data_types) != 1:
      raise NotImplementedError('len(arg_data_types) should be 1.')
    cls._arg_data_type = arg_data_types[0]
    cls._return_data_type = return_data_type
    super().__init__(name, bases, attrs)

  def __call__(cls, arg: 'OpNode', **kwargs: Any):
    """Fake instantiation of the ResolverOp class.

    Original implementation of metaclass.__call__ method is to instantiate
    (__new__) and initialize (__init__) the class instance. For ResolverOp(),
    we don't actually create the ResolverOp instance but returns an OpNode
    which represents a ResolverOp invocation.

    In order to actually create the ResolverOp instance, use ResolverOp.create()
    classmethod instead.

    Args:
      arg: Input argument for the operator.
      **kwargs: Property values for the ResolverOp.

    Returns:
      An OpNode instance that represents the operator call.
    """
    cls._check_arg(arg)
    cls._check_kwargs(kwargs)
    return OpNode(
        op_type=cls,
        arg=arg,
        output_data_type=cls._return_data_type,
        kwargs=kwargs)

  def _check_arg(cls, arg: 'OpNode'):
    if not isinstance(arg, OpNode):
      raise ValueError('Cannot directly call ResolverOp with real values. Use '
                       'output of another operator as an argument.')
    if arg.output_data_type != cls._arg_data_type:
      raise TypeError(f'{cls.__name__} takes {cls._arg_data_type.name} type '
                      f'but got {arg.output_data_type.name} instead.')

  def _check_kwargs(cls, kwargs: Mapping[str, Any]):
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

  def create_from_json(cls, config_json: str) -> 'ResolverOp':
    """Actually create a ResolverOp instance with JSON properties."""
    return cls.create(**json_utils.loads(config_json))


class _Empty:
  """Sentinel class for empty value (!= None)."""
  pass

_EMPTY = _Empty()
_T = TypeVar('_T', bound=json_utils.JsonableValue)


class ResolverOpProperty(Generic[_T]):
  """Property descriptor for ResolverOp.

  Usage:
    class FooOp(ResolverOp):
      foo = ResolverOpProperty(type=int, default=42)

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

  def _isinstance(self, value: Any, typ: Any) -> bool:
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

  Currently ResolverOp signature is limited to:
      (Dict[str, List[Artifact]]) -> Dict[str, List[Artifact]]
  but the constraints may be relaxed in the future.

  Usage:
    output_dict = FooOp(input_dict, foo=123)

  Note that output_dict in the example above is not an FooOp instance, but an
  OpNode representing the operator call Foo(input_dict, foo=123). In order to
  actually create the FooOp instance, use FooOp.create().
  """

  @abc.abstractmethod
  def apply(
      self,
      input_dict: typing_utils.ArtifactMultiMap,
  ) -> typing_utils.ArtifactMultiMap:
    """Implementation of the operator."""

  def set_context(self, context: Context):
    """Set Context to be used when applying the operator."""
    self.context = context


# Output type of an OpNode.
_TOut = TypeVar('_TOut')


@attr.s(kw_only=True, repr=False)
class OpNode(Generic[_TOut]):
  """OpNode represents a ResolverOp invocation."""
  _VALID_OP_TYPES = (ResolverOp,)
  # Singleton OpNode instance representing the input node.
  INPUT_NODE: ClassVar['OpNode']

  # ResolverOp class that is used for the Node.
  op_type = attr.ib()
  # Output data type of ResolverOp.
  output_data_type = attr.ib(default=DataTypes.ARTIFACT_MULTIMAP,
                             validator=attr.validators.instance_of(DataTypes))
  # A single argument to the ResolverOp.
  arg = attr.ib()
  # ResolverOpProperty for the ResolverOp, given as keyword arguments.
  kwargs = attr.ib(factory=dict)

  @classmethod
  def register_valid_op_type(cls, op_type: Type[Any]):
    if op_type not in cls._VALID_OP_TYPES:
      cls._VALID_OP_TYPES += (op_type,)

  @op_type.validator
  def validate_op_type(self, attribute, value):
    if not issubclass(value, self._VALID_OP_TYPES):
      raise TypeError(f'op_type should be subclass of {self._VALID_OP_TYPES} '
                      f'but got {value!r}.')

  @arg.validator
  def validate_arg(self, attribute, value):
    if not isinstance(value, OpNode):
      raise TypeError(f'Invalid arg: {value!r}.')

  def __repr__(self):
    if self.is_input_node:
      return 'INPUT_NODE'
    else:
      all_args = [repr(self.arg)]
      all_args.extend(f'{k}={repr(v)}' for k, v in self.kwargs.items())
      return f'{self.op_type.__qualname__}({", ".join(all_args)})'

  @property
  def is_input_node(self):
    return self is OpNode.INPUT_NODE

attr.set_run_validators(False)
OpNode.INPUT_NODE = OpNode(
    op_type=None,
    output_data_type=DataTypes.ARTIFACT_MULTIMAP,
    arg=None)
attr.set_run_validators(True)
