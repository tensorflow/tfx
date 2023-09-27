# Copyright 2020 Google LLC. All Rights Reserved.
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
"""Placeholders represent not-yet-available values at the component authoring time.
"""

import abc
import copy
import enum
import functools
from typing import Any, Iterator, List, Optional, Sequence, Type, TypeVar, Union, cast

import attr
from tfx.proto.orchestration import placeholder_pb2
from tfx.utils import proto_utils

from google.protobuf import message

# To resolve circular dependency caused by type annotations.
# TODO(b/191610358): Reduce the number of circular type-dependencies.
types = Any  # tfx.types imports channel.py, which in turn imports this module.

# TODO(b/190409099): Support RuntimeParameter.
_ValueLikeType = Union[int, float, str, 'Placeholder']


class _PlaceholderOperator(abc.ABC):
  """An Operator performs an operation on a Placeholder.

  It knows how to encode itself into a proto.
  """

  def __init__(self):
    pass

  @abc.abstractmethod
  def encode(
      self,
      sub_expression_pb: placeholder_pb2.PlaceholderExpression,
      component_spec: Optional[Type['types.ComponentSpec']] = None
  ) -> placeholder_pb2.PlaceholderExpression:
    pass

  def traverse(self) -> Iterator['Placeholder']:
    """Yields all placeholders under this operator."""
    yield from ()  # Empty generator function.


class _ArtifactUriOperator(_PlaceholderOperator):
  """Artifact URI Operator extracts the URI from an artifact Placeholder.

  Prefer to use the .uri property of ArtifactPlaceholder.
  """

  def __init__(self, split: str = ''):
    super().__init__()
    self._split = split

  def encode(
      self,
      sub_expression_pb: placeholder_pb2.PlaceholderExpression,
      component_spec: Optional[Type['types.ComponentSpec']] = None
  ) -> placeholder_pb2.PlaceholderExpression:
    del component_spec  # Unused by ArtifactUriOperator

    result = placeholder_pb2.PlaceholderExpression()
    result.operator.artifact_uri_op.expression.CopyFrom(sub_expression_pb)
    if self._split:
      result.operator.artifact_uri_op.split = self._split
    return result


class _ArtifactValueOperator(_PlaceholderOperator):
  """Artifact Value Operator extracts the value from a primitive artifact Placeholder.

  Prefer to use the .value property of ArtifactPlaceholder.
  """

  def encode(
      self,
      sub_expression_pb: placeholder_pb2.PlaceholderExpression,
      component_spec: Optional[Type['types.ComponentSpec']] = None
  ) -> placeholder_pb2.PlaceholderExpression:
    del component_spec  # Unused by ArtifactValueOperator

    result = placeholder_pb2.PlaceholderExpression()
    result.operator.artifact_value_op.expression.CopyFrom(sub_expression_pb)
    return result


class _IndexOperator(_PlaceholderOperator):
  """Index Operator extracts value at the given index of a Placeholder.

  Prefer to use [index] operator overloading of Placeholder.
  """

  def __init__(self, index: Union[int, str]):
    super().__init__()
    self._index = index

  def encode(
      self,
      sub_expression_pb: placeholder_pb2.PlaceholderExpression,
      component_spec: Optional[Type['types.ComponentSpec']] = None
  ) -> placeholder_pb2.PlaceholderExpression:
    del component_spec  # Unused by IndexOperator

    result = placeholder_pb2.PlaceholderExpression()
    result.operator.index_op.expression.CopyFrom(sub_expression_pb)
    if isinstance(self._index, int):
      result.operator.index_op.index = self._index
    if isinstance(self._index, str):
      result.operator.index_op.key = self._index
    return result


class _PropertyOperator(_PlaceholderOperator):
  """Property Operator gets the property of an artifact Placeholder.

  Prefer to use .property(key) method of Artifact Placeholder.
  """

  def __init__(self, key: str, is_custom_property: bool = False):
    super().__init__()
    self._key = key
    self._is_custom_property = is_custom_property

  def encode(
      self,
      sub_expression_pb: placeholder_pb2.PlaceholderExpression,
      component_spec: Optional[Type['types.ComponentSpec']] = None
  ) -> placeholder_pb2.PlaceholderExpression:
    del component_spec  # Unused by PropertyOperator

    result = placeholder_pb2.PlaceholderExpression()
    result.operator.artifact_property_op.expression.CopyFrom(sub_expression_pb)
    result.operator.artifact_property_op.key = self._key
    result.operator.artifact_property_op.is_custom_property = (
        self._is_custom_property)
    return result


class _ConcatOperator(_PlaceholderOperator):
  """Concat Operator concatenates multiple Placeholders.

  Prefer to use + operator overloading of Placeholder.
  """

  def __init__(self,
               right: Optional[Union[str, 'Placeholder']] = None,
               left: Optional[str] = None):
    super().__init__()
    self._left = left
    self._right = right

  def encode(
      self,
      sub_expression_pb: placeholder_pb2.PlaceholderExpression,
      component_spec: Optional[Type['types.ComponentSpec']] = None
  ) -> placeholder_pb2.PlaceholderExpression:
    del component_spec  # Unused by ConcatOperator

    # ConcatOperator's proto version contains multiple placeholder expressions
    # as operands. For convenience, the Python version is implemented taking
    # only two operands.
    if self._right:
      # Resolve other expression
      if isinstance(self._right, Placeholder):
        other_expression = cast(Placeholder, self._right)
        other_expression_pb = other_expression.encode()
      else:
        other_expression_pb = placeholder_pb2.PlaceholderExpression()
        other_expression_pb.value.string_value = self._right

      # Try combining with existing concat operator
      if sub_expression_pb.HasField(
          'operator') and sub_expression_pb.operator.HasField('concat_op'):
        sub_expression_pb.operator.concat_op.expressions.append(
            other_expression_pb)
        return sub_expression_pb
      else:
        result = placeholder_pb2.PlaceholderExpression()
        result.operator.concat_op.expressions.extend(
            [sub_expression_pb, other_expression_pb])
        return result

    if self._left:
      # Resolve other expression: left operand must be str
      other_expression_pb = placeholder_pb2.PlaceholderExpression()
      other_expression_pb.value.string_value = self._left

      # Try combining with existing concat operator
      if sub_expression_pb.HasField(
          'operator') and sub_expression_pb.operator.HasField('concat_op'):
        sub_expression_pb.operator.concat_op.expressions.insert(
            0, other_expression_pb)
        return sub_expression_pb
      else:
        result = placeholder_pb2.PlaceholderExpression()
        result.operator.concat_op.expressions.extend(
            [other_expression_pb, sub_expression_pb])
        return result

    raise RuntimeError(
        'ConcatOperator does not have the other expression to concat.')

  def traverse(self) -> Iterator['Placeholder']:
    """Yields all placeholders under this operator."""
    if isinstance(self._right, Placeholder):
      yield from self._right.traverse()


class ProtoSerializationFormat(enum.Enum):
  TEXT_FORMAT = placeholder_pb2.ProtoOperator.TEXT_FORMAT
  JSON = placeholder_pb2.ProtoOperator.JSON
  BINARY = placeholder_pb2.ProtoOperator.BINARY
  INLINE_FILE_TEXT_FORMAT = placeholder_pb2.ProtoOperator.INLINE_FILE_TEXT_FORMAT


class ListSerializationFormat(enum.Enum):
  JSON = placeholder_pb2.ListSerializationOperator.JSON
  COMMA_SEPARATED_STR = placeholder_pb2.ListSerializationOperator.COMMA_SEPARATED_STR


class _ProtoOperator(_PlaceholderOperator):
  """Proto Operator helps access/serialze a proto-valued placeholder.

  Prefer to use . operator overloading of ExecPropertyPlaceholder or
  RuntimeInfoPlaceholder for proto field access, use serialize_proto function
  for proto serialization.
  """

  def __init__(self,
               proto_field_path: Optional[str] = None,
               serialization_format: Optional[ProtoSerializationFormat] = None):
    super().__init__()
    self._proto_field_path = [proto_field_path] if proto_field_path else None
    self._serialization_format = serialization_format

  def can_append_field_path(self):
    return self._proto_field_path is not None

  def append_field_path(self, extra_path: str):
    if self._proto_field_path is None:
      raise ValueError('Proto field path not initialized.')
    self._proto_field_path.append(extra_path)

  def encode(
      self,
      sub_expression_pb: placeholder_pb2.PlaceholderExpression,
      component_spec: Optional[Type['types.ComponentSpec']] = None
  ) -> placeholder_pb2.PlaceholderExpression:
    result = placeholder_pb2.PlaceholderExpression()
    result.operator.proto_op.expression.CopyFrom(sub_expression_pb)

    if self._proto_field_path:
      result.operator.proto_op.proto_field_path.extend(self._proto_field_path)
    if self._serialization_format:
      result.operator.proto_op.serialization_format = (
          self._serialization_format.value)

    # Attach proto descriptor if available through component spec.
    if (component_spec and sub_expression_pb.placeholder.type
        == placeholder_pb2.Placeholder.EXEC_PROPERTY):
      exec_property_name = sub_expression_pb.placeholder.key
      if exec_property_name not in component_spec.PARAMETERS:
        raise ValueError(
            f"Can't find provided placeholder key {exec_property_name} in "
            "component spec's exec properties. "
            f'Available exec property keys: {component_spec.PARAMETERS.keys()}.'
        )
      execution_param = component_spec.PARAMETERS[exec_property_name]
      if not issubclass(execution_param.type, message.Message):
        raise ValueError(
            "Can't apply placeholder proto operator on non-proto type "
            f'exec property. Got {execution_param.type}.')
      proto_schema = result.operator.proto_op.proto_schema
      proto_schema.message_type = execution_param.type.DESCRIPTOR.full_name
      proto_utils.build_file_descriptor_set(execution_param.type,
                                            proto_schema.file_descriptors)

    return result


class _ListSerializationOperator(_PlaceholderOperator):
  """ListSerializationOperator serializes list type placeholder.

  Prefer to use the .serialize_list property of ExecPropertyPlaceholder or
  ListPlaceholder.
  """

  def __init__(self, serialization_format: ListSerializationFormat):
    super().__init__()
    self._serialization_format = serialization_format

  def encode(
      self,
      sub_expression_pb: placeholder_pb2.PlaceholderExpression,
      component_spec: Optional[Type['types.ComponentSpec']] = None
  ) -> placeholder_pb2.PlaceholderExpression:
    del component_spec

    result = placeholder_pb2.PlaceholderExpression()
    result.operator.list_serialization_op.expression.CopyFrom(sub_expression_pb)
    result.operator.list_serialization_op.serialization_format = self._serialization_format.value
    return result


class _Base64EncodeOperator(_PlaceholderOperator):
  """Base64EncodeOperator encodes another placeholder using url safe base64.

  Prefer to use the .b64encode method of Placeholder.
  """

  def encode(
      self,
      sub_expression_pb: placeholder_pb2.PlaceholderExpression,
      component_spec: Optional[Type['types.ComponentSpec']] = None
  ) -> placeholder_pb2.PlaceholderExpression:
    del component_spec  # Unused by B64EncodeOperator

    result = placeholder_pb2.PlaceholderExpression()
    result.operator.base64_encode_op.expression.CopyFrom(sub_expression_pb)
    return result


# To ensure that parent class operations on a child class still returns the
# child class instance.
_T = TypeVar('_T')


class Placeholder:
  """A Placeholder represents not-yet-available values at the component authoring time.
  """

  def __init__(self,
               placeholder_type: placeholder_pb2.Placeholder.Type,
               key: Optional[str] = None):
    self._operators = []
    self._type = placeholder_type
    self._key = key  # TODO(b/217597892): Refactor _key as read-only property.

  def _clone_and_use_operators(self: _T,
                               operators: List[_PlaceholderOperator]) -> _T:
    copied = copy.deepcopy(self)
    for op in operators:
      copied._operators.append(op)  # pylint: disable=protected-access
    return copied

  def __add__(self: _T, right: Union[str, 'Placeholder']) -> _T:
    return self._clone_and_use_operators([_ConcatOperator(right=right)])

  def __radd__(self: _T, left: str) -> _T:
    return self._clone_and_use_operators([_ConcatOperator(left=left)])

  def __deepcopy__(self, memo):
    # This method is implemented to make sure Placeholder is deep copyable
    # by copy.deepcopy().
    cls = self.__class__
    result = cls.__new__(cls)
    memo[id(self)] = result
    for k, v in self.__dict__.items():
      setattr(result, k, copy.deepcopy(v, memo))
    return result

  def b64encode(self: _T) -> _T:
    """Encodes the output of another placeholder using url safe base64 encoding.

    Returns:
      A placeholder, when rendering, is a url safe base64 encoded string.
    """
    return self._clone_and_use_operators([_Base64EncodeOperator()])

  def encode(
      self,
      component_spec: Optional[Type['types.ComponentSpec']] = None
  ) -> placeholder_pb2.PlaceholderExpression:
    """Encodes a placeholder as PlaceholderExpression proto.

    Args:
      component_spec: Optional. Information about the component that may be
        needed during encoding.

    Returns:
      Encoded proto containing all information of this placeholder.
    """
    result = placeholder_pb2.PlaceholderExpression()
    result.placeholder.type = self._type
    if self._key:
      result.placeholder.key = self._key
    for op in self._operators:
      result = op.encode(result, component_spec)
    return result

  def traverse(self) -> Iterator['Placeholder']:
    """Yields all placeholders under and including this one."""
    yield self
    for op in self._operators:
      yield from op.traverse()


def join(placeholders: Sequence[Union[str, Placeholder]],
         separator: str = '') -> Union[str, Placeholder]:
  """Joins a list consisting of placeholders and strings using separator.

  Returns an empty string if placeholders is empty.

  Args:
    placeholders: List of placeholders and/or strings.
    separator: The separator to use when joining the passed in values.

  Returns:
    A Placeholder representing the concatenation of all elements passed in, or
    a string in the case that no element was a Placeholder instance.
  """
  if not placeholders:
    return ''

  def joiner(a, b):
    if separator:
      return a + separator + b
    return a + b

  return functools.reduce(joiner, placeholders)


class ArtifactPlaceholder(Placeholder):
  """Artifact Placeholder represents an input or an output artifact.

  Prefer to use input(...) or output(...) to create artifact placeholders.
  """

  @property
  def is_input(self) -> bool:
    return self._type == placeholder_pb2.Placeholder.Type.INPUT_ARTIFACT

  @property
  def is_output(self) -> bool:
    return self._type == placeholder_pb2.Placeholder.Type.OUTPUT_ARTIFACT

  @property
  def key(self) -> Optional[str]:
    return self._key

  @property
  def uri(self: _T) -> _T:
    return self._clone_and_use_operators(
        [*self._optional_index_operator(),
         _ArtifactUriOperator()])

  def split_uri(self: _T, split: str) -> _T:
    return self._clone_and_use_operators(
        [*self._optional_index_operator(),
         _ArtifactUriOperator(split)])

  @property
  def value(self: _T) -> _T:
    return self._clone_and_use_operators(
        [*self._optional_index_operator(),
         _ArtifactValueOperator()])

  def __getitem__(self: _T, key: Union[int, str]) -> _T:
    return self._clone_and_use_operators([_IndexOperator(key)])

  def property(self: _T, key: str) -> _T:
    return self._clone_and_use_operators(
        [*self._optional_index_operator(),
         _PropertyOperator(key)])

  def custom_property(self: _T, key: str) -> _T:
    return self._clone_and_use_operators([
        *self._optional_index_operator(),
        _PropertyOperator(key, is_custom_property=True)
    ])

  def _optional_index_operator(self) -> List[_PlaceholderOperator]:
    if any(isinstance(op, _IndexOperator) for op in self._operators):
      return []
    else:
      return [_IndexOperator(0)]


class _ProtoAccessiblePlaceholder(Placeholder, abc.ABC):
  """A base Placeholder for accessing proto fields using Python proto syntax."""

  def __getattr__(self: _T, field_name: str) -> _T:
    proto_access_field = f'.{field_name}'
    if self._operators and isinstance(
        self._operators[-1],
        _ProtoOperator) and self._operators[-1].can_append_field_path():
      result = self._clone_and_use_operators([])  # makes a copy of self
      result._operators[-1].append_field_path(proto_access_field)
      return result
    else:
      return self._clone_and_use_operators(
          [_ProtoOperator(proto_field_path=proto_access_field)])

  def __getitem__(self: _T, key: Union[int, str]) -> _T:
    if isinstance(key, int):
      return self._clone_and_use_operators([_IndexOperator(key)])
    else:
      proto_access_field = f'[{key!r}]'
      if self._operators and isinstance(
          self._operators[-1],
          _ProtoOperator) and self._operators[-1].can_append_field_path():
        result = self._clone_and_use_operators([])  # makes a copy of self
        result._operators[-1].append_field_path(proto_access_field)
        return result
      else:
        return self._clone_and_use_operators(
            [_ProtoOperator(proto_field_path=proto_access_field)])

  def serialize(self: _T, serialization_format: ProtoSerializationFormat) -> _T:
    """Serialize the proto-valued placeholder using the provided scheme.

    Args:
      serialization_format: The format of how the proto is serialized.

    Returns:
      A placeholder that when rendered is serialized with the scheme.
    """
    return self._clone_and_use_operators(
        [_ProtoOperator(serialization_format=serialization_format)])


class ExecPropertyPlaceholder(_ProtoAccessiblePlaceholder):
  """ExecProperty Placeholder represents an execution property.

  Prefer to use exec_property(...) to create exec property placeholders.
  """

  def __init__(self, key: str):
    super().__init__(placeholder_pb2.Placeholder.Type.EXEC_PROPERTY, key)

  def serialize_list(self: _T,
                     serialization_format: ListSerializationFormat) -> _T:
    """Serializes list-value placeholder to JSON or comma-separated string.

    Here list value includes repeated proto field. This function only
    supports primitive type list element (a.k.a bool, int, float or str) at the
    moment; throws runtime error otherwise.

    Args:
       serialization_format: The format of how the proto is serialized.

    Returns:
      A placeholder.
    """
    return self._clone_and_use_operators(
        [_ListSerializationOperator(serialization_format)])


class RuntimeInfoPlaceholder(_ProtoAccessiblePlaceholder):
  """RuntimeInfo Placeholder represents runtime information for a component.

  Prefer to use runtime_info(...) to create RuntimeInfo placeholders.
  """

  def __init__(self, key: str):
    if key not in _RUNTIME_INFO_KEYS:
      raise ValueError(f'Got unsupported runtime info key: {key}.')
    super().__init__(placeholder_pb2.Placeholder.Type.RUNTIME_INFO, key)


class ExecInvocationPlaceholder(_ProtoAccessiblePlaceholder):
  """Execution Invocation Placeholder helps access ExecutionInvocation proto.

  Prefer to use execution_invocation(...) to create Execution Invocation
  placeholder.
  """

  def __init__(self):
    super().__init__(placeholder_pb2.Placeholder.Type.EXEC_INVOCATION)


class EnvironmentVariablePlaceholder(Placeholder):
  """Environment Variable Placeholder helps access EnvironmentVariable proto.

  Prefer to use environment_variable(...) to create Environment Variable
  placeholder.
  """

  def __init__(self, key: str):
    super().__init__(placeholder_pb2.Placeholder.Type.ENVIRONMENT_VARIABLE, key)


class _CompareOp(enum.Enum):
  """An alias for placeholder_pb2.ComparisonOperator.Operation."""

  EQUAL = placeholder_pb2.ComparisonOperator.Operation.EQUAL
  LESS_THAN = placeholder_pb2.ComparisonOperator.Operation.LESS_THAN
  GREATER_THAN = placeholder_pb2.ComparisonOperator.Operation.GREATER_THAN


class ChannelWrappedPlaceholder(ArtifactPlaceholder):
  """Wraps a Channel in a Placeholder.

  This allows it to make Predicates using syntax like:
    channel.future().value > 5
  """

  def __init__(self, channel: 'types.Channel'):
    super().__init__(placeholder_pb2.Placeholder.Type.INPUT_ARTIFACT)
    self.channel = channel

  def set_key(self, key: Optional[str]):
    self._key = key

  def _clone_and_use_operators(self: _T,
                               operators: List[_PlaceholderOperator]) -> _T:
    # Avoid copying the wrapped channel, because
    # 1. The channel's reference is used during compilation.
    # 2. We only need to copy Placeholder related objects.
    copied = copy.deepcopy(self, {id(self.channel): self.channel})
    for op in operators:
      copied._operators.append(op)  # pylint: disable=protected-access
    return copied

  def __eq__(self, other: _ValueLikeType) -> 'Predicate':
    return Predicate(_Comparison(_CompareOp.EQUAL, left=self, right=other))

  def __ne__(self, other: _ValueLikeType) -> 'Predicate':
    return logical_not(self == other)

  def __lt__(self, other: _ValueLikeType) -> 'Predicate':
    return Predicate(_Comparison(_CompareOp.LESS_THAN, left=self, right=other))

  def __le__(self, other: _ValueLikeType) -> 'Predicate':
    return logical_not(self > other)

  def __gt__(self, other: _ValueLikeType) -> 'Predicate':
    return Predicate(
        _Comparison(_CompareOp.GREATER_THAN, left=self, right=other))

  def __ge__(self, other: _ValueLikeType) -> 'Predicate':
    return logical_not(self < other)


def _encode_value_like(
    x: _ValueLikeType,
    component_spec: Optional[Type['types.ComponentSpec']] = None,
) -> placeholder_pb2.PlaceholderExpression:
  """Encodes x to a placeholder expression proto."""

  if isinstance(x, Placeholder):
    return x.encode(component_spec)
  result = placeholder_pb2.PlaceholderExpression()
  if isinstance(x, int):
    result.value.int_value = x
  elif isinstance(x, float):
    result.value.double_value = x
  elif isinstance(x, str):
    result.value.string_value = x
  else:
    raise ValueError(f'x must be an int, float, str, or Placeholder. x: {x}')
  return result


_PredicateSubtype = Union['_Comparison', '_NotExpression',
                          '_BinaryLogicalExpression']


@attr.s
class _Comparison:
  """Represents a comparison between two placeholders."""

  compare_op = attr.ib(type=_CompareOp)
  left = attr.ib(type=_ValueLikeType)
  right = attr.ib(type=_ValueLikeType)

  def encode(
      self, component_spec: Optional[Type['types.ComponentSpec']] = None
  ) -> placeholder_pb2.PlaceholderExpression:
    result = placeholder_pb2.PlaceholderExpression()
    result.operator.compare_op.op = self.compare_op.value
    result.operator.compare_op.lhs.CopyFrom(
        _encode_value_like(self.left, component_spec)
    )
    result.operator.compare_op.rhs.CopyFrom(
        _encode_value_like(self.right, component_spec)
    )
    return result

  def traverse(self) -> Iterator[Placeholder]:
    """Yields all placeholders under this predicate."""
    if isinstance(self.left, Placeholder):
      yield from self.left.traverse()
    if isinstance(self.right, Placeholder):
      yield from self.right.traverse()


class _LogicalOp(enum.Enum):
  """An alias for logical operation enums in placeholder.proto."""

  NOT = placeholder_pb2.UnaryLogicalOperator.Operation.NOT
  AND = placeholder_pb2.BinaryLogicalOperator.Operation.AND
  OR = placeholder_pb2.BinaryLogicalOperator.Operation.OR


@attr.s
class _NotExpression:
  """Represents a logical negation."""

  pred_dataclass = attr.ib(type=_PredicateSubtype)

  def encode(
      self, component_spec: Optional[Type['types.ComponentSpec']] = None
  ) -> placeholder_pb2.PlaceholderExpression:
    pred_pb = self.pred_dataclass.encode(component_spec)
    # not(not(a)) becomes a
    if isinstance(self.pred_dataclass, _NotExpression):
      return pred_pb.operator.unary_logical_op.expression
    result = placeholder_pb2.PlaceholderExpression()
    result.operator.unary_logical_op.op = _LogicalOp.NOT.value
    result.operator.unary_logical_op.expression.CopyFrom(pred_pb)
    return result

  def traverse(self) -> Iterator[Placeholder]:
    """Yields all placeholders under this predicate."""
    yield from self.pred_dataclass.traverse()


@attr.s
class _BinaryLogicalExpression:
  """Represents a boolean logical expression with exactly two arguments."""

  logical_op = attr.ib(type=_LogicalOp)
  left = attr.ib(type=_PredicateSubtype)
  right = attr.ib(type=_PredicateSubtype)

  def encode(
      self, component_spec: Optional[Type['types.ComponentSpec']] = None
  ) -> placeholder_pb2.PlaceholderExpression:
    result = placeholder_pb2.PlaceholderExpression()
    result.operator.binary_logical_op.op = self.logical_op.value
    result.operator.binary_logical_op.lhs.CopyFrom(
        self.left.encode(component_spec)
    )
    result.operator.binary_logical_op.rhs.CopyFrom(
        self.right.encode(component_spec)
    )
    return result

  def traverse(self) -> Iterator[Placeholder]:
    """Yields all placeholders under this predicate."""
    yield from self.left.traverse()
    yield from self.right.traverse()


class Predicate(Placeholder):
  """A boolean-valued Placeholder.

  Pipeline authors obtain an instance of Predicate by comparing a
  ChannelWrappedPlaceholder with a primitive (int, float, or str), or by
  comparing two ChannelWrappedPlaceholders with each other.
  The Predicate can then be used to define conditional statements using the
  pipeline-authoring DSL.

  Predicates should be instantiated with syntax like `<channel>.future() > 5`.
  """

  def __init__(self, pred_dataclass: _PredicateSubtype):
    """NOT INTENDED TO BE USED DIRECTLY BY PIPELINE AUTHORS."""

    super().__init__(placeholder_pb2.Placeholder.Type.INPUT_ARTIFACT)
    self.pred_dataclass = pred_dataclass

  def __add__(self, right):
    # Unlike Placeholders, Predicates cannot be added.
    raise NotImplementedError

  def __radd__(self, left):
    # Unlike Placeholders, Predicates cannot be added.
    raise NotImplementedError

  def b64encode(self):
    # Unlike Placeholders, Predicates cannot be b64encoded.
    raise NotImplementedError

  def traverse(self) -> Iterator[Placeholder]:
    """Yields all placeholders under this predicate."""
    yield from self.pred_dataclass.traverse()

  def encode(
      self,
      component_spec: Optional[Type['types.ComponentSpec']] = None
  ) -> placeholder_pb2.PlaceholderExpression:
    return self.pred_dataclass.encode(component_spec)


class ListPlaceholder(Placeholder):
  """List of multiple Placeholders.

  Prefer to use list() to create ListPlaceholder.
  """

  def __init__(self, input_placeholders: List['Placeholder']):
    super().__init__(placeholder_pb2.Placeholder.Type.INPUT_ARTIFACT)
    self._validate_type(input_placeholders)
    self._input_placeholders = input_placeholders

  def __add__(self, right: 'ListPlaceholder'):
    self._input_placeholders.extend(right._input_placeholders)
    return self

  def __radd__(self, left: 'ListPlaceholder'):
    self._input_placeholders = left._input_placeholders + self._input_placeholders
    return self

  def _validate_type(self, input_placeholders: List['Placeholder']):
    for input_placeholder in input_placeholders:
      if not isinstance(input_placeholder, Placeholder):
        raise ValueError('Unexpected input placeholder type: %s.' %
                         type(input_placeholder))

  def serialize_list(self: _T,
                     serialization_format: ListSerializationFormat) -> _T:
    """Serializes list-value placeholder to JSON or comma-separated string.

    Only supports primitive type list element (a.k.a bool, int, float or str) at
    the
    moment; throws runtime error otherwise.

    Args:
       serialization_format: The format of how the proto is serialized.

    Returns:
      A placeholder.
    """
    return self._clone_and_use_operators(
        [_ListSerializationOperator(serialization_format)])

  def traverse(self) -> Iterator[Placeholder]:
    """Yields all placeholders under and including this one."""
    yield from super().traverse()
    for p in self._input_placeholders:
      yield from p.traverse()

  def encode(
      self,
      component_spec: Optional[Type['types.ComponentSpec']] = None
  ) -> placeholder_pb2.PlaceholderExpression:
    result = placeholder_pb2.PlaceholderExpression()
    expressions = result.operator.list_concat_op.expressions
    for input_placeholder in self._input_placeholders:
      expressions.append(input_placeholder.encode(component_spec))
    for op in self._operators:
      result = op.encode(result, component_spec)
    return result


def logical_not(pred: Predicate) -> Predicate:
  """Applies a NOT boolean operation on a Predicate.

  Args:
    pred: The Predicate to apply the NOT operation to.

  Returns:
    The negated Predicate.
  """

  return Predicate(_NotExpression(pred.pred_dataclass))


def logical_and(left: Predicate, right: Predicate) -> Predicate:
  """Applies the AND boolean operation on two Predicates.

  Args:
    left: The first argument of the AND operation.
    right: The second argument of the AND operation.

  Returns:
    The Predicate resulting from the AND operation.
  """

  return Predicate(
      _BinaryLogicalExpression(_LogicalOp.AND, left.pred_dataclass,
                               right.pred_dataclass))


def logical_or(left: Predicate, right: Predicate) -> Predicate:
  """Applies the OR boolean operation on two Predicates.

  Args:
    left: The first argument of the OR operation.
    right: The second argument of the OR operation.

  Returns:
    The Predicate resulting from the OR operation.
  """

  return Predicate(
      _BinaryLogicalExpression(_LogicalOp.OR, left.pred_dataclass,
                               right.pred_dataclass))


def input(key: str) -> ArtifactPlaceholder:  # pylint: disable=redefined-builtin
  """Returns a Placeholder that represents an input artifact.

  Args:
    key: The key of the input artifact.

  Returns:
    A Placeholder that supports
      1. Rendering the whole MLMD artifact proto as text_format.
         Example: input('model')
      2. Accessing a specific index using [index], if multiple artifacts are
         associated with the given key. If not specified, default to the first
         artifact.
         Example: input('model')[0]
      3. Getting the URI of an artifact through .uri property.
         Example: input('model').uri or input('model')[0].uri
      4. Getting the URI of a specific split of an artifact using
         .split_uri(split_name) method.
         Example: input('examples')[0].split_uri('train')
      5. Getting the value of a primitive artifact through .value property.
         Example: input('primitive').value
      6. Concatenating with other placeholders or strings.
         Example: input('model').uri + '/model/' + exec_property('version')
  """
  return ArtifactPlaceholder(placeholder_pb2.Placeholder.Type.INPUT_ARTIFACT,
                             key)


def output(key: str) -> ArtifactPlaceholder:
  """Returns a Placeholder that represents an output artifact.

  It is the same as input(...) function, except it is for output artifacts.

  Args:
    key: The key of the output artifact.

  Returns:
    A Placeholder that supports
      1. Rendering the whole artifact as text_format.
         Example: output('model')
      2. Accessing a specific index using [index], if multiple artifacts are
         associated with the given key. If not specified, default to the first
         artifact.
         Example: output('model')[0]
      3. Getting the URI of an artifact through .uri property.
         Example: output('model').uri or output('model')[0].uri
      4. Getting the URI of a specific split of an artifact using
         .split_uri(split_name) method.
         Example: output('examples')[0].split_uri('train')
      5. Getting the value of a primitive artifact through .value property.
         Example: output('primitive').value
      6. Concatenating with other placeholders or strings.
         Example: output('model').uri + '/model/' + exec_property('version')
  """
  return ArtifactPlaceholder(placeholder_pb2.Placeholder.Type.OUTPUT_ARTIFACT,
                             key)


def exec_property(key: str) -> ExecPropertyPlaceholder:
  """Returns a Placeholder that represents an execution property.

  Args:
    key: The key of the output artifact.

  Returns:
    A Placeholder that supports
      1. Rendering the value of an execution property at a given key.
         Example: exec_property('version')
      2. Rendering the whole proto or a proto field of an execution property,
         if the value is a proto type.
         The (possibly nested) proto field in a placeholder can be accessed as
         if accessing a proto field in Python.
         Example: exec_property('model_config').num_layers
      3. Concatenating with other placeholders or strings.
         Example: output('model').uri + '/model/' + exec_property('version')
  """
  return ExecPropertyPlaceholder(key)


class RuntimeInfoKey(enum.Enum):
  PLATFORM_CONFIG = 'platform_config'
  EXECUTOR_SPEC = 'executor_spec'


_RUNTIME_INFO_KEYS = frozenset(key.value for key in RuntimeInfoKey)


def runtime_info(key: str) -> RuntimeInfoPlaceholder:
  """Returns a Placeholder that contains runtime information for component.

  Currently the runtime info includes following keys:
  1. platform_config: A platform_config proto that contains platform specific
     information.
  2. executor_spec: The executor spec proto.

  Args:
    key: The key of the runtime information.

  Returns:
    A Placeholder that will render to the information associated with the key.
    If the placeholder is proto-valued. Accessing a proto field can be
    represented as if accessing a proto field in Python.

  Raises:
    ValueError: If received unsupported key.
  """
  if key not in _RUNTIME_INFO_KEYS:
    raise ValueError(f'Got unsupported key: {key}.')
  return RuntimeInfoPlaceholder(key)


def execution_invocation() -> ExecInvocationPlaceholder:
  """Returns a Placeholder representing ExecutionInvocation proto.

  Returns:
    A Placeholder that will render to the ExecutionInvocation proto.
    Accessing a proto field is the same as if accessing a proto field in Python.

    Prefer to use input(key)/output(key)/exec_property(key) functions instead of
    input_dict/output_dict/execution_properties field from ExecutionInvocation
    proto.
  """
  return ExecInvocationPlaceholder()


def environment_variable(key: str) -> EnvironmentVariablePlaceholder:
  """Returns a Placeholder representing EnvironmentVariable proto.

  Args:
    key: The key of the environment variable.

  Returns:
    A Placeholder that supports
      1. Rendering the value of an environment variable for a given key.
         Example: environment_variable('FOO')
      2. Concatenating with other placeholders or strings.
         Example: 'foo=' + environment_variable('FOO')
  """
  return EnvironmentVariablePlaceholder(key)


def to_list(input_placeholders: List['Placeholder']) -> ListPlaceholder:
  """Returns a ListPlaceholder representing a list of input placeholders."""
  return ListPlaceholder(input_placeholders)
