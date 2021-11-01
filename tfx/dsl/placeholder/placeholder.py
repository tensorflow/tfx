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
"""Placeholders represent not-yet-available values at the component authoring time."""

import abc
import copy
import enum
from typing import Any, Callable, Iterator, List, Optional, Type, TypeVar, Union, cast

import attr
from tfx.proto.orchestration import placeholder_pb2
from tfx.utils import json_utils
from tfx.utils import proto_utils

from google.protobuf import message

# To resolve circular dependency caused by type annotations.
# TODO(b/191610358): Reduce the number of circular type-dependencies.
types = Any  # tfx.types imports channel.py, which in turn imports this module.

# TODO(b/190409099): Support RuntimeParameter.
_ValueLikeType = Union[int, float, str, 'ChannelWrappedPlaceholder']


class _PlaceholderOperator(json_utils.Jsonable):
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

  def placeholders_involved(self) -> List['Placeholder']:
    return []


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

  def __init__(self, index: int):
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
    result.operator.index_op.index = self._index
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

  def placeholders_involved(self) -> List['Placeholder']:
    if self._right and isinstance(self._right, Placeholder):
      return self._right.placeholders_involved()
    return []


class ProtoSerializationFormat(enum.Enum):
  TEXT_FORMAT = placeholder_pb2.ProtoOperator.TEXT_FORMAT
  JSON = placeholder_pb2.ProtoOperator.JSON
  BINARY = placeholder_pb2.ProtoOperator.BINARY


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
    if (component_spec and sub_expression_pb.placeholder.type ==
        placeholder_pb2.Placeholder.EXEC_PROPERTY):
      exec_property_name = sub_expression_pb.placeholder.key
      if exec_property_name not in component_spec.PARAMETERS:
        raise ValueError(
            f"Can't find provided placeholder key {exec_property_name} in "
            "component spec's exec properties. "
            f"Available exec property keys: {component_spec.PARAMETERS.keys()}."
        )
      execution_param = component_spec.PARAMETERS[exec_property_name]
      if not issubclass(execution_param.type, message.Message):
        raise ValueError(
            "Can't apply placeholder proto operator on non-proto type "
            f"exec property. Got {execution_param.type}.")
      proto_schema = result.operator.proto_op.proto_schema
      proto_schema.message_type = execution_param.type.DESCRIPTOR.full_name
      proto_utils.build_file_descriptor_set(execution_param.type,
                                            proto_schema.file_descriptors)

    return result


class _ListSerializationOperator(_PlaceholderOperator):
  """ListSerializationOperator serializes list type placeholder.

  Prefer to use the .serialize_list property of ExecPropertyPlaceholder.
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


class Placeholder(json_utils.Jsonable):
  """A Placeholder represents not-yet-available values at the component authoring time."""

  def __init__(self, placeholder_type: placeholder_pb2.Placeholder.Type,
               key: Optional[str] = None):
    self._operators = []
    self._type = placeholder_type
    self._key = key

  def __add__(self, right: Union[str, 'Placeholder']):
    self._operators.append(_ConcatOperator(right=right))
    return self

  def __radd__(self, left: str):
    self._operators.append(_ConcatOperator(left=left))
    return self

  def __deepcopy__(self, memo):
    # This method is implemented to make sure Placeholder is deep copyable
    # by copy.deepcopy().
    cls = self.__class__
    result = cls.__new__(cls)
    memo[id(self)] = result
    for k, v in self.__dict__.items():
      setattr(result, k, copy.deepcopy(v, memo))
    return result

  def b64encode(self):
    """Encodes the output of another placeholder using url safe base64 encoding.

    Returns:
      A placeholder, when rendering, is a url safe base64 encoded string.
    """
    self._operators.append(_Base64EncodeOperator())
    return self

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

  def placeholders_involved(self) -> List['Placeholder']:
    """Returns a list of all Placeholder involved in this expression."""
    result = [self]
    for op in self._operators:
      result.extend(op.placeholders_involved())
    return result


# To ensure that ArtifactPlaceholder operations on a ChannelWrappedPlaceholder
# still returns a ChannelWrappedPlaceholder.
_T = TypeVar('_T')


class ArtifactPlaceholder(Placeholder):
  """Artifact Placeholder represents an input or an output artifact.

  Prefer to use input(...) or output(...) to create artifact placeholders.
  """

  @property
  def uri(self: _T) -> _T:
    self._try_inject_index_operator()
    self._operators.append(_ArtifactUriOperator())
    return self

  def split_uri(self: _T, split: str) -> _T:
    self._try_inject_index_operator()
    self._operators.append(_ArtifactUriOperator(split))
    return self

  @property
  def value(self: _T) -> _T:
    self._try_inject_index_operator()
    self._operators.append(_ArtifactValueOperator())
    return self

  def __getitem__(self: _T, key: int) -> _T:
    self._operators.append(_IndexOperator(key))
    return self

  def _try_inject_index_operator(self):
    if not self._operators or not any(
        isinstance(op, _IndexOperator) for op in self._operators):
      self._operators.append(_IndexOperator(0))

  def property(self, key: str):
    self._operators.append(_PropertyOperator(key))
    return self

  def custom_property(self, key: str):
    self._operators.append(_PropertyOperator(key, is_custom_property=True))
    return self


class _ProtoAccessiblePlaceholder(Placeholder, abc.ABC):
  """A base Placeholder for accessing proto fields using Python proto syntax."""

  def __getattr__(self, field_name: str):
    proto_access_field = f'.{field_name}'
    if self._operators and isinstance(
        self._operators[-1],
        _ProtoOperator) and self._operators[-1].can_append_field_path():
      self._operators[-1].append_field_path(proto_access_field)
    else:
      self._operators.append(
          _ProtoOperator(proto_field_path=proto_access_field))
    return self

  def __getitem__(self, key: Union[int, str]):
    if isinstance(key, int):
      self._operators.append(_IndexOperator(key))
    else:
      proto_access_field = f'[{key!r}]'
      if self._operators and isinstance(
          self._operators[-1],
          _ProtoOperator) and self._operators[-1].can_append_field_path():
        self._operators[-1].append_field_path(proto_access_field)
      else:
        self._operators.append(
            _ProtoOperator(proto_field_path=proto_access_field))
    return self

  def serialize(self, serialization_format: ProtoSerializationFormat):
    """Serialize the proto-valued placeholder using the provided scheme.

    Args:
      serialization_format: The format of how the proto is serialized.

    Returns:
      A placeholder that when rendered is serialized with the scheme.
    """
    self._operators.append(
        _ProtoOperator(serialization_format=serialization_format))
    return self


class ExecPropertyPlaceholder(_ProtoAccessiblePlaceholder):
  """ExecProperty Placeholder represents an execution property.

  Prefer to use exec_property(...) to create exec property placeholders.
  """

  def __init__(self, key: str):
    super().__init__(placeholder_pb2.Placeholder.Type.EXEC_PROPERTY, key)

  def serialize_list(self, serialization_format: ListSerializationFormat):
    """Serializes list-value placeholder to JSON or comma-separated string.

    Here list value includes repeated proto field. This function only
    supports primitive type list element (a.k.a bool, int, float or str) at the
    moment; throws runtime error otherwise.

    Args:
       serialization_format: The format of how the proto is serialized.

    Returns:
      A placeholder.
    """
    self._operators.append(_ListSerializationOperator(serialization_format))
    return self


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

  def __eq__(self, other: _ValueLikeType) -> 'Predicate':
    return Predicate.from_comparison(_CompareOp.EQUAL, left=self, right=other)

  def __ne__(self, other: _ValueLikeType) -> 'Predicate':
    return logical_not(self == other)

  def __lt__(self, other: _ValueLikeType) -> 'Predicate':
    return Predicate.from_comparison(
        _CompareOp.LESS_THAN, left=self, right=other)

  def __le__(self, other: _ValueLikeType) -> 'Predicate':
    return logical_not(self > other)

  def __gt__(self, other: _ValueLikeType) -> 'Predicate':
    return Predicate.from_comparison(
        _CompareOp.GREATER_THAN, left=self, right=other)

  def __ge__(self, other: _ValueLikeType) -> 'Predicate':
    return logical_not(self < other)


def _encode_value_like(
    x: _ValueLikeType,
    channel_to_key_fn: Optional[Callable[['types.Channel'], str]] = None
) -> placeholder_pb2.PlaceholderExpression:
  """Encodes x to a placeholder expression proto."""

  if isinstance(x, ChannelWrappedPlaceholder):
    if channel_to_key_fn:
      # pylint: disable=protected-access
      old_key = x._key
      x._key = channel_to_key_fn(x.channel)
      result = x.encode()
      x._key = old_key
      # pylint: enable=protected-access
    else:
      result = x.encode()
    return result
  result = placeholder_pb2.PlaceholderExpression()
  if isinstance(x, int):
    result.value.int_value = x
  elif isinstance(x, float):
    result.value.double_value = x
  elif isinstance(x, str):
    result.value.string_value = x
  else:
    raise ValueError(
        f'x must be an int, float, str, or ChannelWrappedPlaceholder. x: {x}')
  return result


_PredicateSubtype = Union['_Comparison', '_NotExpression',
                          '_BinaryLogicalExpression']


@attr.s
class _Comparison:
  """Represents a comparison between two placeholders."""

  compare_op = attr.ib(type=_CompareOp)
  left = attr.ib(type=_ValueLikeType)
  right = attr.ib(type=_ValueLikeType)

  def encode_with_keys(
      self,
      channel_to_key_fn: Optional[Callable[['types.Channel'], str]] = None
  ) -> placeholder_pb2.PlaceholderExpression:
    """Encode as a PlaceholderExpression proto."""

    result = placeholder_pb2.PlaceholderExpression()
    result.operator.compare_op.op = self.compare_op.value
    left_pb = _encode_value_like(self.left, channel_to_key_fn)
    result.operator.compare_op.lhs.CopyFrom(left_pb)
    right_pb = _encode_value_like(self.right, channel_to_key_fn)
    result.operator.compare_op.rhs.CopyFrom(right_pb)
    return result

  def dependent_channels(self) -> Iterator['types.Channel']:
    if isinstance(self.left, ChannelWrappedPlaceholder):
      yield self.left.channel
    if isinstance(self.right, ChannelWrappedPlaceholder):
      yield self.right.channel


class _LogicalOp(enum.Enum):
  """An alias for logical operation enums in placeholder.proto."""

  NOT = placeholder_pb2.UnaryLogicalOperator.Operation.NOT
  AND = placeholder_pb2.BinaryLogicalOperator.Operation.AND
  OR = placeholder_pb2.BinaryLogicalOperator.Operation.OR


@attr.s
class _NotExpression:
  """Represents a logical negation."""

  pred_dataclass = attr.ib(type=_PredicateSubtype)

  def encode_with_keys(
      self,
      channel_to_key_fn: Optional[Callable[['types.Channel'], str]] = None
  ) -> placeholder_pb2.PlaceholderExpression:
    """Encode as a PlaceholderExpression proto."""

    pred_pb = self.pred_dataclass.encode_with_keys(channel_to_key_fn)
    # not(not(a)) becomes a
    if isinstance(self.pred_dataclass, _NotExpression):
      return pred_pb.operator.unary_logical_op.expression
    result = placeholder_pb2.PlaceholderExpression()
    result.operator.unary_logical_op.op = _LogicalOp.NOT.value
    pred_pb = self.pred_dataclass.encode_with_keys(channel_to_key_fn)
    result.operator.unary_logical_op.expression.CopyFrom(pred_pb)
    return result

  def dependent_channels(self) -> Iterator['types.Channel']:
    yield from self.pred_dataclass.dependent_channels()


@attr.s
class _BinaryLogicalExpression:
  """Represents a boolean logical expression with exactly two arguments."""

  logical_op = attr.ib(type=_LogicalOp)
  left = attr.ib(type=_PredicateSubtype)
  right = attr.ib(type=_PredicateSubtype)

  def encode_with_keys(
      self,
      channel_to_key_fn: Optional[Callable[['types.Channel'], str]] = None
  ) -> placeholder_pb2.PlaceholderExpression:
    """Encode as a PlaceholderExpression proto."""

    result = placeholder_pb2.PlaceholderExpression()
    result.operator.binary_logical_op.op = self.logical_op.value
    left_pb = self.left.encode_with_keys(channel_to_key_fn)
    result.operator.binary_logical_op.lhs.CopyFrom(left_pb)
    right_pb = self.right.encode_with_keys(channel_to_key_fn)
    result.operator.binary_logical_op.rhs.CopyFrom(right_pb)
    return result

  def dependent_channels(self) -> Iterator['types.Channel']:
    yield from self.left.dependent_channels()
    yield from self.right.dependent_channels()


class Predicate(Placeholder):
  """A boolean-valued Placeholder.

  Pipeline authors obtain an instance of Predicate by comparing a
  ChannelWrappedPlaceholder with a primitive (int, float, or str), or by
  comparing two ChannelWrappedPlaceholders with each other.
  The Predicate can then be used to define conditional statements using the
  pipeline-authoring DSL.

  Prefer to use syntax like `<channel>.future() > 5` to create a Predicate.
  """

  def __init__(self, pred_dataclass: _PredicateSubtype):
    """NOT INTENDED TO BE USED DIRECTLY BY PIPELINE AUTHORS."""

    super().__init__(placeholder_pb2.Placeholder.Type.INPUT_ARTIFACT)
    self.pred_dataclass = pred_dataclass

  @classmethod
  def from_comparison(cls, compare_op: _CompareOp,
                      left: ChannelWrappedPlaceholder,
                      right: _ValueLikeType) -> 'Predicate':
    """Creates a Predicate instance.

    Note that even though the `left` argument is assumed to be a
    ChannelWrappedPlaceholder, we can still compare placeholders like
    this:

      `5 > channel.future()[0].value()`

    This is because python's comparison operations will automatically flip
    the arguments if only the right side has overloaded comparision
    operations. e.g.,

      `5 > channel.future()[0].value()` becomes
      `(5).__gt__(pred)` which becomes
      `pred.__lt__(5)`.

    Thus, the left argument can always be assumed to be an
    ChannelWrappedPlaceholder.

    Args:
      compare_op: A _CompareOp (EQUAL, LESS_THAN, GREATER_THAN)
      left: The left side of the comparison.
      right: The right side of the comparison. Might also be an int, float, or
        str.

    Returns:
      A Predicate.
    """
    return cls(_Comparison(compare_op, left, right))

  def __add__(self, right):
    # Unlike Placeholders, Predicates cannot be added.
    raise NotImplementedError

  def __radd__(self, left):
    # Unlike Placeholders, Predicates cannot be added.
    raise NotImplementedError

  def b64encode(self):
    # Unlike Placeholders, Predicates cannot be b64encoded.
    raise NotImplementedError

  def dependent_channels(self) -> Iterator['types.Channel']:
    yield from self.pred_dataclass.dependent_channels()

  def encode(
      self,
      component_spec: Optional[Type['types.ComponentSpec']] = None
  ) -> placeholder_pb2.PlaceholderExpression:
    """This just calls `encode_with_keys` without arguments."""
    del component_spec  # not used for encoding Predicates
    return self.encode_with_keys()

  def encode_with_keys(
      self,
      channel_to_key_fn: Optional[Callable[['types.Channel'], str]] = None
  ) -> placeholder_pb2.PlaceholderExpression:
    """Encodes a Predicate as a PlaceholderExpression proto.

    When a ChannelWrappedPlaceholder is initially constructed, it does not have
    a key. This means that Predicates, which comprise of
    ChannelWrappedPlaceholders, do not contain sufficient information to be
    evaluated at run time. Thus, after the Pipeline is fully defined, the
    compiler will use this method to fill in the keys.

    Within the context of each node, each Channel corresponds to a key in the
    input_dict (see `ResolverStrategy.resolve_artifacts`).
    The Predicate has an internal tree data structure to keep track of
    all the placeholders and operations. As we traverse this tree to create
    this proto, `channel_to_key_fn` is called each time a
    ChannelWrappedPlaceholder is encountered, and its output is used as the
    placeholder key in the Placeholder proto produced during encoding.

    Note that the ChannelWrappedPlaceholder itself remains unchanged.

    Args:
      channel_to_key_fn: The function used to determine the placeholder key for
        each ChannelWrappedPlaceholder. If None, no attempt to fill in the
        placeholder keys will be made.

    Returns:
      PlaceholderExpression proto containing all information of this Predicate,
      with the placeholder keys filled in. Note that the Predicate instance
      is unchanged.
    """

    return self.pred_dataclass.encode_with_keys(channel_to_key_fn)


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
