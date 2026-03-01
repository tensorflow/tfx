# Copyright 2023 Google LLC. All Rights Reserved.
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
"""Placeholder base type and basic operators (standard Python operations)."""

from __future__ import annotations

import abc
import enum
import functools
import types
import typing
from typing import Any, Iterator, Mapping, Optional, Sequence, Union

import attr
from tfx.proto.orchestration import placeholder_pb2
from tfx.utils import proto_utils

from google.protobuf import message

# We cannot depend on anything under tfx.types because:
# 1. its channel.py (via its __init__.py) needs access to (Artifact)Placeholder.
# 2. its component_spec.py (via its __init__.py) needs access to Placeholder.
#
# This means we currently cannot use the real tfx.types.* dependency in the
# following use cases we have here in the placeholder module:
# 1. encode() needs ComponentSpec to resolve the PARAMETERS field. This is just
#    a pytype declaration, so we can work around with `types.ComponentSpec`.
# 2. ArtifactPlaceholder should have expected_type=tfx.types.Artifact for better
#    type checks during Tflex compilation. This is currently just set to None,
#    resulting in no type checking.
# The conceptually right dependency chain would be:
# Channel+ChannelWrappedPlaceholder -> ArtifactPlaceholder -> Artifact
#                                       \-> Placeholder -> ComponentSpecIntf
# where ComponentSpecIntf is an interface that gives access to the few things
# that Placeholder::encode() needs from the spec. TODO(pke) Implement.
# TODO(b/191610358): Reduce the number of circular type-dependencies.
types = Any  # To resolve circular dependency caused by type annotations.

# TODO(b/190409099): Support RuntimeParameter.
ValueType = Union[int, float, str, bool]
ValueLikeType = Union[ValueType, 'Placeholder']


class Placeholder(abc.ABC):
  """A placeholder value computed based on a tree of Placeholders and operators.

  This is the base class of the Python placeholder API. It allows users of the
  Tflex DSL to construct somewhat complex expressions with a convenient Python
  API (e.g. using the + operator for string concatenation). Every Placeholder
  instance represents an expression with certain (future) inputs that will yield
  a value of the `expected_type` during pipeline execution.

  Placeholder instances are immutable. (There is one, very well controlled
  exception in ChannelWrappedPlaceholder.set_key(), but the way it is called
  still allows users to treat placeholders as immutable.) Each Placeholder
  instance represents a tree of sub-expressions, which are also immutable, so
  the entire tree is immutable. Thus, newly created Placeholder instances can
  can safely reference any pre-existing Placeholder instances (including their
  entire sub-trees) without having to worry about them being mutated. Any given
  Placeholder could be referenced by multiple parents.

  The ultimate purpose of a Placeholder expression tree is to be encoded into a
  PlaceholderExpression proto, which then becomes part of the intermediate
  representation (IR) shipped to the orchestrator for pipeline execution. So
  this Python API only allows building placeholder expressions and the runtime
  only knows how to evaluate encoded PlaceholderExpressions.
  """

  def __init__(self, expected_type: Optional[type[Any]]):
    """Creates a new Placeholder. Consider this private.

    Args:
      expected_type: The Python type (Union types are allowed) that this
        Placeholder will evaluate to. None means that we don't know the type.
    """
    self.expected_type = expected_type

  def __deepcopy__(self, memo):
    # Placeholders are immutable. While nobody should (want to) invoke deepcopy
    # on a placeholder itself, when they're being cloned as part of a larger
    # deepcopy operation, it is safe to just return the same instance.
    return self

  def _is_maybe_proto_valued(self) -> bool:
    """True if the Placeholder might evaluate to a proto."""
    return _is_maybe_subclass(self.expected_type, message.Message)

  # Functions that allow the Tflex DSL user to apply standard Python operators
  # on a Placeholder, obtaining new Placeholders for that expression.

  def __getitem__(self, key: Union[int, str]) -> Placeholder:
    if isinstance(key, str) and self._is_maybe_proto_valued():
      return _ProtoOperator(self, proto_field_path=[f'[{key!r}]'])
    return _IndexOperator(self, key, is_proto=self._is_maybe_proto_valued())

  def __getattr__(self, field_name: str) -> Placeholder:
    if not field_name.startswith('__') and self._is_maybe_proto_valued():
      return _ProtoOperator(self, proto_field_path=[f'.{field_name}'])
    return super().__getattribute__(field_name)

  def __add__(self, right: Union[str, Placeholder]) -> _ConcatOperator:
    return _ConcatOperator([self, right])

  def __radd__(self, left: str) -> _ConcatOperator:
    return _ConcatOperator([left, self])

  def __eq__(self, other: ValueLikeType) -> 'Predicate':
    # https://github.com/PyCQA/pylint/issues/5857 pylint: disable=too-many-function-args
    return _ComparisonPredicate(_CompareOp.EQUAL, self, other)

  def __ne__(self, other: ValueLikeType) -> 'Predicate':
    return logical_not(self == other)

  def __lt__(self, other: ValueLikeType) -> 'Predicate':
    # https://github.com/PyCQA/pylint/issues/5857 pylint: disable=too-many-function-args
    return _ComparisonPredicate(_CompareOp.LESS_THAN, self, other)

  def __le__(self, other: ValueLikeType) -> 'Predicate':
    return logical_not(self > other)

  def __gt__(self, other: ValueLikeType) -> 'Predicate':
    # https://github.com/PyCQA/pylint/issues/5857 pylint: disable=too-many-function-args
    return _ComparisonPredicate(_CompareOp.GREATER_THAN, self, other)

  def __ge__(self, other: ValueLikeType) -> 'Predicate':
    return logical_not(self < other)

  # Additional functions that Tflex DSL users can apply to their Placeholders,
  # obtaining new Placeholders that represent these transformations.

  def __iter__(self) -> Iterator[Any]:
    raise RuntimeError(
        'Iterating over a placeholder is not supported. '
        'Did you miss the ending `,` in your tuple?'
    )

  def __format__(self, format_spec) -> str:
    raise RuntimeError(
        'Formatting a placeholder is not supported. Did you accidentally use a '
        'placeholder inside an f-string or .format() call? That cannot work '
        'because placeholder values are only known later at runtime. You can '
        'use the + operator for string concatenation.'
    )

  def b64encode(self, url_safe: bool = True) -> _Base64EncodeOperator:
    """Encodes the value with URL-safe Base64 encoding."""
    return _Base64EncodeOperator(self, url_safe)

  def serialize(
      self,
      serialization_format: ProtoSerializationFormat,
  ) -> _ProtoOperator:
    """Serializes the proto-valued placeholder using the provided format.

    Args:
      serialization_format: The format of how the proto is serialized.

    Returns:
      A placeholder representing the serialized proto value.
    """
    assert self._is_maybe_proto_valued()
    return _ProtoOperator(self, [], serialization_format)

  # TODO(pke) Move this down to only the sub-classes that really support it, if
  # pytype allows.
  def serialize_list(
      self,
      serialization_format: ListSerializationFormat,
  ) -> Placeholder:
    """Serializes a list-valued placeholder to JSON or comma-separated string.

    Here list value includes repeated proto field. This function only
    supports primitive type list element (a.k.a bool, int, float or str) at the
    moment; throws runtime error otherwise.

    Args:
       serialization_format: The format of how the list is serialized.

    Returns:
      A placeholder representing the serialized list.
    """
    return _ListSerializationOperator(self, serialization_format)

  @abc.abstractmethod
  def internal_equals(self, other: Placeholder) -> bool:
    """Do not call this as a Tflex user."""
    raise NotImplementedError()

  @abc.abstractmethod
  def encode(
      self, component_spec: Optional[type['types.ComponentSpec']] = None
  ) -> placeholder_pb2.PlaceholderExpression:
    """Do not call this as a Tflex user.

    Encodes the Placeholder for later eval.

    Args:
      component_spec: A Tflex component spec whose PARAMETERS field will be used
        to determine the proto types of its inputs/outputs/parameters. This
        allows the encoded placeholder to include the proto descriptors.

    Returns:
      An encoded PlaceholderExpression, which when evaluated later at pipeline
      runtime will result in the value represented by this Placeholder.
    """
    raise NotImplementedError()

  def traverse(self) -> Iterator[Placeholder]:
    """Yields all placeholders under and including this one."""
    yield self


class Predicate(Placeholder):
  """A boolean-valued Placeholder.

  Pipeline authors obtain an instance of Predicate by comparing a
  Placeholder with a primitive (int, float, or str), or by
  comparing two Placeholders with each other.
  The Predicate can then be used to define conditional statements using the
  pipeline-authoring DSL.

  Predicates should be instantiated with syntax like `<channel>.future() > 5`.
  """

  def __init__(self):
    """Initializes the class. Consider this private."""
    super().__init__(expected_type=bool)

  def __add__(self, right):
    # Unlike Placeholders, Predicates cannot be added.
    raise NotImplementedError

  def __radd__(self, left):
    # Unlike Placeholders, Predicates cannot be added.
    raise NotImplementedError

  def b64encode(self, url_safe: bool = True):
    # Unlike Placeholders, Predicates cannot be b64encoded.
    raise NotImplementedError


def logical_not(pred: Predicate) -> Predicate:
  """Applies a NOT boolean operation on a Predicate.

  Args:
    pred: The Predicate to apply the NOT operation to.

  Returns:
    The negated Predicate.
  """
  # https://github.com/PyCQA/pylint/issues/5857 pylint: disable=too-many-function-args
  return _NotPredicate(pred)


def logical_and(left: Predicate, right: Predicate) -> Predicate:
  """Applies the AND boolean operation on two Predicates.

  Args:
    left: The first argument of the AND operation.
    right: The second argument of the AND operation.

  Returns:
    The Predicate resulting from the AND operation.
  """
  # https://github.com/PyCQA/pylint/issues/5857 pylint: disable=too-many-function-args
  return _BinaryLogicalPredicate(
      placeholder_pb2.BinaryLogicalOperator.Operation.AND, left, right
  )


def logical_or(left: Predicate, right: Predicate) -> Predicate:
  """Applies the OR boolean operation on two Predicates.

  Args:
    left: The first argument of the OR operation.
    right: The second argument of the OR operation.

  Returns:
    The Predicate resulting from the OR operation.
  """
  # https://github.com/PyCQA/pylint/issues/5857 pylint: disable=too-many-function-args
  return _BinaryLogicalPredicate(
      placeholder_pb2.BinaryLogicalOperator.Operation.OR, left, right
  )


def make_list(
    input_placeholders: list[ValueLikeType],
) -> ListPlaceholder:
  """Returns a ListPlaceholder representing a list of input placeholders."""
  return ListPlaceholder(input_placeholders)


def join(
    placeholders: Sequence[Union[str, Placeholder]],
    separator: str = '',
) -> Union[str, Placeholder]:
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


def join_path(
    *args: str | Placeholder,
) -> Placeholder:
  """Runs os.path.join() on placeholder arguments.

  Args:
    *args: (Placeholders that resolve to) strings which will be passed to
      os.path.join().

  Returns:
    A placeholder that will resolve to the joined path.
  """
  return _JoinPathOperator(*args)


class ListPlaceholder(Placeholder):
  """List of multiple Placeholders.

  Prefer to use ph.make_list() to create ListPlaceholder.
  """

  def __init__(self, input_placeholders: list[ValueLikeType]):
    """Initializes the class. Consider this private."""
    super().__init__(expected_type=list)
    self._input_placeholders = input_placeholders

  def __add__(self, right: ListPlaceholder) -> ListPlaceholder:
    return ListPlaceholder(self._input_placeholders + right._input_placeholders)

  def __radd__(self, left: ListPlaceholder) -> ListPlaceholder:
    return ListPlaceholder(left._input_placeholders + self._input_placeholders)

  def serialize_list(
      self, serialization_format: ListSerializationFormat
  ) -> _ListSerializationOperator:
    """Serializes list-value placeholder to JSON or comma-separated string.

    Only supports primitive type list element (a.k.a bool, int, float or str) at
    the moment; throws runtime error otherwise.

    Args:
       serialization_format: The format of how the proto is serialized.

    Returns:
      A placeholder.
    """
    return _ListSerializationOperator(self, serialization_format)

  def internal_equals(self, other: Placeholder) -> bool:
    return (
        isinstance(other, ListPlaceholder)
        and len(self._input_placeholders) == len(other._input_placeholders)  # pylint: disable=protected-access
        and all(
            internal_equals_value_like(a, b)
            for a, b in zip(self._input_placeholders, other._input_placeholders)  # pylint: disable=protected-access
        )
    )

  def traverse(self) -> Iterator[Placeholder]:
    """Yields all placeholders under and including this one."""
    yield from super().traverse()
    for p in self._input_placeholders:
      if isinstance(p, Placeholder):
        yield from p.traverse()

  def encode(
      self, component_spec: Optional[type['types.ComponentSpec']] = None
  ) -> placeholder_pb2.PlaceholderExpression:
    result = placeholder_pb2.PlaceholderExpression()
    result.operator.list_concat_op.SetInParent()
    expressions = result.operator.list_concat_op.expressions
    for input_placeholder in self._input_placeholders:
      expressions.append(encode_value_like(input_placeholder, component_spec))
    return result


def make_dict(
    entries: Union[
        Mapping[str, Union[ValueLikeType, None]],
        Sequence[tuple[Union[str, Placeholder], Union[ValueLikeType, None]]],
    ],
) -> DictPlaceholder:
  """Returns a DictPlaceholder representing a dict of input placeholders.

  Args:
    entries: A mapping that will become the final dict after running placeholder
      resolution on each of the values. Values that resolve to None are dropped.
      If you also want placeholders in the keys, you need to pass the dict as a
      sequence of (k,v) tuples, whereby the key placeholder must evaluate to a
      string.

  Returns:
    A placeholder that will resolve to a dict with the given entries.
  """
  if isinstance(entries, Mapping):
    entries = entries.items()
  return DictPlaceholder(entries)


class DictPlaceholder(Placeholder):
  """Dict of multiple Placeholders. None values are dropped.

  Prefer to use ph.make_dict() to create DictPlaceholder.
  """

  def __init__(
      self,
      entries: Sequence[
          tuple[Union[str, Placeholder], Optional[ValueLikeType]]
      ],
  ):
    """Initializes the class. Consider this private."""
    super().__init__(expected_type=dict)
    self._entries = entries

  def __add__(self, right: DictPlaceholder) -> DictPlaceholder:
    raise NotImplementedError('Add operator not supported for DictPlaceholders')

  def __radd__(self, left: DictPlaceholder) -> DictPlaceholder:
    raise NotImplementedError('Add operator not supported for DictPlaceholders')

  def internal_equals(self, other: Placeholder) -> bool:
    return (
        isinstance(other, DictPlaceholder)
        and len(self._entries) == len(other._entries)  # pylint: disable=protected-access
        and all(
            internal_equals_value_like(ak, bk)
            and internal_equals_value_like(av, bv)
            for (ak, av), (bk, bv) in zip(self._entries, other._entries)  # pylint: disable=protected-access
        )
    )

  def traverse(self) -> Iterator[Placeholder]:
    """Yields all placeholders under and including this one."""
    yield from super().traverse()
    for key, value in self._entries:
      if isinstance(key, Placeholder):
        yield from key.traverse()
      if isinstance(value, Placeholder):
        yield from value.traverse()

  def encode(
      self, component_spec: Optional[type['types.ComponentSpec']] = None
  ) -> placeholder_pb2.PlaceholderExpression:
    result = placeholder_pb2.PlaceholderExpression()
    result.operator.make_dict_op.SetInParent()
    entries = result.operator.make_dict_op.entries
    for key, value in self._entries:
      if value is None:
        continue  # Drop None values
      entries.add(
          key=encode_value_like(key, component_spec),
          value=encode_value_like(value, component_spec),
      )
    return result


class UnaryPlaceholderOperator(Placeholder):
  """Helper class for a placeholder that operates on a single input."""

  def __init__(self, value: Placeholder, expected_type: Optional[type[Any]]):
    """Initializes the class. Consider this private."""
    super().__init__(expected_type)
    self._value = value

  def internal_equals(self, other: Placeholder) -> bool:
    return isinstance(other, type(self)) and self._value.internal_equals(
        other._value  # pylint: disable=protected-access
    )

  def traverse(self) -> Iterator[Placeholder]:
    yield self
    yield from self._value.traverse()


# We wrap the proto enums so we can use them as types. Once we only support
# Python 3.10+, this can be cleaned up.
class ProtoSerializationFormat(enum.Enum):
  TEXT_FORMAT = placeholder_pb2.ProtoOperator.TEXT_FORMAT
  JSON = placeholder_pb2.ProtoOperator.JSON
  BINARY = placeholder_pb2.ProtoOperator.BINARY
  INLINE_FILE_TEXT_FORMAT = (
      placeholder_pb2.ProtoOperator.INLINE_FILE_TEXT_FORMAT
  )


class ListSerializationFormat(enum.Enum):
  JSON = placeholder_pb2.ListSerializationOperator.JSON
  COMMA_SEPARATED_STR = (
      placeholder_pb2.ListSerializationOperator.COMMA_SEPARATED_STR
  )


def _is_maybe_subclass(
    test_type: Optional[type[Any]], parent_type: type[Any]
) -> bool:
  """Like issubclass(), but supports Union types on the sub-class side.

  Args:
    test_type: A sub-type to test. Can be a Union or a plain class.
    parent_type: A parent type (class, type or tuple of classes/types).

  Returns:
    True if the test_type is a sub-type of the parent_type. If the test_type is
    a Union, any of them is allowed. If it's None, returns True.
  """
  if test_type is None:
    return True
  if typing.get_origin(test_type) == Union:
    return any(
        _is_maybe_subclass(t, parent_type) for t in typing.get_args(test_type)
    )
  assert typing.get_origin(test_type) is None
  return issubclass(test_type, parent_type)


class _IndexOperator(UnaryPlaceholderOperator):
  """Index Operator extracts value at the given index of a Placeholder.

  Do not instantiate directly, use the [index] operator on your existing
  Placeholder objects instead.
  """

  def __init__(
      self, value: Placeholder, index: Union[int, str], is_proto: bool
  ):
    super().__init__(
        value,
        expected_type=(
            Union[ValueType, message.Message] if is_proto else ValueType
        ),
    )
    self._index = index

  def internal_equals(self, other: Placeholder) -> bool:
    return (
        isinstance(other, _IndexOperator)
        and self._index == other._index  # pylint: disable=protected-access
        and self._value.internal_equals(other._value)  # pylint: disable=protected-access
    )

  def encode(
      self, component_spec: Optional[type['types.ComponentSpec']] = None
  ) -> placeholder_pb2.PlaceholderExpression:
    result = placeholder_pb2.PlaceholderExpression()
    result.operator.index_op.expression.CopyFrom(
        self._value.encode(component_spec)
    )
    if isinstance(self._index, int):
      result.operator.index_op.index = self._index
    if isinstance(self._index, str):
      result.operator.index_op.key = self._index
    return result


class _ConcatOperator(Placeholder):
  """Concat Operator concatenates multiple Placeholders.

  Do not instantiate directly, use the + operator on your existing
  Placeholder objects instead.
  """

  def __init__(self, items: list[Union[str, Placeholder]]):
    super().__init__(expected_type=str)
    self._items = items

  def __add__(self, right: Union[str, Placeholder]) -> _ConcatOperator:
    return _ConcatOperator(self._items + [right])

  def __radd__(self, left: str) -> _ConcatOperator:
    return _ConcatOperator([left] + self._items)

  def internal_equals(self, other: Placeholder) -> bool:
    return (
        isinstance(other, _ConcatOperator)
        and len(self._items) == len(other._items)  # pylint: disable=protected-access
        and all(
            internal_equals_value_like(item, other_item)
            for item, other_item in zip(self._items, other._items)  # pylint: disable=protected-access
        )
    )

  def encode(
      self, component_spec: Optional[type['types.ComponentSpec']] = None
  ) -> placeholder_pb2.PlaceholderExpression:
    result = placeholder_pb2.PlaceholderExpression()
    result.operator.concat_op.expressions.extend(
        [encode_value_like(item) for item in self._items]
    )
    return result

  def traverse(self) -> Iterator[Placeholder]:
    yield self
    for item in self._items:
      if isinstance(item, Placeholder):
        yield from item.traverse()


class _JoinPathOperator(Placeholder):
  """JoinPath Operator runs os.path.join() on the given arguments.

  Do not instantiate directly, use ph.join_path() instead.
  """

  def __init__(
      self,
      *args: str | Placeholder,
  ):
    super().__init__(expected_type=str)
    self._args = args

  def internal_equals(self, other: Placeholder) -> bool:
    return (
        isinstance(other, _JoinPathOperator)
        and len(self._args) == len(other._args)  # pylint: disable=protected-access
        and all(
            internal_equals_value_like(arg, other_arg)
            for arg, other_arg in zip(self._args, other._args)  # pylint: disable=protected-access
        )
    )

  def traverse(self) -> Iterator[Placeholder]:
    yield self
    for arg in self._args:
      if isinstance(arg, Placeholder):
        yield from arg.traverse()

  def encode(
      self, component_spec: Optional[type['types.ComponentSpec']] = None
  ) -> placeholder_pb2.PlaceholderExpression:
    result = placeholder_pb2.PlaceholderExpression()
    op = result.operator.join_path_op
    for arg in self._args:
      op.expressions.append(encode_value_like(arg, component_spec))
    return result


class _ProtoOperator(UnaryPlaceholderOperator):
  """Proto Operator helps access/serialize a proto-valued placeholder.

  Do not instantiate directly, use the . operator on your existing
  Placeholder objects and/or the Placeholder.serialize() function instead.
  """

  def __init__(
      self,
      value: Placeholder,
      proto_field_path: list[str],
      serialization_format: Optional[ProtoSerializationFormat] = None,
  ):
    assert value._is_maybe_proto_valued()
    super().__init__(
        value,
        expected_type=(
            Union[ValueType, list, message.Message]
            if serialization_format is None
            else str
        ),
    )
    self._proto_field_path = proto_field_path
    self._serialization_format = serialization_format

  def __getitem__(self, key: Union[int, str]) -> Placeholder:
    if self._serialization_format is not None:
      raise IndexError(f'Cannot access index {key} on serialized proto')
    if isinstance(key, str):
      return _ProtoOperator(
          self._value, proto_field_path=self._proto_field_path + [f'[{key!r}]']
      )
    return _IndexOperator(self, key, is_proto=True)

  def __getattr__(self, field_name: str) -> Placeholder:
    if field_name.startswith('__'):  # Don't mess with magic functions.
      return super().__getattr__(field_name)
    if self._serialization_format is not None:
      raise IndexError(f'Cannot access attr {field_name} on serialized proto')
    return _ProtoOperator(
        self._value,
        proto_field_path=self._proto_field_path + [f'.{field_name}'],
    )

  def internal_equals(self, other: Placeholder) -> bool:
    return (
        isinstance(other, _ProtoOperator)
        and self._proto_field_path == other._proto_field_path  # pylint: disable=protected-access
        and self._serialization_format == other._serialization_format  # pylint: disable=protected-access
        and self._value.internal_equals(other._value)  # pylint: disable=protected-access
    )

  def encode(
      self, component_spec: Optional[type['types.ComponentSpec']] = None
  ) -> placeholder_pb2.PlaceholderExpression:
    result = placeholder_pb2.PlaceholderExpression()
    op = result.operator.proto_op
    encoded_value = self._value.encode(component_spec)
    op.expression.CopyFrom(encoded_value)

    if self._proto_field_path:
      op.proto_field_path.extend(self._proto_field_path)
    if self._serialization_format:
      op.serialization_format = self._serialization_format.value

    # Attach proto descriptor if available through component spec.
    if component_spec and (
        encoded_value.placeholder.type
        == placeholder_pb2.Placeholder.EXEC_PROPERTY
    ):
      exec_property_name = encoded_value.placeholder.key
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
            f'exec property. Got {execution_param.type}.'
        )
      op.proto_schema.message_type = execution_param.type.DESCRIPTOR.full_name
      proto_utils.build_file_descriptor_set(
          execution_param.type, op.proto_schema.file_descriptors
      )

    return result


def dirname(
    placeholder: Placeholder,
) -> _DirNameOperator:
  """Runs os.path.dirname() on the path resolved from the input placeholder.

  Args:
    placeholder: Another placeholder to be wrapped in a _DirNameOperator.

  Example:
  ```
  ph.dirname(ph.execution_invocation().output_metadata_uri)
  ```

  Returns:
    A _DirNameOperator operator.
  """
  return _DirNameOperator(placeholder)


class _ListSerializationOperator(UnaryPlaceholderOperator):
  """ListSerializationOperator serializes list type placeholder.

  Do not instantiate directly, use Placeholder.serialize_list() on your existing
  list-valued Placeholder instead.
  """

  def __init__(
      self, value: Placeholder, serialization_format: ListSerializationFormat
  ):
    super().__init__(value, expected_type=str)
    self._serialization_format = serialization_format

  def encode(
      self, component_spec: Optional[type['types.ComponentSpec']] = None
  ) -> placeholder_pb2.PlaceholderExpression:
    result = placeholder_pb2.PlaceholderExpression()
    op = result.operator.list_serialization_op
    op.expression.CopyFrom(self._value.encode(component_spec))
    op.serialization_format = self._serialization_format.value
    return result


class _Base64EncodeOperator(UnaryPlaceholderOperator):
  """Base64EncodeOperator encodes another placeholder using url safe base64.

  Do not instantiate directly, use Placeholder.b64encode() on your existing
  Placeholder object instead.
  """

  def __init__(self, value: Placeholder, url_safe: bool):
    super().__init__(value, expected_type=str)
    self._url_safe = url_safe

  def encode(
      self, component_spec: Optional[type['types.ComponentSpec']] = None
  ) -> placeholder_pb2.PlaceholderExpression:
    result = placeholder_pb2.PlaceholderExpression()
    result.operator.base64_encode_op.expression.CopyFrom(
        self._value.encode(component_spec)
    )
    result.operator.base64_encode_op.is_standard_b64 = not self._url_safe
    return result


class _CompareOp(enum.Enum):
  """An alias for placeholder_pb2.ComparisonOperator.Operation."""

  EQUAL = placeholder_pb2.ComparisonOperator.Operation.EQUAL
  LESS_THAN = placeholder_pb2.ComparisonOperator.Operation.LESS_THAN
  GREATER_THAN = placeholder_pb2.ComparisonOperator.Operation.GREATER_THAN


class _DirNameOperator(UnaryPlaceholderOperator):
  """_DirNameOperator returns directory path given a path."""

  def __init__(
      self,
      value: Placeholder,
  ):
    super().__init__(
        value,
        expected_type=str,
    )

  def encode(
      self, component_spec: Optional[type['types.ComponentSpec']] = None
  ) -> placeholder_pb2.PlaceholderExpression:
    result = placeholder_pb2.PlaceholderExpression()
    op = result.operator.dir_name_op
    op.expression.CopyFrom(self._value.encode(component_spec))

    return result


def internal_equals_value_like(
    a: Optional[ValueLikeType], b: Optional[ValueLikeType]
) -> bool:
  """Equality operator for Placeholders or primitives."""
  if isinstance(a, Placeholder):
    return a.internal_equals(b)
  if isinstance(b, Placeholder):
    return False
  return a == b


def encode_value_like(
    x: ValueLikeType, component_spec: Any = None
) -> placeholder_pb2.PlaceholderExpression:
  """Encodes x to a placeholder expression proto."""

  if isinstance(x, Placeholder):
    return x.encode(component_spec)
  result = placeholder_pb2.PlaceholderExpression()
  if isinstance(x, bool):
    result.value.bool_value = x
  elif isinstance(x, int):
    result.value.int_value = x
  elif isinstance(x, float):
    result.value.double_value = x
  elif isinstance(x, str):
    result.value.string_value = x
  else:
    raise ValueError(f'x must be an int, float, str, or Placeholder. x: {x}')
  return result


@attr.define(eq=False)
class _ComparisonPredicate(Predicate):
  """Represents a comparison between two placeholders."""

  compare_op: _CompareOp
  left: ValueLikeType
  right: ValueLikeType

  def encode(
      self, component_spec: Any = None
  ) -> placeholder_pb2.PlaceholderExpression:
    result = placeholder_pb2.PlaceholderExpression()
    result.operator.compare_op.op = self.compare_op.value
    result.operator.compare_op.lhs.CopyFrom(
        encode_value_like(self.left, component_spec)
    )
    result.operator.compare_op.rhs.CopyFrom(
        encode_value_like(self.right, component_spec)
    )
    return result

  def internal_equals(self, other: Placeholder) -> bool:
    return (
        isinstance(other, _ComparisonPredicate)
        and self.compare_op == other.compare_op
        and internal_equals_value_like(self.left, other.left)
        and internal_equals_value_like(self.right, other.right)
    )

  def traverse(self) -> Iterator[Placeholder]:
    yield self
    if isinstance(self.left, Placeholder):
      yield from self.left.traverse()
    if isinstance(self.right, Placeholder):
      yield from self.right.traverse()


@attr.define(eq=False)
class _NotPredicate(Predicate):
  """Represents a logical negation."""

  value: Predicate

  def encode(
      self, component_spec: Any = None
  ) -> placeholder_pb2.PlaceholderExpression:
    if isinstance(self.value, _NotPredicate):  # not(not(foo)) becomes foo
      return self.value.value.encode(component_spec)
    result = placeholder_pb2.PlaceholderExpression()
    result.operator.unary_logical_op.op = (
        placeholder_pb2.UnaryLogicalOperator.Operation.NOT
    )
    result.operator.unary_logical_op.expression.CopyFrom(
        self.value.encode(component_spec)
    )
    return result

  def internal_equals(self, other: Placeholder) -> bool:
    return isinstance(other, _NotPredicate) and self.value.internal_equals(
        other.value
    )

  def traverse(self) -> Iterator[Placeholder]:
    yield self
    yield from self.value.traverse()


@attr.define(eq=False)
class _BinaryLogicalPredicate(Predicate):
  """Represents a boolean logical expression with exactly two arguments."""

  logical_op: placeholder_pb2.BinaryLogicalOperator.Operation
  left: Predicate
  right: Predicate

  def encode(
      self, component_spec: Any = None
  ) -> placeholder_pb2.PlaceholderExpression:
    result = placeholder_pb2.PlaceholderExpression()
    result.operator.binary_logical_op.op = self.logical_op
    result.operator.binary_logical_op.lhs.CopyFrom(
        self.left.encode(component_spec)
    )
    result.operator.binary_logical_op.rhs.CopyFrom(
        self.right.encode(component_spec)
    )
    return result

  def internal_equals(self, other: Placeholder) -> bool:
    return (
        isinstance(other, _BinaryLogicalPredicate)
        and self.logical_op == other.logical_op
        and self.left.internal_equals(other.left)
        and self.right.internal_equals(other.right)
    )

  def traverse(self) -> Iterator[Placeholder]:
    yield self
    yield from self.left.traverse()
    yield from self.right.traverse()
