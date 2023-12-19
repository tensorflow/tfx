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
"""Placeholders that resolve to entire proto messages."""

from __future__ import annotations

import functools
from typing import Callable, Dict, Generic, Iterator, List, Optional, Tuple, Type, TypeVar, Union

from tfx.dsl.placeholder import placeholder_base
from tfx.proto.orchestration import placeholder_pb2
from tfx.utils import proto_utils

from google.protobuf import any_pb2
from google.protobuf import descriptor as descriptor_lib
from google.protobuf import message

_types = placeholder_base.types
_T = TypeVar('_T', bound=message.Message)
_T2 = TypeVar('_T2', bound=message.Message)


def create_proto(
    message_type: Type[_T],
) -> Callable[..., CreateProtoPlaceholder[_T]]:
  """Returns a proto placeholder factory.

  Basic usage:
  ```python
  flags=[
      (
          'my_proto_flag',
          ph.create_proto(MyProtoType)(
              field1='a plain value',
              field2=ph.execution_invocation().pipeline_run_id,
              field3=ph.create_proto(MySubProtoType)(
                  inner_field1=ph.input('foo').uri + '/somefile.txt',
                  ...
              ),
              ...
          ).serialize(ph.ProtoSerializationFormat.TEXT_FORMAT)
      ),
  ]
  ```

  That is, you call `ph.create_proto(OriginalType)` to receive a new function
  that takes the same arguments as the `OriginalType` initializer, except that
  arguments can also contain placeholders.

  It's important to note that proto creation placeholders are only allowed as
  sub-expressions. That is, you cannot pass their output directly to a flag.
  Instead, you always need to serialize the proto in some way, so that it
  becomes a string.

  Advanced usage (assigning factories to global variables, using a base proto):
  ```python
  _MyProtoType = ph.create_proto(my_pb2.MyProtoType)
  _MySubProtoType = ph.create_proto(my_pb2.MySubProtoType)

  my_base_proto = my_pb2.MyProtoType(field1='a plain value', ...)
  flags=[
      (
          'my_proto_flag',
          _MyProtoType(
              my_base_proto,
              field2=ph.execution_invocation().pipeline_run_id,
              repeated_field3=[
                  _MySubProtoType(inner_field1='one'),
                  _MySubProtoType(inner_field1='two'),
                  my_pb2.MySubProtoType(inner_field1='last')
                  ...
              ],
              ...
          ).serialize(ph.ProtoSerializationFormat.TEXT_FORMAT)
      ),
  ]
  ```

  Limitations:
  * Map fields only support string/placeholder keys, not integral types.
  * MessageSet is not yet supported.
  * Proto extension fields are not supported.
  * `bytes` fields can only populated through Python `str` values, so their
    contents must be valid strings and can't contain arbitrary bytes.
  * All created protos are final. You can use them as a submessage inside
    another ph.create_proto() expression, but you cannot update field values
    after receiving the instance from the factory. (And you shouldn't need to.)

  Args:
    message_type: The proto type that the created placeholders resolve to.

  Returns:
    A function that returns a placeholder that, at runtime, will evaluate to a
    proto message of the given `message_type`. The arguments to this function
    are (all optional):
    * A single positional argument with a plain message of the given
      `message_type`. This is taken as the base of the resulting proto and
      defaults to the default/empty message for that type. The fields passed as
      kwargs (explained below) are merged on top of this base message.
    * A series of keyword arguments, which correspond to fields of the proto
      type and serve to populate the resulting message's fields. Just like when
      constructing regular protos, repeated fields must be passed as lists and
      map fields must be passed either as dicts (if all keys are plain strings)
      or as lists of (key,value) tuples if some of the keys are placeholders.
      In all cases, the values can be placeholders or plain values (strings,
      protos) matching the respective field type. In particular, other
      CreateProtoPlaceholder instances can be passed to populate sub-message
      fields.
    You may cache the returned factory function, use it multiple times, or even
    assign it to a global variable for convenient reuse.
  """
  return functools.partial(CreateProtoPlaceholder, message_type)


# These are for the inner values of a field (whether repeated or not):
_PlainValues = Union[  # The values we use internally.
    placeholder_base.ValueType,  # explicit value
    placeholder_base.Placeholder,  # placeholdered value
]
_InputValues = Union[  # The values users may pass (we convert to _PlainValues).
    _PlainValues,
    message.Message,  # Users may pass in submessage fields as a plain proto.
    None,  # Users may pass None to optional fields.
]
# These are for the outer values of a field (so repeated is a list):
_FieldValues = Union[  # The values we use internally.
    _PlainValues,  # singular (optional or required) field
    placeholder_base.ListPlaceholder,  # repeated field
    placeholder_base.DictPlaceholder,  # map field
]
_InputFieldValues = Union[  # The values users may pass.
    _InputValues,  # singular (optional or required) field
    List[_InputValues],  # repeated field
    Dict[str, _InputValues],  # map field with plain keys
    List[Tuple[Union[str, placeholder_base.Placeholder], _InputValues]],  # map
    None,  # Users may pass None to optional fields.
]
_PROTO_TO_PY_TYPE = {
    # Note: The Python types chosen here must be representable in a
    # ml_metadata.Value proto. So in particular: no `bytes`.
    descriptor_lib.FieldDescriptor.TYPE_BYTES: str,
    descriptor_lib.FieldDescriptor.TYPE_STRING: str,
    descriptor_lib.FieldDescriptor.TYPE_FLOAT: float,
    descriptor_lib.FieldDescriptor.TYPE_DOUBLE: float,
    descriptor_lib.FieldDescriptor.TYPE_BOOL: bool,
    descriptor_lib.FieldDescriptor.TYPE_INT32: int,
    descriptor_lib.FieldDescriptor.TYPE_INT64: int,
    descriptor_lib.FieldDescriptor.TYPE_UINT32: int,
    descriptor_lib.FieldDescriptor.TYPE_UINT64: int,
    descriptor_lib.FieldDescriptor.TYPE_SINT32: int,
    descriptor_lib.FieldDescriptor.TYPE_SINT64: int,
    descriptor_lib.FieldDescriptor.TYPE_FIXED32: int,
    descriptor_lib.FieldDescriptor.TYPE_FIXED64: int,
    descriptor_lib.FieldDescriptor.TYPE_SFIXED32: int,
    descriptor_lib.FieldDescriptor.TYPE_SFIXED64: int,
    descriptor_lib.FieldDescriptor.TYPE_ENUM: int,
}


class CreateProtoPlaceholder(Generic[_T], placeholder_base.Placeholder):
  """A placeholder that evaluates to a proto message."""

  def __init__(
      self,
      message_type: Type[_T],
      base: Optional[_T] = None,
      **kwargs: _InputFieldValues,
  ):
    """Initializes the class. Consider this private."""
    super().__init__(expected_type=message_type)
    self._message_type = message_type
    self._base = base or message_type()
    self._fields = {}
    for key, value in kwargs.items():
      value = self._validate_and_transform_field(key, value)
      if value is not None:
        self._fields[key] = value

  def _validate_and_transform_field(
      self, field: str, value: _InputFieldValues
  ) -> Optional[_FieldValues]:
    """Validates the given value and transforms it to what encode() needs."""
    message_name = self._message_type.DESCRIPTOR.full_name
    if field not in self._base.DESCRIPTOR.fields_by_name:
      raise ValueError(f'Unknown field {field} for proto {message_name}.')
    descriptor: descriptor_lib.FieldDescriptor = (
        self._base.DESCRIPTOR.fields_by_name[field]
    )
    field_name = f'{message_name}.{descriptor.name}'

    if (  # If it's a map<> field.
        descriptor.message_type
        and descriptor.message_type.has_options
        and descriptor.message_type.GetOptions().map_entry
    ):
      if value is None:
        return None
      if isinstance(value, dict):
        value = value.items()
      elif not isinstance(value, list):
        raise ValueError(
            'Expected dict[k,v] or list[tuple[k, v]] input for map field '
            f'{field_name}, got {value!r}.'
        )
      entries: List[
          Tuple[Union[str, placeholder_base.Placeholder], _PlainValues]
      ] = []
      for entry_key, entry_value in value:
        if not isinstance(
            entry_key, (str, placeholder_base.Placeholder)
        ) or isinstance(
            entry_key,
            (
                CreateProtoPlaceholder,
                placeholder_base.ListPlaceholder,
                placeholder_base.DictPlaceholder,
            ),
        ):
          raise ValueError(
              'Expected string (placeholder) for dict key of map field '
              f'{field_name}, got {entry_key!r}.'
          )
        value_descriptor = descriptor.message_type.fields_by_name['value']
        entry_value = self._validate_and_transform_value(
            f'{field_name}.value', value_descriptor, entry_value
        )
        if entry_value is not None:
          entries.append((entry_key, entry_value))
      return placeholder_base.to_dict(entries)
    elif descriptor.label == descriptor_lib.FieldDescriptor.LABEL_REPEATED:
      if value is None or isinstance(value, placeholder_base.Placeholder):
        return value  # pytype: disable=bad-return-type
      if not isinstance(value, list):
        raise ValueError(
            f'Expected list input for repeated field {field_name}, got '
            f'{value!r}.'
        )
      items: List[_PlainValues] = []
      for item in value:
        item = self._validate_and_transform_value(field_name, descriptor, item)
        if item is not None:
          items.append(item)
      return placeholder_base.to_list(items)
    else:
      return self._validate_and_transform_value(field_name, descriptor, value)

  def _validate_and_transform_value(
      self,
      field_name: str,
      descriptor: descriptor_lib.FieldDescriptor,
      value: _InputValues,
  ) -> Optional[_PlainValues]:
    if value is None:
      if descriptor.label == descriptor_lib.FieldDescriptor.LABEL_OPTIONAL:
        return None
      raise ValueError(
          f'Expected value for non-optional field {field_name}, got None.'
      )

    # Deal with sub-message fields first.
    if descriptor.type == descriptor_lib.FieldDescriptor.TYPE_MESSAGE:
      if isinstance(value, message.Message):
        value = CreateProtoPlaceholder(type(value), value)
      elif (
          not isinstance(value, placeholder_base.Placeholder)
          or not value._is_maybe_proto_valued()  # pylint: disable=protected-access
      ):
        raise ValueError(
            f'Expected submessage proto or placeholder for field {field_name}, '
            f'got {value!r}.'
        )

      # Some best-effort validation for the proto type.
      submsg_type = value.expected_type
      if isinstance(submsg_type, type) and issubclass(
          submsg_type, message.Message
      ):
        # The proto placeholder knows exactly which proto type it will resolve
        # to. So we can verify that it's the right one.
        if descriptor.message_type not in (
            submsg_type.DESCRIPTOR,
            any_pb2.Any.DESCRIPTOR,
        ):
          raise ValueError(
              f'Expected message of type {descriptor.message_type.full_name} '
              f'for field {field_name}, got {submsg_type.DESCRIPTOR.full_name}.'
          )
      return value

    # Now we know it's a scalar field.
    if isinstance(value, (message.Message, CreateProtoPlaceholder)):
      raise ValueError(
          f'Expected scalar value for field {field_name}, got {value!r}.'
      )
    if descriptor.type not in _PROTO_TO_PY_TYPE:
      raise ValueError(
          f'Unsupported proto field type {descriptor.type} on {field_name}.'
      )
    expected_type = _PROTO_TO_PY_TYPE[descriptor.type]
    if not isinstance(value, (expected_type, placeholder_base.Placeholder)):
      raise ValueError(
          f'Expected {expected_type} for {field_name}, got {value!r}.'
      )
    return value

  def traverse(self) -> Iterator[placeholder_base.Placeholder]:
    """Yields all placeholders under and including this one."""
    yield from super().traverse()
    for value in self._fields.values():
      if isinstance(value, placeholder_base.Placeholder):
        yield value

  def _clear_sub_descriptors(
      self, placeholder: placeholder_pb2.PlaceholderExpression
  ) -> None:
    """Clears the descriptors set on sub-messages."""
    operator_type = placeholder.operator.WhichOneof('operator_type')
    if operator_type == 'create_proto_op':
      placeholder.operator.create_proto_op.ClearField('file_descriptors')
    elif operator_type == 'list_concat_op':
      for input_placeholder in placeholder.operator.list_concat_op.expressions:
        self._clear_sub_descriptors(input_placeholder)
    elif operator_type == 'create_dict_op':
      for entry in placeholder.operator.create_dict_op.entries:
        self._clear_sub_descriptors(entry.key)
        self._clear_sub_descriptors(entry.value)

  def encode(
      self, component_spec: Optional[Type['_types.ComponentSpec']] = None
  ) -> placeholder_pb2.PlaceholderExpression:
    result = placeholder_pb2.PlaceholderExpression()
    op = result.operator.create_proto_op
    op.base.Pack(self._base)
    proto_utils.build_file_descriptor_set(self._base, op.file_descriptors)

    for key, value in self._fields.items():
      op.fields[key].MergeFrom(
          placeholder_base.encode_value_like(value, component_spec)
      )

      # Clear the descriptors set on sub-messages, as we add our own descriptor
      # above and that transitively contains all the sub-field descriptors.
      for sub_expression in op.fields.values():
        self._clear_sub_descriptors(sub_expression)

    return result
