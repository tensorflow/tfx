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

from typing import Dict, Generic, Iterator, Mapping, Optional, TypeVar, Union

from tfx.dsl.placeholder import placeholder_base
from tfx.proto.orchestration import placeholder_pb2
from tfx.utils import proto_utils

from google.protobuf import any_pb2
from google.protobuf import descriptor as descriptor_lib
from google.protobuf import message
from google.protobuf import message_factory

_types = placeholder_base.types
_T = TypeVar('_T', bound=message.Message)
_T2 = TypeVar('_T2', bound=message.Message)


def make_proto(
    base_message: _T,
    **kwargs: _InputFieldValues,
) -> MakeProtoPlaceholder[_T]:
  """Returns a placeholder that resolves to a proto with the given fields.

  Basic usage:
  ```python
  flags=[
      (
          'my_proto_flag',
          ph.make_proto(
              MyProtoType(),
              field1='a plain value',
              field2=ph.execution_invocation().pipeline_run_id,
              field3=ph.make_proto(
                  MySubProtoType(inner_field1='fixed value'),
                  inner_field2=ph.input('foo').uri + '/somefile.txt',
                  ...
              ),
              ...
          ).serialize(ph.ProtoSerializationFormat.TEXT_FORMAT)
      ),
  ]
  ```

  It's important to note that proto creation placeholders are only allowed as
  sub-expressions. That is, you cannot pass their output directly to a flag.
  Instead, you always need to serialize the proto in some way, so that it
  becomes a string.

  Limitations:
  * Map fields only support string/placeholder keys, not integral types.
  * MessageSet is not yet supported.
  * Proto extension fields are not supported.
  * `bytes` fields can only populated through Python `str` values, so their
    contents must be valid strings and can't contain arbitrary bytes.
  * All constructed protos are final. You can use them as a submessage inside
    another ph.make_proto() expression, but you cannot update field values
    after receiving the instance from the factory. (And you shouldn't need to.)

  Args:
    base_message: An instance of the proto type that the constructed placeholder
      resolves to. This can already have some fields populated, which will be
      passed through to the output, though of course those can't contain any
      placeholders.
    **kwargs: Additional fields to populate in the output proto message, whereby
      the values may contain placeholders. These fields are merged on top of the
      fields present already in the `base_message`. Just like when constructing
      regular protos, repeated fields must be passed as lists and map fields
      must be passed either as dicts (if all keys are plain strings) or as lists
      of (key,value) tuples if some of the keys are placeholders. In all cases,
      the values can be placeholders or plain values (strings, protos) matching
      the respective field type. In particular, other MakeProtoPlaceholder
      instances can be passed to populate sub-message fields.

  Returns:
    A placeholder that, at runtime, will evaluate to a proto message of the
    same type as the `base_message`. It will have the `base_message`'s fields
    populated, but with the `kwargs` fields merged on top.
  """
  return MakeProtoPlaceholder(base_message, **kwargs)


# These are for the inner values of a field (whether repeated or not):
_InputValues = Union[  # The values users may supply to us.
    placeholder_base.ValueLikeType,
    message.Message,  # Users may pass in submessage fields as a plain proto ...
    Dict[str, '_InputValues'],  # ... or as a Python dict.
    None,  # Users may pass None to optional fields.
]
# These are for the outer values of a field (so repeated is a list):
_InputFieldValues = Union[  # The values users may pass.
    _InputValues,  # singular (optional or required) field
    list[_InputValues],  # repeated field
    dict[str, _InputValues],  # map field with plain keys
    list[tuple[Union[str, placeholder_base.Placeholder], _InputValues]],  # map
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


class MakeProtoPlaceholder(Generic[_T], placeholder_base.Placeholder):
  """A placeholder that evaluates to a proto message."""

  def __init__(
      self,
      base_message: _T,
      **kwargs: _InputFieldValues,
  ):
    """Initializes the class. Consider this private."""
    super().__init__(expected_type=type(base_message))
    self._base_message = base_message
    self._fields: dict[str, placeholder_base.ValueLikeType] = {}
    for key, value in kwargs.items():
      value = self._validate_and_transform_field(key, value)
      if value is not None:
        self._fields[key] = value

  def _validate_and_transform_field(
      self, field: str, value: _InputFieldValues
  ) -> Optional[placeholder_base.ValueLikeType]:
    """Validates the given value and transforms it to what encode() needs."""
    message_name = self._base_message.DESCRIPTOR.full_name
    if field not in self._base_message.DESCRIPTOR.fields_by_name:
      raise ValueError(f'Unknown field {field} for proto {message_name}.')
    descriptor: descriptor_lib.FieldDescriptor = (
        self._base_message.DESCRIPTOR.fields_by_name[field]
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
      entries: list[
          tuple[
              Union[str, placeholder_base.Placeholder],
              placeholder_base.ValueLikeType,
          ]
      ] = []
      for entry_key, entry_value in value:
        if not isinstance(
            entry_key, (str, placeholder_base.Placeholder)
        ) or isinstance(
            entry_key,
            (
                MakeProtoPlaceholder,
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
      return placeholder_base.make_dict(entries)
    elif descriptor.label == descriptor_lib.FieldDescriptor.LABEL_REPEATED:
      if value is None or isinstance(value, placeholder_base.Placeholder):
        return value  # pytype: disable=bad-return-type
      if not isinstance(value, list):
        raise ValueError(
            f'Expected list input for repeated field {field_name}, got '
            f'{value!r}.'
        )
      items: list[placeholder_base.ValueLikeType] = []
      for item in value:
        item = self._validate_and_transform_value(field_name, descriptor, item)
        if item is not None:
          items.append(item)
      return placeholder_base.make_list(items)
    else:
      return self._validate_and_transform_value(field_name, descriptor, value)

  def _validate_and_transform_value(
      self,
      field_name: str,
      descriptor: descriptor_lib.FieldDescriptor,
      value: _InputValues,
  ) -> Optional[placeholder_base.ValueLikeType]:
    if value is None:
      if descriptor.label == descriptor_lib.FieldDescriptor.LABEL_OPTIONAL:
        return None
      raise ValueError(
          f'Expected value for non-optional field {field_name}, got None.'
      )

    # Deal with sub-message fields first.
    if descriptor.type == descriptor_lib.FieldDescriptor.TYPE_MESSAGE:
      if isinstance(value, message.Message):
        value = MakeProtoPlaceholder(value)
      elif isinstance(value, Mapping):
        value = MakeProtoPlaceholder(
            # TODO(b/323991103):
            # Switch to using the message_factory.GetMessageClass() function.
            # See http://yaqs/3936732114019418112 for more context.
            message_factory.MessageFactory().GetPrototype(
                descriptor.message_type
            )(**value)
        )
      elif not isinstance(value, MakeProtoPlaceholder):
        raise ValueError(
            'Expected submessage proto or another make_proto() placeholder '
            f'for field {field_name}, got {value!r}.'
        )

      # Validate that the sub-proto type matches the field type.
      submsg_type = value.expected_type
      assert isinstance(submsg_type, type)
      assert issubclass(submsg_type, message.Message)
      if descriptor.message_type.full_name not in (
          submsg_type.DESCRIPTOR.full_name,
          any_pb2.Any.DESCRIPTOR.full_name,
      ):
        raise ValueError(
            f'Expected message of type {descriptor.message_type.full_name} '
            f'for field {field_name}, got {submsg_type.DESCRIPTOR.full_name}.'
        )
      return value

    # Now we know it's a scalar field.
    if isinstance(value, (message.Message, MakeProtoPlaceholder)):
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
    return value  # pytype: disable=bad-return-type

  def traverse(self) -> Iterator[placeholder_base.Placeholder]:
    """Yields all placeholders under and including this one."""
    yield from super().traverse()
    for value in self._fields.values():
      if isinstance(value, placeholder_base.Placeholder):
        yield from value.traverse()

  def _lift_up_descriptors(
      self, op: placeholder_pb2.MakeProtoOperator
  ) -> None:
    """Moves+deduplicates descriptors from sub-messages to the given `op`."""
    known_descriptors = {fd.name for fd in op.file_descriptors.file}
    for field_value in op.fields.values():
      operator_type = field_value.operator.WhichOneof('operator_type')
      if operator_type == 'list_concat_op':
        sub_expressions = field_value.operator.list_concat_op.expressions
      elif operator_type == 'make_dict_op':
        entries = field_value.operator.make_dict_op.entries
        sub_expressions = [entry.key for entry in entries] + [
            entry.value for entry in entries
        ]
      else:
        sub_expressions = [field_value]
      for sub_expression in sub_expressions:
        if (
            sub_expression.operator.WhichOneof('operator_type')
            == 'make_proto_op'
        ):
          sub_op = sub_expression.operator.make_proto_op
          for fd in sub_op.file_descriptors.file:
            if fd.name not in known_descriptors:
              known_descriptors.add(fd.name)
              op.file_descriptors.file.append(fd)
          sub_op.ClearField('file_descriptors')

  def encode(
      self, component_spec: Optional[type['_types.ComponentSpec']] = None
  ) -> placeholder_pb2.PlaceholderExpression:
    result = placeholder_pb2.PlaceholderExpression()
    op = result.operator.make_proto_op
    op.base.Pack(self._base_message)
    proto_utils.build_file_descriptor_set(
        self._base_message, op.file_descriptors
    )

    for key, value in self._fields.items():
      op.fields[key].MergeFrom(
          placeholder_base.encode_value_like(value, component_spec)
      )

    self._lift_up_descriptors(op)

    return result
