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

import collections
from typing import Callable, Dict, Generic, Iterable, Iterator, Mapping, MutableSequence, Optional, Sequence, TypeVar, Union

from tfx.dsl.placeholder import placeholder_base
from tfx.proto.orchestration import placeholder_pb2
from tfx.utils import proto_utils

from google.protobuf import any_pb2
from google.protobuf import descriptor_pb2
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


_E = TypeVar('_E')


def _remove_unless(
    container: MutableSequence[_E], condition: Callable[[_E], bool]
) -> None:
  """yaqs/5214174899863552#a5707702298738688n5649050225344512 in a function."""
  keep_items = [item for item in container if condition(item)]
  del container[:]
  container.extend(keep_items)


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

    self._descriptor_collector: Optional[_DescriptorCollector] = None

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

  def encode(
      self, component_spec: Optional[type['_types.ComponentSpec']] = None
  ) -> placeholder_pb2.PlaceholderExpression:
    # In a tree of MakeProtoPlaceholder.encode() calls, only the root will
    # create a _DescriptorCollector(). This will cause all of the sub-calls to
    # send their descriptors there and _not_ write them to their output
    # PlaceholderExpression.
    descriptor_collector = None  # Populated only in the root.
    if self._descriptor_collector is None:
      descriptor_collector = _DescriptorCollector()
      for p in self.traverse():
        if isinstance(p, MakeProtoPlaceholder):
          p._descriptor_collector = descriptor_collector  # pylint: disable=protected-access
    assert self._descriptor_collector is not None

    result = placeholder_pb2.PlaceholderExpression()
    op = result.operator.make_proto_op
    op.base.Pack(self._base_message)
    for key, value in self._fields.items():
      op.fields[key].MergeFrom(
          placeholder_base.encode_value_like(value, component_spec)
      )

    self._descriptor_collector.add(self._base_message, self._fields.keys())
    if descriptor_collector is not None:
      # This is the root, so emit all the descriptors.
      descriptor_collector.build(op.file_descriptors)
      for p in self.traverse():
        if isinstance(p, MakeProtoPlaceholder):
          p._descriptor_collector = None  # pylint: disable=protected-access

    return result


class _DescriptorCollector:
  """Collects and shrinks proto descriptors for nested make_proto operators."""

  def __init__(self):
    # All files from which we potentially need to include descriptors into the
    # final placeholder IR. It's important that this dict is insertion-ordered,
    # so that it doesn't destroy the order from gather_file_descriptors(). Every
    # dependent file must be processed after its dependencies.
    self.descriptor_files: collections.OrderedDict[
        str, descriptor_lib.FileDescriptor
    ] = collections.OrderedDict()
    # Fully-qualified names of the proto messages/enums whose descriptors we
    # need to keep, because (a) they're the type being constructed by the
    # placeholder, or (b) any of the sub-messages, or (c) any of their nested
    # messages/enum declarations are needed. Crucially, we need to keep a type
    # even if none of its fields occur in `_keep_fields`, in case the user wants
    # to create an empty proto of that type.
    self._keep_types: set[str] = set()
    # Fully-qualified names of fields ("<message_fqn>.<field_name>") we need to
    # keep, because they occur in a base message or as a placeholder field.
    self._keep_fields: set[str] = set()

  def add(self, base_message: message.Message, fields: Iterable[str]) -> None:
    self._collect_from_message(base_message)
    msg_name = base_message.DESCRIPTOR.full_name
    self._keep_fields.update({f'{msg_name}.{field}' for field in fields})

    root_file = base_message.DESCRIPTOR.file
    if root_file.name in self.descriptor_files:
      return
    for fd in proto_utils.gather_file_descriptors(root_file):
      if fd.name not in self.descriptor_files:
        self.descriptor_files[fd.name] = fd

  def _collect_from_message(self, msg: message.Message) -> None:
    """Marks this message and all fields and submessages to be kept."""
    msg_name = msg.DESCRIPTOR.full_name
    self._keep_types.add(msg_name)
    for field, value in msg.ListFields():
      self._keep_fields.add(f'{msg_name}.{field.name}')
      if isinstance(value, message.Message):
        self._collect_from_message(value)
      elif isinstance(value, Sequence):
        for item in value:
          if isinstance(item, message.Message):
            self._collect_from_message(item)
      elif isinstance(value, Mapping):
        self._keep_fields.update({
            f'{field.message_type.full_name}.key',
            f'{field.message_type.full_name}.value',
        })
        for item in value.values():
          if isinstance(item, message.Message):
            self._collect_from_message(item)

  def _shrink_descriptors(self, fds: descriptor_pb2.FileDescriptorSet) -> None:
    """Deletes all field/message descriptors not used by this placeholder."""
    # We don't want to shrink any of the "well-known" proto types (like Any),
    # because because the proto runtime verifies that the descriptor for these
    # well-known types matches what it expects. The runtimes do this because
    # they then replace the message classes with more specific, native classes,
    # to offer APIs like `Any.Pack()`, for instance.
    well_known_types_pkg = 'google.protobuf.'

    # Step 1: Go over all the message descriptors a first time, including
    #         recursion into nested declarations. Delete field declarations we
    #         don't need. Collect target types we need because they're the value
    #         type of a field we want to keep.
    def _shrink_message(
        name_prefix: str, message_descriptor: descriptor_pb2.DescriptorProto
    ) -> None:
      msg_name = f'{name_prefix}.{message_descriptor.name}'
      if not msg_name.startswith(well_known_types_pkg):
        # Mark map<> entry key/value fields as used if the map field is used.
        if (
            message_descriptor.options.map_entry
            and msg_name in self._keep_types
        ):
          self._keep_fields.update({f'{msg_name}.key', f'{msg_name}.value'})

        # Delete unused fields.
        del message_descriptor.extension[:]  # We don't support extension fields
        _remove_unless(
            message_descriptor.field,
            lambda f: f'{msg_name}.{f.name}' in self._keep_fields,
        )

        # Clean up oneofs that have no fields left.
        i = 0
        while i < len(message_descriptor.oneof_decl):
          if all(
              not f.HasField('oneof_index') or f.oneof_index != i
              for f in message_descriptor.field
          ):
            # No references left. Delete this one and shift all indices down.
            del message_descriptor.oneof_decl[i]
            for f in message_descriptor.field:
              if f.oneof_index > i:
                f.oneof_index -= 1
          else:
            i += 1

      # Mark target types of fields as used.
      for field_descriptor in message_descriptor.field:
        if (
            field_descriptor.type
            in (
                descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
                descriptor_pb2.FieldDescriptorProto.TYPE_ENUM,
            )
            and f'{msg_name}.{field_descriptor.name}' in self._keep_fields
        ):
          assert field_descriptor.type_name.startswith('.')
          self._keep_types.add(field_descriptor.type_name.removeprefix('.'))

      # Recurse into nested message types.
      for nested_descriptor in message_descriptor.nested_type:
        _shrink_message(msg_name, nested_descriptor)

    # Outer invocation of step 1 on all files.
    for file_descriptor in fds.file:
      del file_descriptor.service[:]  # We never need RPC services.
      del file_descriptor.extension[:]  # We don't support extension fields.
      for message_descriptor in file_descriptor.message_type:
        _shrink_message(file_descriptor.package, message_descriptor)

    # Step 2: Go over all message descriptors a second time, including recursion
    #         into nested declarations. Delete any nested declarations that were
    #         not marked in the first pass. Mark any messages that have nested
    #         declarations, because runtime descriptor pools require the parent
    #         message to be present (even if unused) before allowing to add
    #         nested message.
    #         (This step is actually called within step 3.)
    def _purge_types(
        name_prefix: str, message_descriptor: descriptor_pb2.DescriptorProto
    ) -> None:
      msg_name = f'{name_prefix}.{message_descriptor.name}'
      for nested_descriptor in message_descriptor.nested_type:
        _purge_types(msg_name, nested_descriptor)
      _remove_unless(
          message_descriptor.nested_type,
          lambda n: f'{msg_name}.{n.name}' in self._keep_types,
      )
      _remove_unless(
          message_descriptor.enum_type,
          lambda e: f'{msg_name}.{e.name}' in self._keep_types,
      )
      if message_descriptor.nested_type or message_descriptor.enum_type:
        self._keep_types.add(msg_name)

    # Step 3: Remove the unused messages and enums from the file descriptors.
    for file_descriptor in fds.file:
      name_prefix = file_descriptor.package
      for message_descriptor in file_descriptor.message_type:
        _purge_types(name_prefix, message_descriptor)  # Step 2
      _remove_unless(
          file_descriptor.message_type,
          lambda m: f'{name_prefix}.{m.name}' in self._keep_types,  # pylint: disable=cell-var-from-loop
      )
      _remove_unless(
          file_descriptor.enum_type,
          lambda e: f'{name_prefix}.{e.name}' in self._keep_types,  # pylint: disable=cell-var-from-loop
      )

    # Step 4: Remove file descriptors that became empty. Remove declared
    # dependencies on other .proto files if those files were removed themselves.
    _remove_unless(fds.file, lambda fd: fd.message_type or fd.enum_type)
    keep_file_names = {fd.name for fd in fds.file}
    for fd in fds.file:
      _remove_unless(fd.dependency, lambda dep: dep in keep_file_names)
      del fd.public_dependency[:]
      del fd.weak_dependency[:]

  def build(self, result: descriptor_pb2.FileDescriptorSet) -> None:
    for fd in self.descriptor_files.values():
      fd.CopyToProto(result.file.add())
    self._shrink_descriptors(result)
