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
from typing import Optional, Type, Union, cast
from tfx import types
from tfx.proto.orchestration import placeholder_pb2
from tfx.utils import proto_utils

from google.protobuf import message


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
      component_spec: Type[types.ComponentSpec] = None
  ) -> placeholder_pb2.PlaceholderExpression:
    pass


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
      component_spec: Optional[Type[types.ComponentSpec]] = None
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
      component_spec: Optional[Type[types.ComponentSpec]] = None
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
      component_spec: Optional[Type[types.ComponentSpec]] = None
  ) -> placeholder_pb2.PlaceholderExpression:
    del component_spec  # Unused by IndexOperator

    result = placeholder_pb2.PlaceholderExpression()
    result.operator.index_op.expression.CopyFrom(sub_expression_pb)
    result.operator.index_op.index = self._index
    return result


class _ConcatOperator(_PlaceholderOperator):
  """Concat Operator concatenates multiple Placeholders.

  Prefer to use + operator overloading of Placeholder.
  """

  def __init__(self, right: Union[str, 'Placeholder'] = None, left: str = None):
    super().__init__()
    self._left = left
    self._right = right

  def encode(
      self,
      sub_expression_pb: placeholder_pb2.PlaceholderExpression,
      component_spec: Optional[Type[types.ComponentSpec]] = None
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


class ProtoSerializationFormat(enum.Enum):
  TEXT_FORMAT = placeholder_pb2.ProtoOperator.TEXT_FORMAT
  JSON = placeholder_pb2.ProtoOperator.JSON
  BINARY = placeholder_pb2.ProtoOperator.BINARY


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
      component_spec: Optional[Type[types.ComponentSpec]] = None
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
      result.operator.proto_op.proto_schema.message_type = (
          execution_param.type.DESCRIPTOR.full_name)
      fd_set = result.operator.proto_op.proto_schema.file_descriptors
      for fd in proto_utils.gather_file_descriptors(
          execution_param.type.DESCRIPTOR):
        fd.CopyToProto(fd_set.file.add())

    return result


class _Base64EncodeOperator(_PlaceholderOperator):
  """Base64EncodeOperator encodes another placeholder using url safe base64.

  Prefer to use the .b64encode method of Placeholder.
  """

  def encode(
      self,
      sub_expression_pb: placeholder_pb2.PlaceholderExpression,
      component_spec: Optional[Type[types.ComponentSpec]] = None
  ) -> placeholder_pb2.PlaceholderExpression:
    del component_spec  # Unused by B64EncodeOperator

    result = placeholder_pb2.PlaceholderExpression()
    result.operator.base64_encode_op.expression.CopyFrom(sub_expression_pb)
    return result


class Placeholder(abc.ABC):
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
      component_spec: Optional[Type[types.ComponentSpec]] = None
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


class ArtifactPlaceholder(Placeholder):
  """Artifact Placeholder represents an input or an output artifact.

  Prefer to use input(...) or output(...) to create artifact placeholders.
  """

  @property
  def uri(self):
    self._try_inject_index_operator()
    self._operators.append(_ArtifactUriOperator())
    return self

  def split_uri(self, split: str):
    self._try_inject_index_operator()
    self._operators.append(_ArtifactUriOperator(split))
    return self

  @property
  def value(self):
    self._try_inject_index_operator()
    self._operators.append(_ArtifactValueOperator())
    return self

  def __getitem__(self, key: int):
    self._operators.append(_IndexOperator(key))
    return self

  def _try_inject_index_operator(self):
    if not self._operators or not isinstance(self._operators[-1],
                                             _IndexOperator):
      self._operators.append(_IndexOperator(0))


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
