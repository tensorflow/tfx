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
"""Special placeholders to obtain values from the runtime environment."""

from __future__ import annotations

import typing
from typing import Any, Literal, Union

from tfx.dsl.placeholder import placeholder_base
from tfx.proto.orchestration import placeholder_pb2

from google.protobuf import message


def exec_property(key: str) -> ExecPropertyPlaceholder:
  """Returns a Placeholder that represents an execution property.

  Args:
    key: The key of the output artifact.

  Returns:
    A Placeholder that supports

      1. Rendering the value of an execution property at a given key.
         Example: `#!python exec_property('version')`
      2. Rendering the whole proto or a proto field of an execution property,
         if the value is a proto type.
         The (possibly nested) proto field in a placeholder can be accessed as
         if accessing a proto field in Python.
         Example: `#!python exec_property('model_config').num_layers`
      3. Concatenating with other placeholders or strings.
         Example: `#!python output('model').uri + '/model/' + exec_property('version')`
  """
  return ExecPropertyPlaceholder(key)


RuntimeInfoKeys = Literal[
    'executor_spec',
    'platform_config',
    'pipeline_platform_config',
]


def runtime_info(key: RuntimeInfoKeys) -> RuntimeInfoPlaceholder:
  """Returns a Placeholder that contains runtime information for component.

  Currently the runtime info includes following keys:
  1. `executor_spec`: The executor spec proto.
  2. `platform_config`: A proto that contains platform-specific information for
         the current pipeline node.
  3. `pipeline_platform_config`: A proto that contains platform-specific
        information for the pipeline as a whole.


  Args:
    key: The key of the runtime information.

  Returns:
    A Placeholder that will render to the information associated with the key.
      If the placeholder is proto-valued. Accessing a proto field can be
      represented as if accessing a proto field in Python.

  Raises:
    ValueError: If received unsupported key.
  """
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


class ExecPropertyPlaceholder(placeholder_base.Placeholder):
  """ExecProperty Placeholder represents an execution property.

  Prefer to use exec_property(...) to create exec property placeholders.
  """

  def __init__(self, key: str):
    """Initializes the class. Consider this private."""
    super().__init__(
        expected_type=Union[message.Message, placeholder_base.ValueType]
    )
    self._key = key

  @property
  def key(self) -> str:
    return self._key

  def internal_equals(self, other: placeholder_base.Placeholder) -> bool:
    return isinstance(other, ExecPropertyPlaceholder) and self.key == other.key

  def encode(
      self, component_spec: Any = None
  ) -> placeholder_pb2.PlaceholderExpression:
    result = placeholder_pb2.PlaceholderExpression()
    result.placeholder.type = placeholder_pb2.Placeholder.Type.EXEC_PROPERTY
    result.placeholder.key = self._key
    return result


class RuntimeInfoPlaceholder(placeholder_base.Placeholder):
  """RuntimeInfo Placeholder represents runtime information for a component.

  Prefer to use runtime_info(...) to create RuntimeInfo placeholders.
  """

  def __init__(self, key: RuntimeInfoKeys):
    """Initializes the class. Consider this private."""
    super().__init__(expected_type=message.Message)
    if key not in typing.get_args(RuntimeInfoKeys):
      raise ValueError(f'Got unsupported runtime info key: {key}.')
    self._key = key

  def internal_equals(self, other: placeholder_base.Placeholder) -> bool:
    return isinstance(other, RuntimeInfoPlaceholder) and self._key == other._key  # pylint: disable=protected-access

  def encode(
      self, component_spec: Any = None
  ) -> placeholder_pb2.PlaceholderExpression:
    result = placeholder_pb2.PlaceholderExpression()
    result.placeholder.type = placeholder_pb2.Placeholder.Type.RUNTIME_INFO
    result.placeholder.key = self._key
    return result


class ExecInvocationPlaceholder(placeholder_base.Placeholder):
  """Execution Invocation Placeholder helps access ExecutionInvocation proto.

  Prefer to use execution_invocation() to create Execution Invocation
  placeholder.
  """

  def __init__(self):
    """Initializes the class. Consider this private."""
    super().__init__(expected_type=message.Message)

  def internal_equals(self, other: placeholder_base.Placeholder) -> bool:
    return isinstance(other, ExecInvocationPlaceholder)

  def encode(
      self, component_spec: None | Any = None
  ) -> placeholder_pb2.PlaceholderExpression:
    result = placeholder_pb2.PlaceholderExpression()
    result.placeholder.type = placeholder_pb2.Placeholder.Type.EXEC_INVOCATION
    return result


class EnvironmentVariablePlaceholder(placeholder_base.Placeholder):
  """Environment Variable Placeholder helps access EnvironmentVariable proto.

  Prefer to use environment_variable(...) to create Environment Variable
  placeholder.
  """

  def __init__(self, key: str):
    """Initializes the class. Consider this private."""
    super().__init__(expected_type=placeholder_base.ValueType)
    self._key = key

  def internal_equals(self, other: placeholder_base.Placeholder) -> bool:
    return (
        isinstance(other, EnvironmentVariablePlaceholder)
        and self._key == other._key  # pylint: disable=protected-access
    )

  def encode(
      self, component_spec: Any = None
  ) -> placeholder_pb2.PlaceholderExpression:
    result = placeholder_pb2.PlaceholderExpression()
    result.placeholder.type = (
        placeholder_pb2.Placeholder.Type.ENVIRONMENT_VARIABLE
    )
    result.placeholder.key = self._key
    return result
