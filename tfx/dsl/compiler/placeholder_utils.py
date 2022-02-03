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
"""Utilities to evaluate and resolve Placeholders."""

import base64
import enum
import re
from typing import Any, Callable, Dict, Union

from absl import logging
import attr
from tfx.dsl.placeholder import placeholder as ph
from tfx.orchestration.portable import data_types
from tfx.proto.orchestration import placeholder_pb2
from tfx.types import artifact
from tfx.types import artifact_utils
from tfx.types import value_artifact
from tfx.utils import json_utils
from tfx.utils import proto_utils

from google.protobuf.internal import containers
from google.protobuf.pyext import _message
from google.protobuf import json_format
from google.protobuf import message
from google.protobuf import text_format


class NullDereferenceError(Exception):
  """Raised by the ExpressionResolver when dereferencing None or empty list."""

  def __init__(self, placeholder):
    self.placeholder = placeholder
    super().__init__()


@attr.s(auto_attribs=True, frozen=True)
class ResolutionContext:
  """A struct to store information needed for resolution.

  Attributes:
    exec_info: An ExecutionInfo object that includes needed information to
      render all kinds of placeholders.
    executor_spec: An executor spec proto for rendering context placeholder.
    platform_config: A platform config proto for rendering context placeholder.
  """
  exec_info: data_types.ExecutionInfo = None
  executor_spec: message.Message = None
  platform_config: message.Message = None


# Includes three basic types from MLMD: int, float, str
# and an additional primitive type from proto field access: bool
# Note: Pytype's int includes long from Python3
# We does not support bytes, which may result from proto field access. Must use
# base64 encode operator to explicitly convert it into str.
_PlaceholderResolvedTypes = (int, float, str, bool, type(None), list)
_PlaceholderResolvedTypeHints = Union[_PlaceholderResolvedTypes]


def resolve_placeholder_expression(
    expression: placeholder_pb2.PlaceholderExpression,
    context: ResolutionContext) -> _PlaceholderResolvedTypeHints:
  """Evaluates a placeholder expression using the given context.

  Normally the resolved value will be used as command line flags in strings.
  This function does not automatically perform the string conversion, i.e.,
  the return type is the same as the type of the value originally has. Currently
  it can be
    exec property supported primitive types: int, float, string.
    if use proto operator: serilaized proto message, or proto primitive fields.
  The caller needs to perform desired string conversions.

  Args:
    expression: A placeholder expression to be resolved.
    context: Information needed to resolve the expression.

  Returns:
    Resolved expression value.
  """
  try:
    result = _ExpressionResolver(context).resolve(expression)
  except NullDereferenceError as err:
    logging.warning(
        "Dereferenced None during placeholder evaluation. Ignoring.")
    logging.warning("Placeholder=%s", err.placeholder)
    return None
  except Exception as e:
    raise ValueError(
        f"Failed to resolve placeholder expression: {debug_str(expression)}"
    ) from e

  if not isinstance(result, _PlaceholderResolvedTypes):
    raise ValueError(f"Placeholder {debug_str(expression)} evaluates to "
                     f"an unsupported type: {type(result)}.")
  return result


class _Operation(enum.Enum):
  """Alias for Operation enum types in placeholder.proto."""

  EQUAL = placeholder_pb2.ComparisonOperator.Operation.EQUAL
  LESS_THAN = placeholder_pb2.ComparisonOperator.Operation.LESS_THAN
  GREATER_THAN = placeholder_pb2.ComparisonOperator.Operation.GREATER_THAN
  NOT = placeholder_pb2.UnaryLogicalOperator.Operation.NOT
  AND = placeholder_pb2.BinaryLogicalOperator.Operation.AND
  OR = placeholder_pb2.BinaryLogicalOperator.Operation.OR


def _resolve_and_ensure_boolean(
    resolve_fn: Callable[[placeholder_pb2.PlaceholderExpression], Any],
    expression: placeholder_pb2.PlaceholderExpression,
    error_message: str,
) -> bool:
  # TODO(b/173529355): Block invalid placeholders during compilation time
  """Ensures that expression resolves to boolean.

  NOTE: This is currently used to ensure that the inputs to logical operators
  are boolean. Since the DSL for creating Predicate expressions do not currently
  perform implicit boolean conversions, the evaluator should not support it
  either. If we support implicit boolean conversions in the DSL in the
  future, this check can be removed.

  Args:
    resolve_fn: The function for resolving placeholder expressions.
    expression: The placeholder expression to resolve.
    error_message: The error message to display if the expression does not
      resolve to a boolean type.

  Returns:
    The resolved boolean value.

  Raises:
    ValueError if expression does not resolve to boolean type.
  """
  value = resolve_fn(expression)
  if isinstance(value, bool):
    return value
  raise ValueError(f"{error_message}\n"
                   f"expression: {expression}\n"
                   f"resolved value type: {type(value)}\n"
                   f"resolved value: {value}")


# Dictionary of registered placeholder operators,
# maps from operator proto type names to actual operator functions.
_PLACEHOLDER_OPERATORS: Dict[str, Callable[..., Any]] = {}


def _register(op_proto):
  """Decorator function for registering operators. Internal in this module."""

  def decorator(op: Callable[..., Any]):
    _PLACEHOLDER_OPERATORS[op_proto.DESCRIPTOR.name] = op
    return op

  return decorator


class _ExpressionResolver:
  """Utility class to resolve Placeholder expressions.

  Placeholder expression is defined as a proto structure
  placeholder_pb2.PlaceholderExpression. It can be resolved with
  ResolutionContext to a concrete value.
  """

  def __init__(self, context: ResolutionContext):
    self._resolution_values = {
        placeholder_pb2.Placeholder.Type.INPUT_ARTIFACT:
            context.exec_info.input_dict,
        placeholder_pb2.Placeholder.Type.OUTPUT_ARTIFACT:
            context.exec_info.output_dict,
        placeholder_pb2.Placeholder.Type.EXEC_PROPERTY:
            context.exec_info.exec_properties,
        placeholder_pb2.Placeholder.Type.RUNTIME_INFO: {
            ph.RuntimeInfoKey.EXECUTOR_SPEC.value: context.executor_spec,
            ph.RuntimeInfoKey.PLATFORM_CONFIG.value: context.platform_config,
        },
        placeholder_pb2.Placeholder.Type.EXEC_INVOCATION:
            context.exec_info.to_proto(),
    }

  def resolve(self, expression: placeholder_pb2.PlaceholderExpression) -> Any:
    """Recursively evaluates a placeholder expression."""
    if expression.HasField("value"):
      return getattr(expression.value, expression.value.WhichOneof("value"))
    elif expression.HasField("placeholder"):
      return self._resolve_placeholder(expression.placeholder)
    elif expression.HasField("operator"):
      return self._resolve_placeholder_operator(expression.operator)
    else:
      raise ValueError("Unexpected placeholder expression type: "
                       f"{expression.WhichOneof('expression_type')}.")

  def _resolve_placeholder(self,
                           placeholder: placeholder_pb2.Placeholder) -> Any:
    """Evaluates a placeholder using the contexts."""
    try:
      context = self._resolution_values[placeholder.type]
    except KeyError as e:
      raise KeyError(
          f"Unsupported placeholder type: {placeholder.type}.") from e

    # Handle the special case of EXEC_INVOCATION placeholders, which don't take
    # a key.
    if placeholder.type == placeholder_pb2.Placeholder.Type.EXEC_INVOCATION:
      return context

    # Handle remaining placeholder types.
    try:
      return context[placeholder.key]
    except KeyError as e:
      # Handle placeholders that access a missing optional channel or exec
      # property. In both cases the requested key will not be present in the
      # context. However this means we cannot distinguish between a correct
      # placeholder with an optional value vs. an incorrect placeholder.
      # TODO(b/172001324): Handle this at compile time.
      raise NullDereferenceError(placeholder)

  def _resolve_placeholder_operator(
      self, placeholder_operator: placeholder_pb2.PlaceholderExpressionOperator
  ) -> Any:
    """Evaluates a placeholder operator by dispatching to the operator methods."""
    operator_name = placeholder_operator.WhichOneof("operator_type")
    operator_pb = getattr(placeholder_operator, operator_name)
    try:
      operator_fn = _PLACEHOLDER_OPERATORS[operator_pb.DESCRIPTOR.name]
    except KeyError as e:
      raise KeyError(
          f"Unsupported placeholder operator: {operator_pb.DESCRIPTOR.name}."
      ) from e
    return operator_fn(self, operator_pb)

  @_register(placeholder_pb2.ArtifactUriOperator)
  def _resolve_artifact_uri_operator(
      self, op: placeholder_pb2.ArtifactUriOperator) -> str:
    """Evaluates the artifact URI operator."""
    resolved_artifact = self.resolve(op.expression)
    if resolved_artifact is None:
      raise NullDereferenceError(op.expression)
    if not isinstance(resolved_artifact, artifact.Artifact):
      raise ValueError("ArtifactUriOperator expects the expression "
                       "to evaluate to an artifact. "
                       f"Got {type(resolved_artifact)}")
    if op.split:
      return artifact_utils.get_split_uri([resolved_artifact], op.split)
    else:
      return resolved_artifact.uri

  @_register(placeholder_pb2.ArtifactValueOperator)
  def _resolve_artifact_value_operator(
      self, op: placeholder_pb2.ArtifactValueOperator) -> str:
    """Evaluates the artifact value operator."""
    resolved_artifact = self.resolve(op.expression)
    if resolved_artifact is None:
      raise NullDereferenceError(op.expression)
    if not isinstance(resolved_artifact, value_artifact.ValueArtifact):
      raise ValueError("ArtifactValueOperator expects the expression "
                       "to evaluate to a value artifact."
                       f"Got {type(resolved_artifact)}")
    return resolved_artifact.read()

  @_register(placeholder_pb2.ConcatOperator)
  def _resolve_concat_operator(self, op: placeholder_pb2.ConcatOperator) -> str:
    """Evaluates the concat operator."""
    parts = []
    for e in op.expressions:
      value = self.resolve(e)
      if value is None:
        raise NullDereferenceError(e)
      parts.append(value)
    return "".join(str(part) for part in parts)

  @_register(placeholder_pb2.IndexOperator)
  def _resolve_index_operator(self, op: placeholder_pb2.IndexOperator) -> Any:
    """Evaluates the index operator."""
    value = self.resolve(op.expression)
    if value is None or not value:
      raise NullDereferenceError(op.expression)
    try:
      return value[op.index]
    except (TypeError, IndexError) as e:
      raise ValueError(
          f"IndexOperator failed to access the given index {op.index}.") from e

  @_register(placeholder_pb2.ArtifactPropertyOperator)
  def _resolve_property_operator(
      self, op: placeholder_pb2.ArtifactPropertyOperator) -> Any:
    """Evaluates the artifact property operator."""
    value = self.resolve(op.expression)
    if value is None or not value:
      raise NullDereferenceError(op.expression)
    if not isinstance(value, artifact.Artifact):
      raise ValueError("ArtifactPropertyOperator failed to access property "
                       f"{op.key}. Expected Artifact type. Got {type(value)}.")
    try:
      if op.is_custom_property:
        return value.get_custom_property(op.key)
      return value.__getattr__(op.key)
    except:
      raise ValueError("ArtifactPropertyOperator failed to find property with "
                       f"key {op.key}.")

  @_register(placeholder_pb2.Base64EncodeOperator)
  def _resolve_base64_encode_operator(
      self, op: placeholder_pb2.Base64EncodeOperator) -> str:
    """Evaluates the Base64 encode operator."""
    value = self.resolve(op.expression)
    if value is None:
      raise NullDereferenceError(op.expression)
    if isinstance(value, str):
      return base64.urlsafe_b64encode(value.encode()).decode("ascii")
    elif isinstance(value, bytes):
      return base64.urlsafe_b64encode(value).decode("ascii")
    else:
      raise ValueError(
          f"Failed to Base64 encode {value} of type {type(value)}.")

  @_register(placeholder_pb2.ListSerializationOperator)
  def _resolve_list_serialization_operator(
      self, op: placeholder_pb2.ListSerializationOperator) -> str:
    """Evaluates the list operator."""
    value = self.resolve(op.expression)
    if value is None:
      raise NullDereferenceError(op.expression)
    elif not all(isinstance(val, (str, int, float, bool)) for val in value):
      raise ValueError(
          "Only list of bool, int, float or str is supported for list serialization."
      )

    if op.serialization_format == placeholder_pb2.ListSerializationOperator.JSON:
      return json_utils.dumps(value)
    elif op.serialization_format == placeholder_pb2.ListSerializationOperator.COMMA_SEPARATED_STR:
      return ",".join(
          f'"{val}"' if isinstance(val, str) else str(val) for val in value)

    raise ValueError(
        "List serialization operator failed to resolve. A serialization format is needed."
    )

  @_register(placeholder_pb2.ProtoOperator)
  def _resolve_proto_operator(self, op: placeholder_pb2.ProtoOperator) -> Any:
    """Evaluates the proto operator."""
    raw_message = self.resolve(op.expression)
    if raw_message is None:
      raise NullDereferenceError(op.expression)

    if isinstance(raw_message, str):
      value = proto_utils.deserialize_proto_message(
          raw_message,
          op.proto_schema.message_type,
          file_descriptors=op.proto_schema.file_descriptors)
    elif isinstance(raw_message, message.Message):
      # Message such as platform config should not be encoded.
      value = raw_message
    else:
      raise ValueError(
          f"Got unsupported value type for proto operator: {type(raw_message)}."
      )

    if op.proto_field_path:
      for field in op.proto_field_path:
        if field.startswith("."):
          try:
            value = getattr(value, field[1:])
          except AttributeError:
            raise ValueError("While evaluting placeholder proto operator, "
                             f"got unknown proto field {field}.")
          continue
        map_key = re.findall(r"\[['\"](.+)['\"]\]", field)
        if len(map_key) == 1:
          try:
            value = value[map_key[0]]
          except KeyError:
            raise ValueError("While evaluting placeholder proto operator, "
                             f"got unknown map field {field}.")
          continue
        # Going forward, index access for proto fields should be handled by
        # index op. This code here is kept to avoid breaking existing executor
        # specs.
        index = re.findall(r"\[(\d+)\]", field)
        if index and str.isdecimal(index[0]):
          try:
            value = value[int(index[0])]
          except IndexError:
            raise ValueError("While evaluting placeholder proto operator, "
                             f"got unknown index field {field}.")
          continue
        raise ValueError(f"Got unsupported proto field path: {field}")

    # Non-message primitive values are returned directly.
    if isinstance(value, (int, float, str, bool, bytes)):
      return value

    # Return repeated fields as list.
    if isinstance(
        value,
        (_message.RepeatedCompositeContainer, _message.RepeatedScalarContainer,
         containers.RepeatedCompositeFieldContainer,
         containers.RepeatedScalarFieldContainer)):
      return list(value)

    if not isinstance(value, message.Message):
      raise ValueError(f"Got unsupported value type {type(value)} "
                       "from accessing proto field path.")

    # For message-typed values, we need to consider serialization format.
    if op.serialization_format:
      if op.serialization_format == placeholder_pb2.ProtoOperator.JSON:
        return json_format.MessageToJson(
            message=value, sort_keys=True, preserving_proto_field_name=True)
      if op.serialization_format == placeholder_pb2.ProtoOperator.TEXT_FORMAT:
        return text_format.MessageToString(value)
      if op.serialization_format == placeholder_pb2.ProtoOperator.BINARY:
        return value.SerializeToString()

    raise ValueError(
        "Proto operator resolves to a proto message value. A serialization "
        "format is needed to render it.")

  @_register(placeholder_pb2.ComparisonOperator)
  def _resolve_comparison_operator(
      self, op: placeholder_pb2.ComparisonOperator) -> bool:
    """Evaluates the comparison operator."""
    lhs_value = self.resolve(op.lhs)
    rhs_value = self.resolve(op.rhs)
    if op.op == _Operation.EQUAL.value:
      return bool(lhs_value == rhs_value)
    elif op.op == _Operation.LESS_THAN.value:
      return bool(lhs_value < rhs_value)
    elif op.op == _Operation.GREATER_THAN.value:
      return bool(lhs_value > rhs_value)

    raise ValueError(f"Unrecognized comparison operation {op.op}")

  @_register(placeholder_pb2.UnaryLogicalOperator)
  def _resolve_unary_logical_operator(
      self, op: placeholder_pb2.UnaryLogicalOperator) -> bool:
    """Evaluates the unary logical operator."""
    error_message = (
        "Unary logical operations' sub-expression must resolve to bool.")
    value = _resolve_and_ensure_boolean(self.resolve, op.expression,
                                        error_message)
    if op.op == _Operation.NOT.value:
      return not value

    raise ValueError(f"Unrecognized unary logical operation {op.op}.")

  @_register(placeholder_pb2.BinaryLogicalOperator)
  def _resolve_binary_logical_operator(
      self, op: placeholder_pb2.BinaryLogicalOperator) -> bool:
    """Evaluates the binary logical operator."""
    error_message = (
        "Binary logical operations' sub-expression must resolve to bool. "
        "{} is not bool.")
    lhs_value = _resolve_and_ensure_boolean(self.resolve, op.lhs,
                                            error_message.format("lhs"))
    rhs_value = _resolve_and_ensure_boolean(self.resolve, op.rhs,
                                            error_message.format("rhs"))
    if op.op == _Operation.AND.value:
      return lhs_value and rhs_value
    elif op.op == _Operation.OR.value:
      return lhs_value or rhs_value

    raise ValueError(f"Unrecognized binary logical operation {op.op}.")


def debug_str(expression: placeholder_pb2.PlaceholderExpression) -> str:
  """Gets the debug string of a placeholder expression proto.

  Args:
    expression: A placeholder expression proto.

  Returns:
    Debug string of the placeholder expression.
  """
  if expression.HasField("value"):
    value_field_name = expression.value.WhichOneof("value")
    return f"\"{getattr(expression.value, value_field_name)}\""

  if expression.HasField("placeholder"):
    placeholder_pb = expression.placeholder
    ph_names_map = {
        placeholder_pb2.Placeholder.INPUT_ARTIFACT: "input",
        placeholder_pb2.Placeholder.OUTPUT_ARTIFACT: "output",
        placeholder_pb2.Placeholder.EXEC_PROPERTY: "exec_property",
        placeholder_pb2.Placeholder.RUNTIME_INFO: "runtime_info",
        placeholder_pb2.Placeholder.EXEC_INVOCATION: "execution_invocation"
    }
    ph_name = ph_names_map[placeholder_pb.type]
    if placeholder_pb.key:
      return f"{ph_name}(\"{placeholder_pb.key}\")"
    else:
      return f"{ph_name}()"

  if expression.HasField("operator"):
    operator_name = expression.operator.WhichOneof("operator_type")
    operator_pb = getattr(expression.operator, operator_name)
    if operator_name == "artifact_uri_op":
      sub_expression_str = debug_str(operator_pb.expression)
      if operator_pb.split:
        return f"{sub_expression_str}.split_uri(\"{operator_pb.split}\")"
      else:
        return f"{sub_expression_str}.uri"

    if operator_name == "artifact_value_op":
      sub_expression_str = debug_str(operator_pb.expression)
      return f"{sub_expression_str}.value"

    if operator_name == "artifact_property_op":
      sub_expression_str = debug_str(operator_pb.expression)
      if operator_pb.is_custom_property:
        return f"{sub_expression_str}.custom_property(\"{operator_pb.key}\")"
      else:
        return f"{sub_expression_str}.property(\"{operator_pb.key}\")"

    if operator_name == "concat_op":
      expression_str = " + ".join(debug_str(e) for e in operator_pb.expressions)
      return f"({expression_str})"

    if operator_name == "index_op":
      sub_expression_str = debug_str(operator_pb.expression)
      return f"{sub_expression_str}[{operator_pb.index}]"

    if operator_name == "proto_op":
      sub_expression_str = debug_str(operator_pb.expression)
      field_path = "".join(operator_pb.proto_field_path)
      expression_str = f"{sub_expression_str}{field_path}"
      if operator_pb.serialization_format:
        format_str = placeholder_pb2.ProtoOperator.SerializationFormat.Name(
            operator_pb.serialization_format)
        return f"{expression_str}.serialize({format_str})"
      return expression_str

    if operator_name == "base64_encode_op":
      sub_expression_str = debug_str(operator_pb.expression)
      return f"{sub_expression_str}.b64encode()"

    if operator_name == "compare_op":
      lhs_str = debug_str(operator_pb.lhs)
      rhs_str = debug_str(operator_pb.rhs)
      if operator_pb.op == _Operation.EQUAL.value:
        op_str = "=="
      elif operator_pb.op == _Operation.LESS_THAN.value:
        op_str = "<"
      elif operator_pb.op == _Operation.GREATER_THAN.value:
        op_str = ">"
      else:
        return f"Unknown Comparison Operation {operator_pb.op}"
      return f"({lhs_str} {op_str} {rhs_str})"

    if operator_name == "unary_logical_op":
      expression_str = debug_str(operator_pb.expression)
      if operator_pb.op == _Operation.NOT.value:
        op_str = "not"
      else:
        return f"Unknown Unary Logical Operation {operator_pb.op}"
      return f"{op_str}({expression_str})"

    if operator_name == "binary_logical_op":
      lhs_str = debug_str(operator_pb.lhs)
      rhs_str = debug_str(operator_pb.rhs)
      if operator_pb.op == _Operation.AND.value:
        op_str = "and"
      elif operator_pb.op == _Operation.OR.value:
        op_str = "or"
      else:
        return f"Unknown Binary Logical Operation {operator_pb.op}"
      return f"({lhs_str} {op_str} {rhs_str})"

    if operator_name == "list_serialization_op":
      expression_str = debug_str(operator_pb.expression)
      if operator_pb.serialization_format:
        format_str = placeholder_pb2.ProtoOperator.SerializationFormat.Name(
            operator_pb.serialization_format)
        return f"{expression_str}.serialize_list({format_str})"
      return expression_str

    return "Unknown placeholder operator"

  return "Unknown placeholder expression"
