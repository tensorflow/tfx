# Lint as: python2, python3
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

# TODO(b/149535307): Remove __future__ imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import operator
from typing import Any, Dict, Text, List
import attr
from tfx.proto.orchestration import placeholder_pb2
from tfx.types import artifact_utils
from tfx.types.artifact import Artifact

from google.protobuf import descriptor_pool
from google.protobuf import message_factory
from google.protobuf import text_format


@attr.s
class ResolutionContext(object):
  """A struct to store information for needed for resolution."""
  # The input artifact map
  input_dict = attr.ib(type=Dict[Text, List[Artifact]], default=None)
  # The output artifact map
  output_dict = attr.ib(type=Dict[Text, List[Artifact]], default=None)
  # The exec_properties map
  exec_properties = attr.ib(type=Dict[Text, Any], default=None)


def resolve_placeholder_expression(
    expression: placeholder_pb2.PlaceholderExpression,
    context: ResolutionContext):
  """Evaluate the placeholder expression using the given context."""
  if expression.HasField("primitive_value"):
    return getattr(expression.primitive_value,
                   expression.primitive_value.WhichOneof("value"))
  elif expression.HasField("placeholder"):
    return resolve_placeholder(expression.placeholder, context)
  elif expression.HasField("operator"):
    operator_dict = {
        "artifact_uri_op": resolve_artifact_uri_operator,
        "concat_op": resolve_concat_operator,
        "index_op": resolve_index_operator,
        "proto_op": resolve_proto_operator,
    }
    op_name = expression.operator.WhichOneof("operator_type")
    if op_name not in operator_dict:
      raise ValueError("Unsupported operator type: {}.".format(op_name))
    return operator_dict[op_name](getattr(expression.operator, op_name),
                                  context)
  else:
    raise ValueError("Unexpected placeholder expression type: {}.".format(
        expression.WhichOneof("expression_type")))


def resolve_placeholder(placeholder: placeholder_pb2.Placeholder,
                        context: ResolutionContext):
  """Evaluate the placeholder using the given context."""
  placeholder_type_dict = {
      placeholder_pb2.Placeholder.Type.INPUT_ARTIFACT: context.input_dict,
      placeholder_pb2.Placeholder.Type.OUTPUT_ARTIFACT: context.output_dict,
      placeholder_pb2.Placeholder.Type.EXEC_PROPERTY: context.exec_properties,
  }
  if placeholder.type not in placeholder_type_dict:
    raise ValueError("Unsupported placeholder type: {}.".format(
        placeholder.type))
  if placeholder.key not in placeholder_type_dict[placeholder.type]:
    raise ValueError("Failed to find key {} of placeholder type {}.".format(
        placeholder.key, placeholder.type))
  return placeholder_type_dict[placeholder.type][placeholder.key]


def resolve_artifact_uri_operator(op: placeholder_pb2.ArtifactUriOperator,
                                  context: ResolutionContext) -> Text:
  """Evaluate the artfiact uri operator."""
  artifact = resolve_placeholder_expression(op.expression, context)
  if not isinstance(artifact, Artifact):
    raise ValueError(
        "ArtifactUriOperator expects the expression to evaluate to an artifact."
    )
  if op.split:
    return artifact_utils.get_split_uri([artifact], op.split)
  else:
    return artifact.uri


def resolve_concat_operator(op: placeholder_pb2.ConcatOperator,
                            context: ResolutionContext):
  """Evaluate the concat operator."""
  return "".join(
      [str(resolve_placeholder_expression(e, context)) for e in op.expressions])


def resolve_index_operator(op: placeholder_pb2.IndexOperator,
                           context: ResolutionContext):
  """Evaluate the index operator."""
  value = resolve_placeholder_expression(op.expression, context)
  try:
    return value[op.index]
  except:
    raise ValueError(
        "IndexOperator failed to access the given index {}.".format(op.index))


def resolve_proto_operator(op: placeholder_pb2.ProtoOperator,
                           context: ResolutionContext):
  """Evaluate the proto operator."""
  raw_message = resolve_placeholder_expression(op.expression, context)

  pool = descriptor_pool.DescriptorPool()
  for file_descriptor in op.proto_schema.file_descriptors.file:
    pool.Add(file_descriptor)
  message_descriptor = pool.FindMessageTypeByName(op.proto_schema.message_type)
  factory = message_factory.MessageFactory(pool)
  message_type = factory.GetPrototype(message_descriptor)
  message = message_type.FromString(raw_message)

  if op.proto_field_path:
    return operator.attrgetter(op.proto_field_path)(message)
  return text_format.MessageToString(message)
