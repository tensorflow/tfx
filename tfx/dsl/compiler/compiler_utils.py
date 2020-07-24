# Lint as: python3
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
"""Utility functions for DSL Compiler."""

# TODO(b/149535307): Remove __future__ imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Type, List, Optional, Text

from tfx import types
from tfx.components.base import base_component
from tfx.components.base import base_node
from tfx.components.common_nodes import resolver_node
from tfx.proto.orchestration import pipeline_pb2


def set_runtime_parameter_pb(
    pb: pipeline_pb2.RuntimeParameter,
    name: Text,
    ptype: Type[types.Property],
    default_value: Optional[types.Property] = None
) -> pipeline_pb2.RuntimeParameter:
  """Helper function to fill a RuntimeParameter proto.

  Args:
    pb: A RuntimeParameter proto to be filled in.
    name: Name to be set at pb.name.
    ptype: The Python type to be set at pb.type.
    default_value: Optional. If provided, it will be pb.default_value.

  Returns:
    A RuntimeParameter proto filled with provided values.
  """
  pb.name = name
  if ptype == int:
    pb.type = pipeline_pb2.RuntimeParameter.Type.INT
    if default_value:
      pb.default_value.int_value = default_value
  elif ptype == float:
    pb.type = pipeline_pb2.RuntimeParameter.Type.DOUBLE
    if default_value:
      pb.default_value.double_value = default_value
  elif ptype == str:
    pb.type = pipeline_pb2.RuntimeParameter.Type.STRING
    if default_value:
      pb.default_value.string_value = default_value
  else:
    raise ValueError("Got unsupported runtime parameter type: {}".format(ptype))
  return pb


def is_resolver(node: base_node.BaseNode) -> bool:
  """Helper function to check if a TFX node is a Resolver."""
  return isinstance(node, resolver_node.ResolverNode)


def is_component(node: base_node.BaseNode) -> bool:
  """Helper function to check if a TFX node is a normal component."""
  return isinstance(node, base_component.BaseComponent)


def ensure_topological_order(nodes: List[base_node.BaseNode]) -> bool:
  """Helper function to check if nodes are topologically sorted."""
  visited = set()
  for node in nodes:
    for upstream_node in node.upstream_nodes:
      if upstream_node not in visited:
        return False
    visited.add(node)
  return True
