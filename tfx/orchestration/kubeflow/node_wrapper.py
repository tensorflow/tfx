# Lint as: python2, python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A wrapper to pass a node without its type information."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, Text

from tfx.components.base import base_node
from tfx.types import node_common


class NodeWrapper(base_node.BaseNode):
  """Wrapper of a node.

  The wrapper is needed for container entrypoint to deserialize a component
  wihtout knowning it's original python class. This enables users
  to use container base component without re-compiling the tfx base image every
  time they change the component and spec definitions.
  """

  def __init__(self, node: base_node.BaseNode):
    self.executor_spec = node.executor_spec
    self.driver_class = node.driver_class
    self._type = node.type
    self._id = node.id
    self._inputs = node.inputs
    self._outputs = node.outputs
    self._exec_properties = node.exec_properties

  @property
  def type(self) -> Text:
    return self._type

  @property
  def id(self) -> Text:
    return self._id

  @property
  def inputs(self) -> node_common._PropertyDictWrapper:  # pylint: disable=protected-access
    return self._inputs

  @property
  def outputs(self) -> node_common._PropertyDictWrapper:  # pylint: disable=protected-access
    return self._outputs

  @property
  def exec_properties(self) -> Dict[Text, Any]:
    return self._exec_properties
