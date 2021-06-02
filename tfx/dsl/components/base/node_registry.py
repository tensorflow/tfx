# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Node registry."""
import threading
from typing import Any, FrozenSet

# To resolve circular dependency caused by type annotations.
base_node = Any  # base_node.py imports this module.


class _NodeRegistry(threading.local):
  """Stores registered nodes in the local thread."""

  def __init__(self):
    super().__init__()
    self._nodes = set()

  def register(self, node: 'base_node.BaseNode'):
    self._nodes.add(node)

  def registered_nodes(self):
    return self._nodes

_node_registry = _NodeRegistry()


def register_node(node: 'base_node.BaseNode'):
  """Register a node in the local thread."""
  _node_registry.register(node)


def registered_nodes() -> FrozenSet['base_node.BaseNode']:
  """Get registered nodes in the local thread."""
  return frozenset(_node_registry.registered_nodes())
