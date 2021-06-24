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
"""TFX Conditionals."""
import collections
import threading
from typing import Tuple

from tfx.dsl.components.base import base_node
from tfx.dsl.components.base import node_registry
from tfx.dsl.placeholder import placeholder


class _ConditionalRegistry(threading.local):
  """Registers the predicates that a node is associated with in local thread."""

  def __init__(self):
    super().__init__()
    # node_anchors maps from a predicate to a set of nodes.
    # Each entry marks the registered nodes at the time when a conditional
    # context (i.e. a predicate) starts.
    self.node_anchors = dict()
    # A reverse map of a node frame, mapping from a node to the predicates
    # it's associated with.
    self.conditional_map = collections.defaultdict(list)

  def enter_conditional(self, predicate: placeholder.Predicate):
    if predicate in self.node_anchors:
      raise ValueError(
          f'Nested conditionals with duplicate predicates: {predicate}.'
          'Consider merging the nested conditionals.')
    self.node_anchors[predicate] = node_registry.registered_nodes()

  def exit_conditional(self, predicate: placeholder.Predicate):
    nodes_in_frame = (
        node_registry.registered_nodes() - self.node_anchors[predicate])
    for node in nodes_in_frame:
      self.conditional_map[node].append(predicate)
    del self.node_anchors[predicate]


_conditional_registry = _ConditionalRegistry()


def get_predicates(node: base_node.BaseNode) -> Tuple[placeholder.Predicate]:
  """Gets predicates that a node is associated with in local thread."""
  # Because inner with-block exits first, we reverse the list to make the order
  # correct. Returns a tuple to ensure the result is consistent across tests.
  return tuple(_conditional_registry.conditional_map[node][::-1])


class Cond:
  """Context manager that registers a predicate with nodes in local thread."""

  def __init__(self, predicate: placeholder.Predicate):
    self._predicate = predicate

  def __enter__(self):
    _conditional_registry.enter_conditional(self._predicate)

  def __exit__(self, exc_type, exc_val, exc_tb):
    _conditional_registry.exit_conditional(self._predicate)
