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
"""Utilities for topological sort."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Callable, List, Sequence, Text, TypeVar

NodeT = TypeVar('NodeT')


class InvalidDAGError(Exception):
  """Error to indicate invalid DAG."""


def topsorted_layers(
    nodes: Sequence[NodeT], get_node_id_fn: Callable[[NodeT], Text],
    get_parent_nodes: Callable[[NodeT], List[NodeT]],
    get_child_nodes: Callable[[NodeT], List[NodeT]]) -> List[List[NodeT]]:
  """Sorts the DAG of nodes in topological order.

  Args:
    nodes: A sequence of nodes.
    get_node_id_fn: Callable that returns a unique text identifier for a node.
    get_parent_nodes: Callable that returns a list of parent nodes for a node.
    get_child_nodes: Callable that returns a list of chlid nodes for a node.

  Returns:
    A list of topologically ordered node layers. Each layer of nodes is sorted
    by its node id given by `get_node_id_fn`.

  Raises:
    InvalidDAGError: If the input nodes don't form a DAG.
    ValueError: If the nodes are not unique.
  """
  # Make sure the nodes are unique.
  if len(set(get_node_id_fn(n) for n in nodes)) != len(nodes):
    raise ValueError('Nodes must have unique ids.')

  # The first layer contains nodes with no incoming edges.
  layer = [node for node in nodes if not get_parent_nodes(node)]

  visited = set()
  layers = []
  while layer:
    layer = sorted(layer, key=get_node_id_fn)
    layers.append(layer)

    next_layer = []
    for node in layer:
      visited.add(get_node_id_fn(node))
      for child_node in get_child_nodes(node):
        # Include the child node if all its parents are visited. If the child
        # node is part of a cycle, it will never be included since it will have
        # at least one unvisited parent node which is also part of the cycle.
        parent_node_ids = set(
            get_node_id_fn(p) for p in get_parent_nodes(child_node))
        if parent_node_ids.issubset(visited):
          next_layer.append(child_node)
    layer = next_layer

  # Nodes in cycles are not included in layers; raise an error if this happens.
  if sum(len(layer) for layer in layers) < len(nodes):
    raise InvalidDAGError('Cycle detected.')

  return layers
