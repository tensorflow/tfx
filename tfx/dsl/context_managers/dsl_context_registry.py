# Copyright 2022 Google LLC. All Rights Reserved.
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
"""Module for DslContextRegistry."""

import collections
import contextlib
import threading
from typing import Any, List, Optional, Iterator

from tfx.dsl.context_managers import dsl_context

# Use Any to avoid cyclic import.
_BaseNode = Any
_Pipeline = Any


def _format_node(node: _BaseNode) -> str:
  return f'{type(node).__name__}({node.id})#{id(node)}'


class DslContextRegistry:
  """Registry for DslContexts and associated BaseNodes of a pipeline DSL.

  DslContextRegistry manages the active DslContexts, their orders, BaseNodes,
  and association between DslContext and BaseNodes during the pipeline DSL
  definition. DslContext and BaseNode always belong to exactly one
  DslContextRegistry (1:n relationship).
  """

  def __init__(self):
    super().__init__()
    # Frame of currently active DSL context. Ordered by parent -> child.
    self._active_contexts: List[dsl_context.DslContext] = []
    # All DSL contexts that have ever been defined so far.
    self._all_contexts: List[dsl_context.DslContext] = []
    self._all_nodes: List[_BaseNode] = []
    # Mapping from Context ID to a list of nodes that belong to each context.
    # Each list of node is sorted chronologically.
    self._nodes_by_context = collections.defaultdict(list)
    self._finalized = False

  def __copy__(self):
    result = DslContextRegistry()
    result._active_contexts = list(self._active_contexts)
    result._all_contexts = list(self._all_contexts)
    result._all_nodes = list(self._all_nodes)
    result._nodes_by_context.update({
        k: list(v) for k, v in self._nodes_by_context.items()
    })
    result._finalized = False
    return result

  @property
  def all_contexts(self) -> List[dsl_context.DslContext]:
    """All contexts defined during the lifespan of the registry."""
    return list(self._all_contexts)

  @property
  def active_contexts(self) -> List[dsl_context.DslContext]:
    """All active context frame in parent -> child order."""
    return list(self._active_contexts)

  @property
  def all_nodes(self):
    return self._all_nodes

  def finalize(self):
    """Finalize and make the instance immutable."""
    self._finalized = True

  def _check_mutable(self):
    if self._finalized:
      raise RuntimeError('Cannot mutate DslContextRegistry after finalized.')

  @contextlib.contextmanager
  def temporary_mutable(self):
    """Temporarily make the registry mutable."""
    is_finalized = self._finalized
    self._finalized = False
    try:
      yield
    finally:
      self._finalized = is_finalized

  def push_context(self, context: dsl_context.DslContext):
    """Pushes the context to the top of active context frames."""
    self._check_mutable()
    self._active_contexts.append(context)
    self._all_contexts.append(context)

  def pop_context(self) -> dsl_context.DslContext:
    """Removes the top context from the active context frame."""
    self._check_mutable()
    assert self._active_contexts, (
        'Internal assertion error; no active contexts to remove.')
    return self._active_contexts.pop()

  def peek_context(self) -> Optional[dsl_context.DslContext]:
    """Returns the top context of the active context frame."""
    return self._active_contexts[-1] if self._active_contexts else None

  def put_node(self, node: _BaseNode) -> None:
    """Associates the node to all active contexts."""
    self._check_mutable()
    self._all_nodes.append(node)
    for context in self._active_contexts:
      self._nodes_by_context[context].append(node)

  def replace_node(self, node_from: _BaseNode, node_to: _BaseNode) -> None:
    """Replaces one node instance to another in a registry."""
    self._check_mutable()
    if node_from not in self._all_nodes:
      raise ValueError(f'{node_from} does not exist in pipeline registry.')
    self._all_nodes[self._all_nodes.index(node_from)] = node_to
    for context in self._all_contexts:
      nodes = self._nodes_by_context[context]
      if node_from in nodes:
        nodes[nodes.index(node_from)] = node_to
        context.replace_node(node_from, node_to)
        context.validate(nodes)

  def get_nodes(self, context: dsl_context.DslContext) -> List[_BaseNode]:
    """Gets all BaseNodes that belongs to the context.

    Args:
      context: A DslContext that has been put to the registry.
    Raises:
      ValueError: If the context is unknown to the registry.
    Returns:
      Nodes that belong to the context, possibly empty list.
    """
    if context not in self._all_contexts:
      raise ValueError(f'Context {context} does not exist in the registry.')
    return list(self._nodes_by_context[context])

  def get_contexts(self, node: _BaseNode) -> List[dsl_context.DslContext]:
    """Gets all dsl_context.DslContexts that the node belongs to.

    Args:
      node: A BaseNode that has been put to the registry.
    Raises:
      ValueError: If the node is unknown to the registry.
    Returns:
      List of DslContexts that wraps the node, possibly empty list.
    """
    # This is O(N^2), but not performance critical.
    if node not in self._all_nodes:
      raise ValueError(
          f'Node {_format_node(node)} does not exist in the registry. '
          f'Valid: {", ".join([_format_node(n) for n in self._all_nodes])})')
    result = []
    for context in self._all_contexts:
      if node in self._nodes_by_context[context]:
        result.append(context)
    return result

  def extract_for_pipeline(self, pipeline: _Pipeline) -> 'DslContextRegistry':
    """Creates new registry with pipeline level contexts filtered out.

    This function should be called in the pipeline constructor, i.e., where
    pipeine scope ends, to persist contexts defined within the pipeline scope
    in the pipeline object.

    Args:
      pipeline: A pipeline that may exist in the registry.
    Returns:
      A new DSL context registry with only contexts within the pipeline scope.
    """
    # pylint:disable=protected-access
    result = DslContextRegistry()
    latest_pipeline_level_context = None
    for context in reversed(self._all_contexts):
      if pipeline in self._nodes_by_context[context]:
        latest_pipeline_level_context = context
        break
      else:
        context.parent = None
        result._all_contexts.append(context)
    result._all_contexts.reverse()

    result._active_contexts = []

    if latest_pipeline_level_context:
      result._all_nodes = self._nodes_by_context[latest_pipeline_level_context]
    else:
      result._all_nodes = self._all_nodes

    for context, nodes in self._nodes_by_context.items():
      if context in result._all_contexts:
        result._nodes_by_context[context] = [
            n for n in nodes if n in result._all_nodes
        ]

    result._finalized = False
    # pylint:enable=protected-access
    return result

_registry_holder = threading.local()


def get() -> DslContextRegistry:
  """Gets the current active registry that observes DSL definitions."""
  if not getattr(_registry_holder, 'current', None):
    _registry_holder.current = DslContextRegistry()
  return _registry_holder.current


@contextlib.contextmanager
def use_registry(registry: DslContextRegistry) -> Iterator[DslContextRegistry]:
  """Use the given registry as a global scope."""
  old_registry = get()
  _registry_holder.current = registry
  try:
    with registry.temporary_mutable():
      yield registry
  finally:
    _registry_holder.current = old_registry


@contextlib.contextmanager
def new_registry() -> Iterator[DslContextRegistry]:
  """Push the new registry to the global scope."""
  with use_registry(DslContextRegistry()) as result:
    yield result
