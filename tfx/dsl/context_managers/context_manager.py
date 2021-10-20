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
"""Module for DSL context managers and contexts.

In pipeline DSL, there are advanced semantics using "with" block. Context
managers (DslContextManager; e.g. Cond) are used to define a scope of a context.

    with Cond(predicate):  # Cond is a DslContextManager
      node = Node(...)     # The node is associated with ConditionalContext.

Nodes defined within the context manager are associated with the DslContext
(e.g. ConditionalContext). If with blocks are nested, multiple DslContexts are
associated (order matters). DslContext itself is not a public API.
"""

import abc
import collections
import threading
import types
from typing import Any, Optional, List, Iterable, Type, TypeVar, Generic

import attr

# Cannot use base_node.BaseNode here due to circular dependency.
# String forward ref ('BaseNode') is also not possible as the name is coming
# from the circularly dependent module.
_BaseNode = Any


def _generate_context_id(self) -> str:
  return f'{self.__class__.__name__}:{len(_registry.all_contexts)}'


def _peek_registry() -> Optional['DslContext']:
  return _registry.peek()


@attr.s(auto_attribs=True, kw_only=True)
class DslContext:
  """A context whose scope is defined by DslContextManager in a pipeline DSL.

  While DslContextManager (e.g. Cond) is public API and used in a pipeline DSL
  to define a scope of the DslContext, underlying context is not visible to the
  public users.

  DslContext always belongs to a single _DslContextRegistry (1:n relationship),
  and its id is unique among the registry (can be duplicated with a context
  from a different registry).

  DslContext is associated with BaseNodes that are defined within the context
  manager ("with" block) that has created the context. If the node is defined
  within the multiple context mangers, all active DslContexts are associated
  with the node (n:m relationship).

  DO NOT create and use DslContext directly. DslContext is expected to be only
  created from DslContextManager.create_context.
  """
  # ID that is unique to the registered DslContextRegistry.
  id: str = attr.Factory(_generate_context_id, takes_self=True)
  # Parent DslContext that is creatd from the closest outer DslContextManger.
  parent: Optional['DslContext'] = attr.Factory(_peek_registry)

  def validate(self):
    """Hook method to validate the context."""

  def will_add_node(self, node: _BaseNode):
    """Hook method before adding a node to the context."""

  @property
  def is_background(self):
    return False

  @property
  def ancestors(self) -> Iterable['DslContext']:
    """All ancestor DslContexts in parent -> child order."""
    if self.parent:
      yield from self.parent.ancestors
      yield self.parent

  @property
  def nodes(self) -> List[_BaseNode]:
    """All nodes that are associated with this DslContext."""
    return _registry.get_nodes(self)


class _BackgroundContext(DslContext):
  """BackgroundContext is placed at the bottom of any other contexts."""

  @property
  def is_background(self):
    return True


class _DslContextRegistry(threading.local):
  """Registry for DslContexts and associated BaseNodes from a pipeline DSL.

  _DslContextRegistry manages the active DslContexts, their orders, and
  associations with BaseNodes. DslContext and BaseNode always belong to exactly
  one _DslContextRegistry (1:n relationship).

  Since the registry is a global thread local storage, it is NOT SAFE to define
  multiple pipelines in the same thread, but is SAFE to define a single pipeline
  from each thread.
  """

  def __init__(self):
    super().__init__()
    # Frame of currently active DSL context. Ordered by parent -> child.
    self._active: List[DslContext] = []
    # All DSL contexts that have ever been defined so far.
    self._all: List[DslContext] = []
    # Mapping from Context ID to a list of nodes that belong to each context.
    # Each list of node is sorted chronologically.
    self._nodes_by_context_ids = collections.defaultdict(list)
    self.push(_BackgroundContext(id='background', parent=None))

  @property
  def background_context(self) -> _BackgroundContext:
    return self._all[0]

  @property
  def all_contexts(self) -> List[DslContext]:
    """All contexts defined during the lifespan of the registry."""
    return list(self._all)

  @property
  def active_contexts(self) -> List[DslContext]:
    """All active context frame in parent -> child order."""
    return list(self._active)

  def push(self, context: DslContext):
    self._active.append(context)
    self._all.append(context)

  def pop(self) -> DslContext:
    """Removes the top context from the active context frame."""
    assert len(self._active) > 1, (
        'Internal assertion error; background context should not be removed.')
    return self._active.pop()

  def peek(self) -> Optional[DslContext]:
    """Returns the top context of the active context frame."""
    return self._active[-1] if self._active else None

  def put_node(self, node: _BaseNode) -> None:
    """Associates the node to all active contexts."""
    for context in self._active:
      context.will_add_node(node)
    for context in self._active:
      self._nodes_by_context_ids[context.id].append(node)

  def get_nodes(self, context: Optional[DslContext] = None) -> List[_BaseNode]:
    """Gets all BaseNodes that belongs to the context."""
    if context is None:
      context = self.background_context
    return list(self._nodes_by_context_ids[context.id])

  def get_contexts(self, node: _BaseNode) -> List[DslContext]:
    """Gets all DslContexts that the node belongs to."""
    # This is O(N^2), but not performance critical.
    result = []
    for context in self._all:
      if node in self._nodes_by_context_ids[context.id]:
        result.append(context)
    return result


# Currently we don't have a mechanism to figure out the beginning and the end
# of the pipeline definition. Sub-optimally we use a singleton registry to
# listen to all context manager and node definitions, and regard all of them
# belong to the same pipeline.
_registry = _DslContextRegistry()


def put_node(node: _BaseNode) -> None:
  """Puts BaseNode to the currently active DSL contexts."""
  _registry.put_node(node)


def get_contexts(node: _BaseNode) -> List[DslContext]:
  """Gets all DslContexts that the node belongs to."""
  return _registry.get_contexts(node)


def get_nodes(context: Optional[DslContext] = None) -> List[_BaseNode]:
  """Gets all BaseNode that belongs to the context."""
  return _registry.get_nodes(context)


_Handle = TypeVar('_Handle')


class DslContextManager(Generic[_Handle], abc.ABC):
  """Base class for all context managers for pipeline DSL."""

  @abc.abstractmethod
  def create_context(self) -> DslContext:
    """Creates an underlying DslContext object.

    All nodes defined within this DslContextManager would be associated with
    the created DslContext.

    Since DslContextManager can __enter__ multiple times and each represents
    a different context, the return value should be newly created (not reused).

    Returns:
      Newly created DslContext object.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def enter(self, context: DslContext) -> _Handle:  # pylint: disable=unused-argument
    """Subclass hook method for __enter__.

    Returned value is captured at "with..as" clause. It can be any helper object
    to augment DSL syntax.

    Args:
      context: Newly created DslContext that DslContextManager has created for
          the __enter__().
    """

  def __enter__(self) -> _Handle:
    context = self.create_context()
    context.validate()
    _registry.push(context)
    return self.enter(context)

  def __exit__(self, exc_type: Optional[Type[BaseException]],
               exc_val: Optional[BaseException],
               exc_tb: Optional[types.TracebackType]):
    _registry.pop()
