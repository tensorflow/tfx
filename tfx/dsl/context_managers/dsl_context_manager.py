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
"""Module for DslContextManager definition.

In pipeline DSL, there are advanced semantics using "with" block. Context
managers (DslContextManager; e.g. Cond) are used to define a scope of a context.

    with Cond(predicate):  # Cond is a DslContextManager
      node = Node(...)     # The node is associated with CondContext.

Nodes defined within the context manager are associated with the DslContext
(e.g. CondContext). If with blocks are nested, multiple DslContexts are
associated (order matters). DslContext is not directly exposed to user, but
instead DslContextManager can define arbitrary handle that is captured in
with-as block.
"""

import abc
import types
from typing import TypeVar, Generic, Optional, Type

from tfx.dsl.context_managers import dsl_context
from tfx.dsl.context_managers import dsl_context_registry

_Handle = TypeVar('_Handle')


class DslContextManager(Generic[_Handle], abc.ABC):
  """Base class for all context managers for pipeline DSL."""

  def __init__(self):
    self._pushed = None

  @abc.abstractmethod
  def create_context(self) -> dsl_context.DslContext:
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
  def enter(self, context: dsl_context.DslContext) -> _Handle:  # pylint: disable=unused-argument
    """Subclass hook method for __enter__.

    Returned value is captured at "with..as" clause. It can be any helper object
    to augment DSL syntax.

    Args:
      context: Newly created DslContext that DslContextManager has created for
          the __enter__().
    """

  def __enter__(self) -> _Handle:
    if self._pushed:
      raise RuntimeError(f'{self} is already in use.')

    reg = dsl_context_registry.get()
    context = self.create_context()
    context.parent = reg.peek_context()
    reg.push_context(context)
    result = self.enter(context)
    self._pushed = context
    return result

  def __exit__(self, exc_type: Optional[Type[BaseException]],
               exc_val: Optional[BaseException],
               exc_tb: Optional[types.TracebackType]):
    if self._pushed:
      reg = dsl_context_registry.get()
      assert reg.peek_context() == self._pushed
      context = reg.pop_context()
      self._pushed = None
      context.validate(reg.get_nodes(context))
