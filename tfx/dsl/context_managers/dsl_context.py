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
"""Module for DslContext definition.

See more doc from dsl_context_manager.py.
"""

from typing import Any, Optional, Iterable, cast, Sequence

# Use Any to avoid cyclic import.
_BaseNode = Any


class DslContext:
  """A context whose scope is defined by DslContextManager in a pipeline DSL.

  While DslContextManager (e.g. Cond) is public API and used in a pipeline DSL
  to define a scope of the DslContext, underlying context is not visible to the
  public users.

  DslContext always belongs to a single _Registry (1:n relationship),
  and its id is unique among the registry (can be duplicated with a context
  from a different registry).

  DslContext is associated with BaseNodes that are defined within the context
  manager ("with" block) that has created the context. If the node is defined
  within the multiple context mangers, all active DslContexts are associated
  with the node (n:m relationship).

  DO NOT create and use DslContext directly. DslContext is expected to be only
  created from DslContextManager.create_context.
  """
  # Parent DslContext that is creatd from the closest outer DslContextManger.
  # Will be injected by DslContextManager.
  parent: Optional['DslContext']

  def validate(self, containing_nodes: Sequence[_BaseNode]):
    """Hook method to validate the context with its containing nodes."""

  def replace_node(self, node_from: _BaseNode, node_to: _BaseNode):
    """Hook method for replacing associated node to another."""

  @property
  def ancestors(self) -> Iterable['DslContext']:
    """All ancestor DslContexts in parent -> child order."""
    if self.parent:
      parent = cast(DslContext, self.parent)
      yield from parent.ancestors
      yield parent

  def __eq__(self, other: Any):
    return self is other

  def __hash__(self):
    return hash(id(self))
