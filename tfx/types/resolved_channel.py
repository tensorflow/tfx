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
"""Module for ResolvedChannel.

The main purpose of this module is to break the cyclic import dependency.
"""

import dataclasses

from typing import Any, Optional, Type, Set, Sequence, Mapping

from tfx.dsl.control_flow import for_each_internal
from tfx.dsl.input_resolution import resolver_op
from tfx.types import artifact
from tfx.types import channel
from tfx.utils import doc_controls
from tfx.utils import typing_utils


# TODO(b/259604560): Make Invocation more general, e.g. to handle tracing
# other function calls.
@dataclasses.dataclass(frozen=True)
class Invocation:
  """Stores resolver function invocation details for later reconstruction.

  Attributes:
    function: The called object.
    args: The non-keyword arguments to the resolver function.
    kwargs: The keyword argument dictionary to the resolver function.
  """
  function: Any
  args: Sequence[Any]
  kwargs: Mapping[str, Any]


@doc_controls.do_not_generate_docs
class ResolvedChannel(channel.BaseChannel):
  """A BaseChannel that refers to the resolver function's output.

  For example:

      trainer_inputs = trainer_resolver_function(
          examples=example_gen.outputs['examples'])

      trainer = Trainer(examples=trainer_inputs['examples'])

  then the `trainer_inputs['examples']` is a ResolvedChannel instance.
  """

  def __init__(
      self,
      artifact_type: Type[artifact.Artifact],
      output_node: resolver_op.Node,
      output_key: Optional[str] = None,
      invocation: Optional[Invocation] = None,
      for_each_context: Optional[for_each_internal.ForEachContext] = None):
    super().__init__(artifact_type)
    self._output_node = output_node
    self._output_key = output_key
    self._invocation = invocation
    self._for_each_context = for_each_context

  def get_data_dependent_node_ids(self) -> Set[str]:
    result = set()
    for input_node in resolver_op.get_input_nodes(self._output_node):
      wrapped = input_node.wrapped
      if isinstance(wrapped, channel.BaseChannel):
        result.update(wrapped.get_data_dependent_node_ids())
      elif typing_utils.is_compatible(
          wrapped, Mapping[str, channel.BaseChannel]
      ):
        for chan in wrapped.values():
          result.update(chan.get_data_dependent_node_ids())
    return result

  @property
  def output_node(self) -> resolver_op.Node:
    return self._output_node

  @property
  def output_key(self) -> Optional[str]:
    return self._output_key

  @property
  def for_each_context(self) -> Optional[for_each_internal.ForEachContext]:
    return self._for_each_context

  @property
  def invocation(self) -> Invocation:
    return self._invocation

  def __repr__(self) -> str:
    debug_str = str(self._output_node)
    if self._for_each_context is not None:
      debug_str = f'ForEach({debug_str})'
    if self._output_key is not None:
      debug_str += f'["{self._output_key}"]'
    return f'ResolvedChannel(artifact_type={self.type_name}, {debug_str})'
