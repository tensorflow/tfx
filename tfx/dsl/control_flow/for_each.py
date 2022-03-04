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
"""Module for `ForEach` context manager."""
from typing import Sequence

import attr

from tfx import types
from tfx.dsl.components.base import base_node
from tfx.dsl.context_managers import dsl_context
from tfx.dsl.context_managers import dsl_context_manager


@attr.s(auto_attribs=True, kw_only=True, hash=False, eq=False)
class ForEachContext(dsl_context.DslContext):
  """DslContext for ForEach."""
  wrapped_channel: types.BaseChannel

  def validate(self, containing_nodes: Sequence[base_node.BaseNode]):
    for parent in self.ancestors:
      if isinstance(parent, ForEachContext):
        raise NotImplementedError('Nested ForEach block is not supported yet.')

    if len(containing_nodes) > 1:
      raise NotImplementedError(
          'Cannot define more than one component within ForEach yet.')

    for node in containing_nodes:
      for input_channel in node.inputs.values():
        if (isinstance(input_channel, types.LoopVarChannel) and
            input_channel.wrapped is self.wrapped_channel):
          break
      else:
        raise ValueError(
            f'Node {node.id} does not use ForEach sliced value and will always '
            'see the same input. Please define the node outside the ForEach '
            'block.')


class ForEach(dsl_context_manager.DslContextManager[types.LoopVarChannel]):
  """ForEach context manager.

  ForEach context manager is a declarative version of For loop in a pipeline
  defintion (in DSL). When some TFX component can generate more than one
  artifacts and want to consume each output artifact separately, ForEach block
  can be used to aid handling multiple artifacts individually.

  ```python
  example_gen = BufferedExampleGen()
  # example_gen.outputs['examples'] containing N artifacts.

  with ForEach(example_gen.outputs['examples']) as examples:
    trainer = Trainer(
        examples=examples,  # instead of using example_gen.outputs['examples'].
        ...)
  ```

  In the above example, only a single Trainer component is declared in the
  pipeline, but it can be executed multiple times (or even zero time) in a
  single pipeline run depending on the number of output artifacts that
  example_gen has generated.
  """

  def __init__(self, channel: types.BaseChannel):
    super().__init__()
    self._channel = channel

  def create_context(self) -> ForEachContext:
    return ForEachContext(wrapped_channel=self._channel)

  def enter(self, context: ForEachContext) -> types.LoopVarChannel:
    return types.LoopVarChannel(self._channel, context)
