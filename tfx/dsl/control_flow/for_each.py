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

import attr

from tfx import types
from tfx.dsl.components.base import base_node
from tfx.dsl.context_managers import context_manager


@attr.s(auto_attribs=True, kw_only=True)
class ForEachContext(context_manager.DslContext):
  """DslContext for ForEach."""
  wrapped_channel: types.BaseChannel

  def validate(self):
    for parent in self.ancestors:
      if isinstance(parent, ForEachContext):
        raise NotImplementedError('Nested ForEach block is not supported yet.')

  def will_add_node(self, node: base_node.BaseNode):
    if self.nodes:
      raise NotImplementedError(
          'Defining multiple nodes inside ForEach block is not supported yet. '
          'Try placing a node in a separate ForEach block.')

    # Check the iterating channel is directly used.
    # Ex:
    #     with ForEach(a.outputs['aa']) as aa:
    #       b = B(aa=aa)
    #
    # Currently we're only allowing one component under ForEach block so the
    # component must directly use the LoopVarChannel, but as we allow multiple
    # components and nested ForEach block, we need to allow indirect usage of
    # LoopVarChannel as well.
    for input_channel in node.inputs.values():
      if (isinstance(input_channel, types.LoopVarChannel) and
          input_channel.wrapped is self.wrapped_channel):
        break
    else:
      raise ValueError(
          f'Cannot define {node.id} within the ForEach block if it does not '
          'use the iterating channel as an input. Please define the node '
          'outside the ForEach block.')


class ForEach(context_manager.DslContextManager[types.LoopVarChannel]):
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
    self._channel = channel

  def create_context(self) -> ForEachContext:
    return ForEachContext(wrapped_channel=self._channel)

  def enter(self, context: ForEachContext) -> types.LoopVarChannel:
    return types.LoopVarChannel(self._channel, context.id)
