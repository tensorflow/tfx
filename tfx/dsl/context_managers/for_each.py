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
"""Module for `ForEach` definition.

go/tfx-foreach

`ForEach` context manager is a declarative version of For loop in a pipeline
defintion (in DSL). When some TFX component can generate more than one artifacts
and want to consume each output artifact separately, `ForEach` syntax can be
used to aid handling multiple artifacts individually.

```python
example_gen = BufferedExampleGen()
# example_gen.outputs['examples'] containing N artifacts.

with ForEach(example_gen.outputs['examples']) as examples:
  trainer = Trainer(
      examples=examples,  # instead of using example_gen.outputs['examples'].
      ...)
```

In the above example, you're defining a single `Trainer` component in the
pipeline, but it can be executed multiple times (or even zero time) in a single
pipeline run depending on the number of output artifacts `example_gen` has
generated.
"""

import attr

from tfx import types
from tfx.dsl.components.base import base_node
from tfx.dsl.context_managers import context_manager


class SlicedChannel(types.Channel):
  """A wrapper channel to be used as a loop variable in ForEach block."""

  def __init__(self, wrapped: types.Channel, context_id: str):
    self._wrapped = wrapped
    self._context_id = context_id
    super().__init__(
        type=wrapped.type,
        additional_properties=wrapped.additional_properties,
        additional_custom_properties=wrapped.additional_custom_properties,
        producer_component_id=wrapped.producer_component_id,
        output_key=wrapped.output_key)

  @property
  def wrapped(self) -> types.Channel:
    return self._wrapped

  @property
  def context_id(self) -> str:
    return self._context_id

  @property
  def producer_component_id(self) -> str:
    return self._wrapped.producer_component_id

  @producer_component_id.setter
  def producer_component_id(self, value: str) -> None:
    self._wrapped.producer_component_id = value

  @property
  def output_key(self) -> str:
    return self._wrapped.output_key

  @output_key.setter
  def output_key(self, value: str) -> None:
    self._wrapped.output_key = value


@attr.s(auto_attribs=True, kw_only=True)
class ForEachContext(context_manager.DslContext):
  """DslContext for ForEach."""
  channel: types.Channel

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
    # component must directly use the SlicedChannel, but as we allow multiple
    # components and nested ForEach block, we need to allow indirect usage of
    # SlicedChannel as well.
    for maybe_sliced_channel in node.inputs.values():
      if (isinstance(maybe_sliced_channel, SlicedChannel) and
          maybe_sliced_channel.wrapped is self.channel):
        break
    else:
      raise ValueError(
          f'Cannot define {node.id} within the ForEach block if it does not '
          'use the iterating channel as an input. Please define the node '
          'outside the ForEach block.')


class ForEach(context_manager.DslContextManager[SlicedChannel]):
  """ForEach context manager.

  See module comments for more information.
  """

  def __init__(self, channel: types.Channel):
    self._channel = channel

  def create_context(self) -> ForEachContext:
    return ForEachContext(channel=self._channel)

  def enter(self, context: ForEachContext) -> SlicedChannel:
    return SlicedChannel(self._channel, context.id)
