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

from typing import Union

from tfx.dsl.context_managers import dsl_context_manager
from tfx.dsl.control_flow import for_each_internal
from tfx.types import channel as channel_types

_Loopable = Union[channel_types.BaseChannel, for_each_internal.Loopable]


# Re-export name for backward compatibility.
ForEachContext = for_each_internal.ForEachContext


def _channel_as_loopable(
    channel: channel_types.BaseChannel) -> for_each_internal.Loopable:

  # TODO(b/239761275): Change to ResolvedChannel for simplicity.
  def factory(context: for_each_internal.ForEachContext):
    context.wrapped_channel = channel
    return channel_types.LoopVarChannel(channel, context)

  return for_each_internal.Loopable(factory)


class ForEach(dsl_context_manager.DslContextManager[for_each_internal.LoopVar]):
  """ForEach context manager.

  ForEach context manager is a declarative version of For loop in a pipeline
  defintion (in DSL). When some TFX components generate more than one artifacts,
  or resolver function returns a multiple inputs, ForEach is used to handle
  each artifact or input dictionary individually.

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

  ResolverFunction can also output a loopable which can be accessed from the
  handle as a dict of channels:

  ```python
  evaluator_inputs = sliding_window(examples=examples)

  with ForEach(evaluator_inputs) as each_input:
    evaluator = Evaluator(
        examples=each_input['examples'],
        model=trainer.outputs['model'],
        ...
    )
  ```
  """

  def __init__(self, loopable: _Loopable):
    super().__init__()
    if isinstance(loopable, channel_types.BaseChannel):
      self._loopable = _channel_as_loopable(loopable)
    elif isinstance(loopable, for_each_internal.Loopable):
      self._loopable = loopable
    else:
      raise ValueError(f'{loopable} cannot be used as a ForEach argument.')

  def create_context(self) -> ForEachContext:
    return for_each_internal.ForEachContext()

  def enter(  # pytype: disable=signature-mismatch  # overriding-parameter-type-checks
      self, context: ForEachContext) -> for_each_internal.LoopVar:
    return self._loopable.get_loop_var(context)
