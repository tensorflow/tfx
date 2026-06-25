# Copyright 2026 Google LLC. All Rights Reserved.
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
"""TFX experimental Filter component definition."""

from typing import Optional

from tfx import types
from tfx.components.experimental.filter import executor
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import executor_spec
from tfx.types import standard_artifacts
from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ComponentSpec
from tfx.types.component_spec import ExecutionParameter


class FilterSpec(ComponentSpec):
  """Filter component spec."""

  PARAMETERS = {
      'filter_fn_path': ExecutionParameter(type=str),
  }
  INPUTS = {
      'examples': ChannelParameter(type=standard_artifacts.Examples),
  }
  OUTPUTS = {
      'filtered_examples': ChannelParameter(type=standard_artifacts.Examples),
  }


class FilterComponent(base_component.BaseComponent):
  """A TFX component to filter examples based on a user-defined function.

  The FilterComponent reads examples from each split of the input `examples`
  artifact, applies a user-defined filter function using an Apache Beam
  pipeline, and writes the filtered examples to the `filtered_examples` output
  artifact, preserving the split structure.

  Example usage:
  ```python
  # Filter out examples where age <= 18
  filter_component = FilterComponent(
      examples=example_gen.outputs['examples'],
      filter_fn_path='my_filters.custom_filter_fn'
  )
  ```
  """

  SPEC_CLASS = FilterSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

  def __init__(self,
               examples: types.BaseChannel,
               filter_fn_path: str,
               filtered_examples: Optional[types.Channel] = None):
    """Construct a FilterComponent.

    Args:
      examples: A [BaseChannel] of type [standard_artifacts.Examples].
      filter_fn_path: The Python import path to the filter function.
        e.g., 'my_module.my_filter_fn'. The function must have the signature:
        `def my_filter_fn(serialized_example: bytes) -> bool`
      filtered_examples: Optional output channel of type [standard_artifacts.Examples].
    """
    if filtered_examples is None:
      filtered_examples = types.Channel(type=standard_artifacts.Examples)

    spec = FilterSpec(
        examples=examples,
        filter_fn_path=filter_fn_path,
        filtered_examples=filtered_examples)
    super().__init__(spec=spec)
