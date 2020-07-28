# Copyright 2020 Google LLC. All Rights Reserved.
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
"""TFX DataViewBinder component definition."""
from typing import Optional, Text

from tfx import types
from tfx.components.base import base_component
from tfx.components.base import executor_spec
from tfx.components.experimental.data_view import binder_executor
from tfx.types import artifact_utils
from tfx.types import channel_utils
from tfx.types import standard_artifacts
from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ComponentSpec


class _DataViewBinderComponentSpec(ComponentSpec):
  """ComponentSpec for Custom TFX Hello World Component."""

  PARAMETERS = {}
  INPUTS = {
      'input_examples': ChannelParameter(type=standard_artifacts.Examples),
      'data_view': ChannelParameter(type=standard_artifacts.DataView),
  }
  OUTPUTS = {
      'output_examples': ChannelParameter(type=standard_artifacts.Examples),
  }


class DataViewBinder(base_component.BaseComponent):
  """A component that binds a DataView to ExamplesArtifact.

  It takes as inputs a channel of Examples and a channel of DataView, and
  binds the DataView (i.e. attaching information from the DataView as custom
  properties) to the Examples in the input channel, producing new Examples
  Artifacts that are identical to the input Examples (including the uris),
  except for the additional information attached.

  Example:
  ```
    # We assume Examples are imported by ExampleGen
    example_gen = ...
    # First, create a dataview:
    data_view_provider = TfGraphDataViewProvider(
        module_file=module_file,
        create_decoder_func='create_decoder')
    # Then, bind the DataView to Examples:
    data_view_binder = DataViewBinder(
        input_examples=example_gen.outputs['examples'],
        data_view=data_view_provider.outputs['data_view'],
        )
    # Downstream component can then consume the output of the DataViewBinder:
    stats_gen = StatisticsGen(
        examples=data_view_binder.outputs['output_examples'], ...)
  ```
  """
  SPEC_CLASS = _DataViewBinderComponentSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(
      binder_executor.DataViewBinderExecutor)

  def __init__(self,
               input_examples: types.Channel,
               data_view: types.Channel,
               output_examples: Optional[types.Channel] = None,
               instance_name: Optional[Text] = None):
    if not output_examples:
      output_artifact = standard_artifacts.Examples()
      output_artifact.copy_from(
          artifact_utils.get_single_instance(list(input_examples.get())))
      output_examples = channel_utils.as_channel([output_artifact])

    spec = _DataViewBinderComponentSpec(
        input_examples=input_examples,
        data_view=data_view,
        output_examples=output_examples)
    super().__init__(spec=spec, instance_name=instance_name)
