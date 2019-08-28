# Copyright 2019 Google LLC. All Rights Reserved.
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
"""TFX ExampleGen component definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional, Text

from tfx import types
from tfx.components.base import base_component
from tfx.components.base import base_executor
from tfx.components.base import executor_spec
from tfx.components.example_gen import driver
from tfx.components.example_gen import utils
from tfx.proto import example_gen_pb2
from tfx.types import channel_utils
from tfx.types import standard_artifacts
from tfx.types.standard_component_specs import FileBasedExampleGenSpec
from tfx.types.standard_component_specs import QueryBasedExampleGenSpec


class _QueryBasedExampleGen(base_component.BaseComponent):
  """TFX query-based ExampleGen component base class.

  ExampleGen component takes input data source, and generates train
  and eval example splits (or custom splits) for downsteam components.
  """

  SPEC_CLASS = QueryBasedExampleGenSpec
  # EXECUTOR_SPEC should be overridden by subclasses.
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(base_executor.BaseExecutor)

  def __init__(self,
               input_config: example_gen_pb2.Input,
               output_config: Optional[example_gen_pb2.Output] = None,
               custom_config: Optional[example_gen_pb2.CustomConfig] = None,
               component_name: Optional[Text] = 'ExampleGen',
               example_artifacts: Optional[types.Channel] = None,
               name: Optional[Text] = None):
    """Construct an QueryBasedExampleGen component.

    Args:
      input_config: An example_gen_pb2.Input instance, providing input
        configuration.
      output_config: An example_gen_pb2.Output instance, providing output
        configuration. If unset, default splits will be 'train' and 'eval' with
        size 2:1.
      custom_config: An optional example_gen_pb2.CustomConfig instance,
        providing custom configuration for executor.
      component_name: Name of the component, should be unique per component
        class. Default to 'ExampleGen', can be overwritten by sub-classes.
      example_artifacts: Optional channel of 'ExamplesPath' for output train and
        eval examples.
      name: Unique name for every component class instance.
    """
    # Configure outputs.
    output_config = output_config or utils.make_default_output_config(
        input_config)
    example_artifacts = example_artifacts or channel_utils.as_channel([
        standard_artifacts.Examples(split=split_name)
        for split_name in utils.generate_output_split_names(
            input_config, output_config)
    ])
    spec = QueryBasedExampleGenSpec(
        input_config=input_config,
        output_config=output_config,
        custom_config=custom_config,
        examples=example_artifacts)
    super(_QueryBasedExampleGen, self).__init__(spec=spec, name=name)


class FileBasedExampleGen(base_component.BaseComponent):
  """TFX file-based ExampleGen component base class.

  ExampleGen component takes input data source, and generates train
  and eval example splits (or custom splits) for downsteam components.
  """

  SPEC_CLASS = FileBasedExampleGenSpec
  # EXECUTOR_SPEC should be overridden by subclasses.
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(base_executor.BaseExecutor)
  DRIVER_CLASS = driver.Driver

  def __init__(
      self,
      input_base: types.Channel = None,
      input_config: Optional[example_gen_pb2.Input] = None,
      output_config: Optional[example_gen_pb2.Output] = None,
      custom_config: Optional[example_gen_pb2.CustomConfig] = None,
      component_name: Optional[Text] = 'ExampleGen',
      example_artifacts: Optional[types.Channel] = None,
      custom_executor_spec: Optional[executor_spec.ExecutorSpec] = None,
      input: Optional[types.Channel] = None,  # pylint: disable=redefined-builtin
      name: Optional[Text] = None):
    """Construct a FileBasedExampleGen component.

    Args:
      input_base: A Channel of 'ExternalPath' type, which includes one artifact
        whose uri is an external directory with data files inside (required).
      input_config: An optional example_gen_pb2.Input instance, providing input
        configuration. If unset, the files under input_base (must set) will be
        treated as a single split.
      output_config: An optional example_gen_pb2.Output instance, providing
        output configuration. If unset, default splits will be 'train' and
        'eval' with size 2:1.
      custom_config: An optional example_gen_pb2.CustomConfig instance,
        providing custom configuration for executor.
      component_name: Name of the component, should be unique per component
        class. Default to 'ExampleGen', can be overwritten by sub-classes.
      example_artifacts: Optional channel of 'ExamplesPath' for output train and
        eval examples.
      custom_executor_spec: Optional custom executor spec overriding the default
        executor spec specified in the component attribute.
      input: Forwards compatibility alias for the 'input_base' argument.
      name: Unique name for every component class instance.
    """
    input_base = input_base or input
    # Configure inputs and outputs.
    input_config = input_config or utils.make_default_input_config()
    output_config = output_config or utils.make_default_output_config(
        input_config)
    example_artifacts = example_artifacts or channel_utils.as_channel([
        standard_artifacts.Examples(split=split_name)
        for split_name in utils.generate_output_split_names(
            input_config, output_config)
    ])
    spec = FileBasedExampleGenSpec(
        input_base=input_base,
        input_config=input_config,
        output_config=output_config,
        custom_config=custom_config,
        examples=example_artifacts)
    super(FileBasedExampleGen, self).__init__(
        spec=spec, custom_executor_spec=custom_executor_spec, name=name)
