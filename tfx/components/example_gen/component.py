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

from typing import Any, Optional, Text

from tfx.components.base import base_component
from tfx.components.base.base_component import ChannelInput
from tfx.components.base.base_component import ChannelOutput
from tfx.components.base.base_component import Parameter
from tfx.components.example_gen import driver
from tfx.components.example_gen import utils
from tfx.proto import example_gen_pb2
from tfx.utils import channel
from tfx.utils import types


class ExampleGenSpec(base_component.ComponentSpec):
  """ExampleGen component spec."""

  COMPONENT_NAME = 'ExampleGen'
  PARAMETERS = [
      Parameter('input_config', type=example_gen_pb2.Input),
      Parameter('output_config', type=example_gen_pb2.Output),
  ]
  INPUTS = []
  OUTPUTS = [
      ChannelOutput('examples', type='ExamplesPath'),
  ]


class FilebasedExampleGenSpec(base_component.ComponentSpec):
  """File-based ExampleGen component spec."""

  COMPONENT_NAME = 'ExampleGen'
  PARAMETERS = [
      Parameter('input_config', type=example_gen_pb2.Input),
      Parameter('output_config', type=example_gen_pb2.Output),
  ]
  INPUTS = [
      ChannelInput('input_base', type='ExternalPath'),
  ]
  OUTPUTS = [
      ChannelOutput('examples', type='ExamplesPath'),
  ]


class ExampleGen(base_component.BaseComponent):
  """Official TFX ExampleGen component.

  ExampleGen component takes input data source, and generates train
  and eval example splits (or custom splits) for downsteam components.

  Args:
    executor: Executor class to do the real execution work.
    input_base: Optional, for file based example gen only. A Channel of
      'ExternalPath' type, which includes one artifact whose uri is an external
      directory with data files inside.
    input_config: An example_gen_pb2.Input instance, providing input
      configuration. If unset, the files under input_base (must set) will be
      treated as a single split.
    output_config: An example_gen_pb2.Output instance, providing output
      configuration. If unset, default splits will be 'train' and 'eval' with
      size 2:1.
    component_name: Name of the component, should be unique per component class.
      Default to 'ExampleGen', can be overwritten by sub-classes.
    unique_name: Unique name for every component class instance.
    outputs: Optional dict from name to output channel.
    example_artifacts: Optional channel of 'ExamplesPath' for output train and
      eval examples.

  Raises:
    RuntimeError: If both input_base and input_config are unset.
  """

  def __init__(self,
               executor: Any,
               input_base: Optional[channel.Channel] = None,
               input_config: Optional[example_gen_pb2.Input] = None,
               output_config: Optional[example_gen_pb2.Output] = None,
               component_name: Optional[Text] = 'ExampleGen',
               unique_name: Optional[Text] = None,
               example_artifacts: Optional[channel.Channel] = None):
    if input_base is None and input_config is None:
      raise RuntimeError('One of input_base or input_config must be set.')

    # Configure inputs and outputs.
    input_config = input_config or utils.make_default_input_config()
    output_config = (
        output_config or utils.make_default_output_config(input_config))
    if not example_artifacts:
      example_artifacts = channel.as_channel(
          [types.TfxArtifact('ExamplesPath', split=split_name)
           for split_name in utils.generate_output_split_names(
               input_config, output_config)])

    # Configure ComponentSpec.
    custom_driver = None
    if input_base:
      spec = FilebasedExampleGenSpec(
          component_name=component_name,
          input_base=input_base,
          input_config=input_config,
          output_config=output_config,
          examples=example_artifacts)
      # A custom driver is needed for file-based ExampleGen.
      custom_driver = driver.Driver
    else:
      spec = ExampleGenSpec(
          component_name=component_name,
          input_config=input_config,
          output_config=output_config,
          examples=example_artifacts)

    super(ExampleGen, self).__init__(
        unique_name=unique_name, spec=spec, executor=executor,
        driver=custom_driver)
