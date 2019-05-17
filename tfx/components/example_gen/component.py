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

from typing import Any, Dict, Optional, Text

from tfx.components.base import base_component
from tfx.components.base import base_driver
from tfx.components.example_gen import driver
from tfx.components.example_gen import utils
from tfx.proto import example_gen_pb2
from tfx.utils import channel
from tfx.utils import types
from google.protobuf import json_format


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
  Attributes:
    outputs: A ComponentOutputs including following keys:
      - examples: A channel of 'ExamplesPath' with train and eval examples.

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
               outputs: Optional[base_component.ComponentOutputs] = None):
    if input_base is None and input_config is None:
      raise RuntimeError('One of input_base and input_config must be set.')
    input_dict = {
        'input-base': channel.as_channel(input_base)
    } if input_base else {}
    # Default value need to be set in component instead of executor as output
    # artifacts depend on it.
    self._input_config = input_config or utils.make_default_input_config()
    self._output_config = output_config or utils.make_default_output_config(
        self._input_config)
    exec_properties = {
        'input': json_format.MessageToJson(self._input_config),
        'output': json_format.MessageToJson(self._output_config)
    }
    super(ExampleGen, self).__init__(
        component_name=component_name,
        unique_name=unique_name,
        driver=driver.Driver if input_base else base_driver.BaseDriver,
        executor=executor,
        input_dict=input_dict,
        outputs=outputs,
        exec_properties=exec_properties)

  def _create_outputs(self) -> base_component.ComponentOutputs:
    """Creates outputs for ExampleGen.

    Returns:
      ComponentOutputs object containing the dict of [Text -> Channel]
    """
    output_artifact_collection = [
        types.TfxArtifact('ExamplesPath', split=split_name)
        for split_name in utils.generate_output_split_names(
            self._input_config, self._output_config)
    ]
    return base_component.ComponentOutputs({
        'examples':
            channel.Channel(
                type_name='ExamplesPath',
                static_artifact_collection=output_artifact_collection)
    })

  def _type_check(self, input_dict: Dict[Text, channel.Channel],
                  exec_properties: Dict[Text, Any]) -> None:
    """Does type checking for the inputs and exec_properties.

    Args:
      input_dict: A Dict[Text, Channel] as the inputs of the Component.
      exec_properties: A Dict[Text, Any] as the execution properties of the
        component. Unused right now.

    Raises:
      TypeError: if the type_name of given Channel is different from expected.
    """
    del exec_properties  # Unused right now.
    # TODO(jyzhao): apply the check for all components for codegen.
    if 'input-base' in input_dict:
      input_dict['input-base'].type_check('ExternalPath')
