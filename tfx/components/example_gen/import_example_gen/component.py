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
"""TFX ImportExampleGen component definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, Optional, Text

from tfx.components.base import base_component
from tfx.components.example_gen import driver
from tfx.components.example_gen import utils
from tfx.components.example_gen.import_example_gen import executor
from tfx.proto import example_gen_pb2
from tfx.utils import channel
from tfx.utils import types
from google.protobuf import json_format


class ImportExampleGen(base_component.BaseComponent):
  """Official TFX ImportExampleGen component.

  The ImportExampleGen component takes TFRecord files with TF Example data
  format, and generates train and eval examples for downsteam components.
  This component provides consistent and configurable partition, and it also
  shuffle the dataset for ML best practice.

  Args:
    input_base: A Channel of 'ExternalPath' type, which includes one artifact
      whose uri is an external directory with TFRecord files inside.
    input_config: An example_gen_pb2.Input instance, providing input
      configuration. If unset, the files under input_base will be treated as a
      single split.
    output_config: An example_gen_pb2.Output instance, providing output
      configuration. If unset, default splits will be 'train' and 'eval' with
      size 2:1.
    name: Optional unique name. Necessary if multiple ImportExampleGen
      components are declared in the same pipeline.
    outputs: Optional dict from name to output channel.
  Attributes:
    outputs: A ComponentOutputs including following keys:
      - examples: A channel of 'ExamplesPath' with train and eval examples.
  """

  def __init__(self,
               input_base,
               input_config = None,
               output_config = None,
               name = None,
               outputs = None):
    component_name = 'ImportExampleGen'
    input_dict = {'input-base': channel.as_channel(input_base)}
    # Default value need to be set in component instead of executor as output
    # artifacts depend on it.
    self._input_config = input_config or utils.make_default_input_config()
    self._output_config = output_config or utils.make_default_output_config(
        self._input_config)
    exec_properties = {
        'input': json_format.MessageToJson(self._input_config),
        'output': json_format.MessageToJson(self._output_config)
    }
    super(ImportExampleGen, self).__init__(
        component_name=component_name,
        unique_name=name,
        driver=driver.Driver,
        executor=executor.Executor,
        input_dict=input_dict,
        outputs=outputs,
        exec_properties=exec_properties)

  def _create_outputs(self):
    """Creates outputs for ImportExampleGen.

    Returns:
      ComponentOutputs object containing the dict of [Text -> Channel]
    """
    output_artifact_collection = [
        types.TfxType('ExamplesPath', split=split_name)
        for split_name in utils.generate_output_split_names(
            self._input_config, self._output_config)
    ]
    return base_component.ComponentOutputs({
        'examples':
            channel.Channel(
                type_name='ExamplesPath',
                static_artifact_collection=output_artifact_collection)
    })

  def _type_check(self, input_dict,
                  exec_properties):
    """Does type checking for the inputs and exec_properties.

    Args:
      input_dict: A Dict[Text, Channel] as the inputs of the Component.
      exec_properties: A Dict[Text, Any] as the execution properties of the
        component. Unused right now.

    Raises:
      TypeError: if the type_name of given Channel is different from expected.
    """
    del exec_properties  # Unused right now.
    input_dict['input-base'].type_check('ExternalPath')
