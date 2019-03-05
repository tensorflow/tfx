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
"""TFX ExampleValidator component definition."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, Text

from tfx.components.base import base_component
from tfx.components.base import base_driver
from tfx.components.schema_gen import executor
from tfx.utils import channel
from tfx.utils import types


class SchemaGen(base_component.BaseComponent):
  """Official TFX SchemaGen component.

  The SchemaGen component uses Tensorflow Data Validation (tfdv) to
  generate a schema from input statistics.

  Attributes:
    outputs: A ComponentOutputs including following keys:
      - output: A channel of 'SchemaPath' type.
  """

  def __init__(self,
               stats,
               name = None,
               outputs = None):
    """Constructs a SchemaGen component.

    Args:
      stats: A Channel of 'ExampleStatisticsPath' type. This should contain at
        least 'train' split. Other splits are ignored currently.
      name: Optional unique name. Necessary iff multiple SchemaGen components
        are declared in the same pipeline.
      outputs: Optional dict from name to output channel.
    """
    component_name = 'SchemaGen'
    input_dict = {'stats': channel.as_channel(stats)}
    exec_properties = {}
    super(SchemaGen, self).__init__(
        component_name=component_name,
        unique_name=name,
        driver=base_driver.BaseDriver,
        executor=executor.Executor,
        input_dict=input_dict,
        outputs=outputs,
        exec_properties=exec_properties)

  def _create_outputs(self):
    """Creates outputs for ExampleValidator.

    Returns:
      ComponentOutputs object containing the dict of [Text -> Channel]
    """
    output_artifact_collection = [types.TfxType('SchemaPath')]
    return base_component.ComponentOutputs({
        'output':
            channel.Channel(
                type_name='SchemaPath',
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
      TypeError if the type_name of given Channel is different from expected.
    """
    del exec_properties  # Unused right now.
    input_dict['stats'].type_check('ExampleStatisticsPath')
