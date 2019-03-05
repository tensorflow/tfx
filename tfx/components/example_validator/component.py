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
from tfx.components.example_validator import executor
from tfx.utils import channel
from tfx.utils import types


class ExampleValidator(base_component.BaseComponent):
  """Official TFX ExampleValidator component.

  The ExampleValidator component uses Tensorflow Data Validation (tfdv) to
  validate the statistics of some splits on input examples against a schema.

  Args:
    stats: A Channel of 'ExampleStatisticsPath' type. This should contain at
      least 'eval' split. Other splits are ignored currently.
    schema: A Channel of "SchemaPath' type.
    name: Optional unique name. Necessary iff multiple ExampleValidator
      components are declared in the same pipeline.
    output: Optional dict from name to output channel.
  Attributes:
    outputs: A ComponentOutputs including following keys:
      - output: A channel of 'ExampleValidationPath' type.
  """

  def __init__(self,
               stats,
               schema,
               name = None,
               outputs = None):
    """Construct an ExampleValidator component.

    Args:
      stats: A Channel of 'ExampleStatisticsPath' type. This should contain at
        least 'eval' split. Other splits are ignored currently.
      schema: A Channel of "SchemaPath' type.
      name: Optional unique name. Necessary iff multiple ExampleValidator
        components are declared in the same pipeline.
      outputs: Optional dict from name to output channel.
    """
    # TODO(zhitaoli): Move all constants to a constants.py.
    component_name = 'ExampleValidator'
    input_dict = {
        'stats': channel.as_channel(stats),
        'schema': channel.as_channel(schema)
    }
    exec_properties = {}
    super(ExampleValidator, self).__init__(
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
    output_artifact_collection = [types.TfxType('ExampleValidationPath',)]
    return base_component.ComponentOutputs({
        'output':
            channel.Channel(
                type_name='ExampleValidationPath',
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
    del exec_properties
    input_dict['stats'].type_check('ExampleStatisticsPath')
    input_dict['schema'].type_check('SchemaPath')
