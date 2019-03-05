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
"""TFX StatisticsGen component definition."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, Text

from tfx.components.base import base_component
from tfx.components.base import base_driver
from tfx.components.statistics_gen import executor
from tfx.utils import channel
from tfx.utils import types


class StatisticsGen(base_component.BaseComponent):
  """Official TFX StatisticsGen component.

  The StatisticsGen component wraps Tensorflow Data Validation (tfdv) to
  generate stats for every slice of input examples.


  Attributes:
    outputs: A ComponentOutputs including following keys:
      - output: A channel of 'ExampleStatisticsPath' with statistics for every
        split in input examples.
  """

  def __init__(self,
               input_data,
               name = None,
               outputs = None):
    """Constructs a StatisticsGen component.

    Args:
      input_data: A Channel of 'ExamplesPath' type. This should contain two
        splits 'train' and 'eval'.
      name: Optional unique name. Necessary iff multiple StatisticsGen
        components are declared in the same pipeline.
      outputs: Optional dict from name to output channel.
    """
    component_name = 'StatisticsGen'
    input_dict = {'input_data': channel.as_channel(input_data)}
    exec_properties = {}
    super(StatisticsGen, self).__init__(
        component_name=component_name,
        unique_name=name,
        driver=base_driver.BaseDriver,
        executor=executor.Executor,
        input_dict=input_dict,
        outputs=outputs,
        exec_properties=exec_properties)

  def _create_outputs(self):
    """Creates outputs for StatisticsGen.

    Returns:
      ComponentOutputs object containing the dict of [Text -> Channel]
    """
    # pylint: disable=g-complex-comprehension
    output_artifact_collection = [
        types.TfxType(
            'ExampleStatisticsPath',
            split=split,
        ) for split in types.DEFAULT_EXAMPLE_SPLITS
    ]
    # pylint: enable=g-complex-comprehension
    return base_component.ComponentOutputs({
        'output':
            channel.Channel(
                type_name='ExampleStatisticsPath',
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
    input_dict['input_data'].type_check('ExamplesPath')
