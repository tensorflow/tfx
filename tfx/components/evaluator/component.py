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
"""TFX Evaluator component definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, Optional, Text

from tfx.components.base import base_component
from tfx.components.base import base_driver
from tfx.components.evaluator import executor
from tfx.utils import channel
from tfx.utils import types


class Evaluator(base_component.BaseComponent):
  """Official TFX Evaluator component.

  The evaluator component can be used to perform model evaluations.

  Args:
    examples: A Channel of 'ExamplesPath' type, usually produced by ExampleGen
      component.
    model_exports: A Channel of 'ModelExportPath' type, usually produced by
      Trainer component.
    name: Optional unique name. Necessary if multiple Evaluator components are
      declared in the same pipeline.
    outputs: Optional dict from name to output channel.
  Attributes:
    outputs: A ComponentOutputs including following keys:
      - output: A channel of 'ModelEvalPath' with result of evaluation.
  """

  def __init__(self,
               examples: channel.Channel,
               model_exports: channel.Channel,
               name: Optional[Text] = None,
               outputs: Optional[base_component.ComponentOutputs] = None):
    component_name = 'Evaluator'
    input_dict = {
        'examples': channel.as_channel(examples),
        'model_exports': channel.as_channel(model_exports),
    }
    exec_properties = {}
    super(Evaluator, self).__init__(
        component_name=component_name,
        unique_name=name,
        driver=base_driver.BaseDriver,
        executor=executor.Executor,
        input_dict=input_dict,
        outputs=outputs,
        exec_properties=exec_properties)

  def _create_outputs(self) -> base_component.ComponentOutputs:
    """Creates outputs for Evaluator.

    Returns:
      ComponentOutputs object containing the dict of [Text -> Channel]
    """
    output_artifact_collection = [types.TfxType('ModelEvalPath')]
    return base_component.ComponentOutputs({
        'output':
            channel.Channel(
                type_name='ModelEvalPath',
                static_artifact_collection=output_artifact_collection),
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
    input_dict['examples'].type_check('ExamplesPath')
    input_dict['model_exports'].type_check('ModelExportPath')
