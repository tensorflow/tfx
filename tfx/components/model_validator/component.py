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
"""TFX ModelValidator component definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, Optional, Text

from tfx.components.base import base_component
from tfx.components.model_validator import driver
from tfx.components.model_validator import executor
from tfx.utils import channel
from tfx.utils import types


class ModelValidator(base_component.BaseComponent):
  """Official TFX ModelValidator component.

  The model validator component can be used to check model metrics threshold
  and validate current model against preivously blessed model. If there isn't
  blessed model yet, model validator will just make sure the threshold passed.

  This component includes a custom driver to resolve last blessed model.

  Args:
    examples: A Channel of 'ExamplesPath' type, usually produced by ExampleGen
      component.
    model: A Channel of 'ModelExportPath' type, usually produced by Trainer
      component.
    name: Optional unique name. Necessary if multiple ModelValidator components
      are declared in the same pipeline.
    outputs: Optional dict from name to output channel.
  Attributes:
    outputs: A ComponentOutputs including following keys:
      - blessing: A channel of 'ModelBlessingPath' with result of blessing.
      - results: A channel of 'ModelValidationPath' with result of validation.
  """

  def __init__(self,
               examples,
               model,
               name = None,
               outputs = None):
    component_name = 'ModelValidator'
    input_dict = {
        'examples': channel.as_channel(examples),
        'model': channel.as_channel(model),
    }
    exec_properties = {
        'blessed_model': None,  # This will be set in driver.
        'blessed_model_id': None,  # This will be set in driver.
    }
    super(ModelValidator, self).__init__(
        component_name=component_name,
        unique_name=name,
        driver=driver.Driver,
        executor=executor.Executor,
        input_dict=input_dict,
        outputs=outputs,
        exec_properties=exec_properties)

  def _create_outputs(self):
    """Creates outputs for ModelValidator.

    Returns:
      ComponentOutputs object containing the dict of [Text -> Channel]
    """
    blessing_artifact_collection = [types.TfxType('ModelBlessingPath')]
    results_artifact_collection = [types.TfxType('ModelValidationPath')]
    return base_component.ComponentOutputs({
        'blessing':
            channel.Channel(
                type_name='ModelBlessingPath',
                static_artifact_collection=blessing_artifact_collection),
        'results':
            channel.Channel(
                type_name='ModelValidationPath',
                static_artifact_collection=results_artifact_collection),
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
    input_dict['examples'].type_check('ExamplesPath')
    input_dict['model'].type_check('ModelExportPath')
