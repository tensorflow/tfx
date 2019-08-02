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

from typing import Optional, Text

from tfx import types
from tfx.components.base import base_component
from tfx.components.base.base_component import ChannelParameter
from tfx.components.model_validator import driver
from tfx.components.model_validator import executor
from tfx.utils import channel


class ModelValidatorSpec(base_component.ComponentSpec):
  """ModelValidator component spec."""

  PARAMETERS = {}
  INPUTS = {
      'examples': ChannelParameter(type_name='ExamplesPath'),
      'model': ChannelParameter(type_name='ModelExportPath'),
  }
  OUTPUTS = {
      'blessing': ChannelParameter(type_name='ModelBlessingPath'),
  }


class ModelValidator(base_component.BaseComponent):
  """Official TFX ModelValidator component.

  The model validator component can be used to check model metrics threshold
  and validate current model against preivously blessed model. If there isn't
  blessed model yet, model validator will just make sure the threshold passed.

  This component includes a custom driver to resolve last blessed model.
  """

  SPEC_CLASS = ModelValidatorSpec
  EXECUTOR_CLASS = executor.Executor
  DRIVER_CLASS = driver.Driver

  def __init__(self,
               examples: channel.Channel,
               model: channel.Channel,
               blessing: Optional[channel.Channel] = None,
               name: Optional[Text] = None):
    """Construct a ModelValidator component.

    Args:
      examples: A Channel of 'ExamplesPath' type, usually produced by ExampleGen
        component.
      model: A Channel of 'ModelExportPath' type, usually produced by Trainer
        component.
      blessing: Optional output channel of 'ModelBlessingPath' for result of
        blessing.
      name: Optional unique name. Necessary if multiple ModelValidator
        components are declared in the same pipeline.
    """
    blessing = blessing or channel.Channel(
        type_name='ModelBlessingPath',
        artifacts=[types.Artifact('ModelBlessingPath')])
    spec = ModelValidatorSpec(
        examples=channel.as_channel(examples),
        model=channel.as_channel(model),
        blessing=blessing)
    super(ModelValidator, self).__init__(spec=spec, name=name)
