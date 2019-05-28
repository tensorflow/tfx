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

from tfx.components.base import base_component
from tfx.components.base.base_component import ChannelInput
from tfx.components.base.base_component import ChannelOutput
from tfx.components.model_validator import driver
from tfx.components.model_validator import executor
from tfx.utils import channel
from tfx.utils import types


class ModelValidatorSpec(base_component.ComponentSpec):
  """ModelValidator component spec."""

  COMPONENT_NAME = 'ModelValidator'
  PARAMETERS = []
  INPUTS = [
      ChannelInput('examples', type='ExamplesPath'),
      ChannelInput('model', type='ModelExportPath'),
  ]
  OUTPUTS = [
      ChannelOutput('blessing', type='ModelBlessingPath'),
  ]


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
    blessing: Optional output channel of 'ModelBlessingPath' for result of
      blessing.
  """

  def __init__(self,
               examples: channel.Channel,
               model: channel.Channel,
               name: Optional[Text] = None,
               blessing: Optional[channel.Channel] = None):
    if not blessing:
      blessing = channel.Channel(
          type_name='ModelBlessingPath',
          static_artifact_collection=[types.TfxArtifact('ModelBlessingPath')])
    spec = ModelValidatorSpec(
        examples=channel.as_channel(examples),
        model=channel.as_channel(model),
        blessing=blessing)

    super(ModelValidator, self).__init__(
        unique_name=name,
        spec=spec,
        executor=executor.Executor,
        driver=driver.Driver)
