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

from typing import Optional, Text

from tfx.components.base import base_component
from tfx.components.base.base_component import ChannelParameter
from tfx.components.example_validator import executor
from tfx.utils import channel
from tfx.utils import types


class ExampleValidatorSpec(base_component.ComponentSpec):
  """ExampleValidator component spec."""

  COMPONENT_NAME = 'ExampleValidator'
  PARAMETERS = {}
  INPUTS = {
      'stats': ChannelParameter(type_name='ExampleStatisticsPath'),
      'schema': ChannelParameter(type_name='SchemaPath'),
  }
  OUTPUTS = {
      'output': ChannelParameter(type_name='ExampleValidationPath'),
  }


class ExampleValidator(base_component.BaseComponent):
  """Official TFX ExampleValidator component.

  The ExampleValidator component uses Tensorflow Data Validation (tfdv) to
  validate the statistics of some splits on input examples against a schema.
  """

  SPEC_CLASS = ExampleValidatorSpec
  EXECUTOR_CLASS = executor.Executor

  def __init__(self,
               stats: channel.Channel,
               schema: channel.Channel,
               output: Optional[channel.Channel] = None,
               name: Optional[Text] = None):
    """Construct an ExampleValidator component.

    Args:
      stats: A Channel of 'ExampleStatisticsPath' type. This should contain at
        least 'eval' split. Other splits are ignored currently.
      schema: A Channel of "SchemaPath' type.
      output: Optional output channel of 'ExampleValidationPath' type.
      name: Optional unique name. Necessary iff multiple ExampleValidator
        components are declared in the same pipeline.
    """
    output = output or channel.Channel(
        type_name='ExampleValidationPath',
        artifacts=[
            types.TfxArtifact('ExampleValidationPath')])
    spec = ExampleValidatorSpec(
        stats=channel.as_channel(stats),
        schema=channel.as_channel(schema),
        output=output)
    super(ExampleValidator, self).__init__(spec=spec, name=name)
