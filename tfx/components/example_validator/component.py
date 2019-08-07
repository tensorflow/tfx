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

from tfx import types
from tfx.components.base import base_component
from tfx.components.base.base_component import ChannelParameter
from tfx.components.example_validator import executor
from tfx.types import channel_utils
from tfx.types import standard_artifacts


class ExampleValidatorSpec(base_component.ComponentSpec):
  """ExampleValidator component spec."""

  PARAMETERS = {}
  INPUTS = {
      'stats': ChannelParameter(type=standard_artifacts.ExampleStatistics),
      'schema': ChannelParameter(type=standard_artifacts.Schema),
  }
  OUTPUTS = {
      'output':
          ChannelParameter(type=standard_artifacts.ExampleValidationResult),
  }


class ExampleValidator(base_component.BaseComponent):
  """Official TFX ExampleValidator component.

  The ExampleValidator component uses Tensorflow Data Validation (tfdv) to
  validate the statistics of some splits on input examples against a schema.
  """

  SPEC_CLASS = ExampleValidatorSpec
  EXECUTOR_CLASS = executor.Executor

  def __init__(self,
               stats: types.Channel,
               schema: types.Channel,
               output: Optional[types.Channel] = None,
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
    output = output or types.Channel(
        type=standard_artifacts.ExampleValidationResult,
        artifacts=[standard_artifacts.ExampleValidationResult()])
    spec = ExampleValidatorSpec(
        stats=channel_utils.as_channel(stats),
        schema=channel_utils.as_channel(schema),
        output=output)
    super(ExampleValidator, self).__init__(spec=spec, name=name)
