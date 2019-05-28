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

from typing import Optional, Text

from tfx.components.base import base_component
from tfx.components.base.base_component import ChannelInput
from tfx.components.base.base_component import ChannelOutput
from tfx.components.statistics_gen import executor
from tfx.utils import channel
from tfx.utils import types


class StatisticsGenSpec(base_component.ComponentSpec):
  """StatisticsGen component spec."""

  COMPONENT_NAME = 'StatisticsGen'
  PARAMETERS = []
  INPUTS = [
      ChannelInput('input_data', type='ExamplesPath'),
  ]
  OUTPUTS = [
      ChannelOutput('output', type='ExampleStatisticsPath'),
  ]


class StatisticsGen(base_component.BaseComponent):
  """Official TFX StatisticsGen component.

  The StatisticsGen component wraps Tensorflow Data Validation (tfdv) to
  generate stats for every slice of input examples.

  Args:
    input_data: A Channel of 'ExamplesPath' type. This should contain two
      splits 'train' and 'eval'.
    name: Optional unique name. Necessary iff multiple StatisticsGen
      components are declared in the same pipeline.
    output: Optional 'ExampleStatisticsPath' channel for statistics of each
      split provided in input examples.
  """

  def __init__(self,
               input_data: channel.Channel,
               name: Text = None,
               output: Optional[channel.Channel] = None):
    if not output:
      output = channel.Channel(
          type_name='ExampleStatisticsPath',
          static_artifact_collection=[
              types.TfxArtifact('ExampleStatisticsPath', split=split)
              for split in types.DEFAULT_EXAMPLE_SPLITS])
    spec = StatisticsGenSpec(
        input_data=channel.as_channel(input_data),
        output=output)

    super(StatisticsGen, self).__init__(
        spec=spec,
        unique_name=name,
        executor=executor.Executor)
