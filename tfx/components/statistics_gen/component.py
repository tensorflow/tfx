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

from tfx import types
from tfx.components.base import base_component
from tfx.components.base.base_component import ChannelParameter
from tfx.components.statistics_gen import executor
from tfx.types import channel_utils
from tfx.types import standard_artifacts


class StatisticsGenSpec(base_component.ComponentSpec):
  """StatisticsGen component spec."""

  PARAMETERS = {}
  INPUTS = {
      'input_data': ChannelParameter(type=standard_artifacts.Examples),
  }
  OUTPUTS = {
      'output': ChannelParameter(type=standard_artifacts.ExampleStatistics),
  }


class StatisticsGen(base_component.BaseComponent):
  """Official TFX StatisticsGen component.

  The StatisticsGen component wraps Tensorflow Data Validation (tfdv) to
  generate stats for every slice of input examples.
  """

  SPEC_CLASS = StatisticsGenSpec
  EXECUTOR_CLASS = executor.Executor

  def __init__(self,
               input_data: types.Channel = None,
               output: Optional[types.Channel] = None,
               name: Optional[Text] = None):
    """Construct a StatisticsGen component.

    Args:
      input_data: A Channel of 'ExamplesPath' type. This should contain two
        splits 'train' and 'eval' (required if spec is not passed).
      output: Optional 'ExampleStatisticsPath' channel for statistics of each
        split provided in input examples.
      name: Optional unique name. Necessary iff multiple StatisticsGen
        components are declared in the same pipeline.
    """
    output = output or types.Channel(
        type=standard_artifacts.ExampleStatistics,
        artifacts=[
            standard_artifacts.ExampleStatistics(split=split)
            for split in types.DEFAULT_EXAMPLE_SPLITS
        ])
    spec = StatisticsGenSpec(
        input_data=channel_utils.as_channel(input_data), output=output)
    super(StatisticsGen, self).__init__(spec=spec, name=name)
