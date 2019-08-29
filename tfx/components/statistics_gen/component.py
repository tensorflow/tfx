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
from tfx.components.base import executor_spec
from tfx.components.statistics_gen import executor
from tfx.types import artifact
from tfx.types import standard_artifacts
from tfx.types.standard_component_specs import StatisticsGenSpec


class StatisticsGen(base_component.BaseComponent):
  """Official TFX StatisticsGen component.

  The StatisticsGen component generates features statistics and random samples
  over training data, which can be used for visualization and validation.
  StatisticsGen uses Apache Beam and approximate algorithms to scale to large
  datasets.

  Please see https://www.tensorflow.org/tfx/data_validation for more details.

  ## Example
  ```
    # Computes statistics over data for visualization and example validation.
    statistics_gen = StatisticsGen(input_data=example_gen.outputs.examples)
  ```
  """

  SPEC_CLASS = StatisticsGenSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

  def __init__(self,
               input_data: types.Channel = None,
               output: Optional[types.Channel] = None,
               examples: Optional[types.Channel] = None,
               instance_name: Optional[Text] = None):
    """Construct a StatisticsGen component.

    Args:
      input_data: A Channel of `ExamplesPath` type, likely generated by the
        [ExampleGen component](https://www.tensorflow.org/tfx/guide/examplegen).
        This needs to contain two splits labeled `train` and `eval`. _required_
      output: `ExampleStatisticsPath` channel for statistics of each split
        provided in the input examples.
      examples: Forwards compatibility alias for the `input_data` argument.
      instance_name: Optional name assigned to this specific instance of
        StatisticsGen.  Required only if multiple StatisticsGen components are
        declared in the same pipeline.
    """
    input_data = input_data or examples
    output = output or types.Channel(
        type=standard_artifacts.ExampleStatistics,
        artifacts=[
            standard_artifacts.ExampleStatistics(split=split)
            for split in artifact.DEFAULT_EXAMPLE_SPLITS
        ])
    spec = StatisticsGenSpec(
        input_data=input_data, output=output)
    super(StatisticsGen, self).__init__(spec=spec, instance_name=instance_name)
