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
"""TFX CsvExampleGen component definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional, Text

from tfx.components.example_gen import component
from tfx.components.example_gen.csv_example_gen import executor
from tfx.proto import example_gen_pb2
from tfx.utils import channel


class CsvExampleGen(component.ExampleGen):
  """Official TFX CsvExampleGen component.

  The csv examplegen component takes csv data, and generates train
  and eval examples for downsteam components.

  Args:
    input_base: A Channel of 'ExternalPath' type, which includes one artifact
      whose uri is an external directory with csv files inside.
    input_config: An example_gen_pb2.Input instance, providing input
      configuration. If unset, the files under input_base will be treated as a
      single split.
    output_config: An example_gen_pb2.Output instance, providing output
      configuration. If unset, default splits will be 'train' and 'eval' with
      size 2:1.
    name: Optional unique name. Necessary if multiple CsvExampleGen components
      are declared in the same pipeline.
    example_artifacts: Optional channel of 'ExamplesPath' for output train and
      eval examples.
  """

  def __init__(self,
               input_base: channel.Channel,
               input_config: Optional[example_gen_pb2.Input] = None,
               output_config: Optional[example_gen_pb2.Output] = None,
               name: Optional[Text] = None,
               example_artifacts: Optional[channel.Channel] = None):
    super(CsvExampleGen, self).__init__(
        executor=executor.Executor,
        input_base=input_base,
        input_config=input_config,
        output_config=output_config,
        component_name='CsvExampleGen',
        unique_name=name,
        example_artifacts=example_artifacts)
