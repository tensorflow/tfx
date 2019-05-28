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
from tfx.components.base.base_component import ChannelInput
from tfx.components.base.base_component import ChannelOutput
from tfx.components.schema_gen import executor
from tfx.utils import channel
from tfx.utils import types


class SchemaGenSpec(base_component.ComponentSpec):
  """SchemaGen component spec."""

  COMPONENT_NAME = 'SchemaGen'
  PARAMETERS = []
  INPUTS = [
      ChannelInput('stats', type='ExampleStatisticsPath'),
  ]
  OUTPUTS = [
      ChannelOutput('output', type='SchemaPath'),
  ]


class SchemaGen(base_component.BaseComponent):
  """Official TFX SchemaGen component.

  The SchemaGen component uses Tensorflow Data Validation (tfdv) to
  generate a schema from input statistics.

  Attributes:
    outputs: A ComponentOutputs including following keys:
      - output: A channel of 'SchemaPath' type.
  """

  def __init__(self,
               stats: channel.Channel,
               name: Text = None,
               output: Optional[channel.Channel] = None):
    """Constructs a SchemaGen component.

    Args:
      stats: A Channel of 'ExampleStatisticsPath' type. This should contain at
        least 'train' split. Other splits are ignored currently.
      name: Optional unique name. Necessary iff multiple SchemaGen components
        are declared in the same pipeline.
      output: Optional output 'SchemaPath' channel for schema result.
    """
    if not output:
      output = channel.Channel(
          type_name='SchemaPath',
          static_artifact_collection=[types.TfxArtifact('SchemaPath')])
    spec = SchemaGenSpec(
        stats=channel.as_channel(stats),
        output=output)
    super(SchemaGen, self).__init__(
        unique_name=name,
        spec=spec,
        executor=executor.Executor)
