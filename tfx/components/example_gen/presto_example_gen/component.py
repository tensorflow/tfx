# Copyright 2019 Google LLC. All Rights Reserved.
# Copyright 2019 Naver Corp. All Rights Reserved.
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
"""TFX PrestoExampleGen component definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional, Text, Dict

from tfx.components.example_gen import component
from tfx.components.example_gen import utils
from tfx.components.base.base_component import ExecutionParameter, BaseComponent
from tfx.components.example_gen.presto_example_gen import executor
from tfx.proto import example_gen_pb2
from tfx.utils import channel
from tfx.utils import types


class PrestoExampleGenSpec(component.QueryBasedExampleGenSpec):
  """Presto ExampleGen component spec"""

  COMPONENT_NAME = 'PrestoExampleGen'
  PARAMETERS = {
      'input_config': ExecutionParameter(type=example_gen_pb2.Input),
      'output_config': ExecutionParameter(type=example_gen_pb2.Output),
      'connection_info': ExecutionParameter(type=Dict),
  }


class PrestoExampleGen(BaseComponent):  # pylint: disable=protected-access
  """TFX PrestoExampleGen component.

  The Presto examplegen component takes a query, and generates train
  and eval examples for downsteam components.
  """

  SPEC_CLASS = PrestoExampleGenSpec
  EXECUTOR_CLASS = executor.Executor

  def __init__(self,
               query: Optional[Text] = None,
               connection_info: Dict = None,
               input_config: Optional[example_gen_pb2.Input] = None,
               output_config: Optional[example_gen_pb2.Output] = None,
               component_name: Optional[Text] = 'PrestoExampleGen',
               example_artifacts: Optional[channel.Channel] = None,
               name: Optional[Text] = None):
    """Constructs a PrestoExampleGen component.

    Args:
      query: Presto sql string, query result will be treated as a single
        split, can be overwritten by input_config.
      input_config: An example_gen_pb2.Input instance with Split.pattern as
        Presto sql string. If set, it overwrites the 'query' arg, and allows
        different queries per split.
      output_config: An example_gen_pb2.Output instance, providing output
        configuration. If unset, default splits will be 'train' and 'eval' with
        size 2:1.
      example_artifacts: Optional channel of 'ExamplesPath' for output train and
        eval examples.
      name: Optional unique name. Necessary if multiple PrestoExampleGen
        components are declared in the same pipeline.

    Raises:
      RuntimeError: Only one of query and input_config should be set.
    """
    if bool(query) == bool(input_config):
      raise RuntimeError('Exactly one of query and input_config should be set.')
    input_config = input_config or utils.make_default_input_config(query)
    output_config = output_config or utils.make_default_output_config(
        input_config)
    example_artifacts = example_artifacts or channel.as_channel(
        [types.TfxArtifact('ExamplesPath', split=split_name)
         for split_name in utils.generate_output_split_names(
             input_config, output_config)])
    spec = PrestoExampleGenSpec(
        component_name=component_name,
        input_config=input_config,
        output_config=output_config,
        connection_info=connection_info,
        examples=example_artifacts)
    super(PrestoExampleGen, self).__init__(spec=spec, name=name)