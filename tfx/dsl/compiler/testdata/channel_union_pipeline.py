# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Test pipeline using custom components with channel.union()."""

import json
import os
from typing import Any, Dict, List, Optional

from tfx import types
from tfx.components import CsvExampleGen
from tfx.components import StatisticsGen
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import base_executor
from tfx.dsl.components.base import executor_spec
from tfx.dsl.components.common import resolver
from tfx.dsl.input_resolution.strategies import latest_artifact_strategy
from tfx.dsl.io import fileio
from tfx.orchestration import pipeline
from tfx.types import artifact_utils
from tfx.types import channel
from tfx.types import channel_utils
from tfx.types import standard_artifacts
from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ExecutionParameter
from tfx.utils import io_utils

_pipeline_name = 'channel_union_pipeline'

_data_root = 'channel_union'
_data_root_1 = os.path.join(_data_root, 'data1')
_data_root_2 = os.path.join(_data_root, 'data2')

_tfx_root = os.path.join(_data_root, 'tfx')
_pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)


class Executor(base_executor.BaseExecutor):
  """Executor for ChannelUnionComponent."""

  def Do(self, input_dict: Dict[str, List[types.Artifact]],
         output_dict: Dict[str, List[types.Artifact]],
         exec_properties: Dict[str, Any]) -> None:
    """Copy the input_data to the output_data.

    Args:
      input_dict: Input dict from input key to a list of artifacts, including:
        - input_data: A list of type `standard_artifacts.Examples` which will
          often contain two splits, 'train' and 'eval'.
      output_dict: Output dict from key to a list of artifacts, including:
        - output_data: A list of type `standard_artifacts.Examples` which will
          usually contain the same splits as input_data.
      exec_properties: A dict of execution properties, including:
        - name: Optional unique name. Necessary iff multiple ChannelUnion
          components are declared in the same pipeline.

    Returns:
      None

    Raises:
      OSError and its subclasses
    """
    self._log_startup(input_dict, output_dict, exec_properties)
    input_artifacts = input_dict['input_data']
    assert len(input_artifacts) == 2

    output_artifact = artifact_utils.get_single_instance(
        output_dict['output_data'])
    for input_artifact in input_artifacts:
      output_artifact.split_names = input_artifact.split_names
      split_to_instance = {}

      for split in json.loads(input_artifact.split_names):
        uri = artifact_utils.get_split_uri([input_artifact], split)
        split_to_instance[split] = uri

      for split, instance in split_to_instance.items():
        input_dir = instance
        output_dir = artifact_utils.get_split_uri([output_artifact], split)
        for filename in fileio.listdir(input_dir):
          input_uri = os.path.join(input_dir, filename)
          output_uri = os.path.join(output_dir, filename)
          io_utils.copy_file(src=input_uri, dst=output_uri, overwrite=True)


class ChannelUnionComponentSpec(types.ComponentSpec):
  """ComponentSpec for Custom TFX ChannelUnion World Component."""

  PARAMETERS = {
      # These are parameters that will be passed in the call to
      # create an instance of this component.
      'name': ExecutionParameter(type=str),
  }
  INPUTS = {
      # This will be a dictionary with input artifacts, including URIs
      'input_data': ChannelParameter(type=standard_artifacts.Examples),
  }
  OUTPUTS = {
      # This will be a dictionary which this component will populate
      'output_data': ChannelParameter(type=standard_artifacts.Examples),
  }


class ChannelUnionComponent(base_component.BaseComponent):
  """Custom TFX Component that can handle _UnionChannel.

  This custom component class consists of only a constructor.
  """

  SPEC_CLASS = ChannelUnionComponentSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(Executor)

  def __init__(self,
               input_data: Optional[types.BaseChannel] = None,
               output_data: Optional[types.Channel] = None,
               name: Optional[str] = None):
    """Construct a ChannelUnionComponent.

    Args:
      input_data: A Channel of type `standard_artifacts.Examples`. This will
        often contain two splits: 'train', and 'eval'.
      output_data: A Channel of type `standard_artifacts.Examples`. This will
        usually contain the same splits as input_data.
      name: Optional unique name. Necessary if multiple ChannelUnion components
        are declared in the same pipeline.
    """
    if not output_data:
      output_data = channel_utils.as_channel([standard_artifacts.Examples()])

    spec = ChannelUnionComponentSpec(
        input_data=input_data, output_data=output_data, name=name)
    super().__init__(spec=spec)


def _create_pipeline(pipeline_name: str, pipeline_root: str, data_root_1: str,
                     data_root_2: str) -> pipeline.Pipeline:
  """Implements a pipeline with channel.union()."""
  # Brings data into the pipeline or otherwise joins/converts training data.
  example_gen_1 = CsvExampleGen(input_base=data_root_1).with_id('example_gen_1')
  example_gen_2 = CsvExampleGen(input_base=data_root_2).with_id('example_gen_2')

  channel_union = ChannelUnionComponent(
      input_data=channel.union([
          example_gen_1.outputs['examples'], example_gen_2.outputs['examples']
      ]),
      name='channel_union_input')

  # Get the latest channel.
  latest_artifacts_resolver = resolver.Resolver(
      strategy_class=latest_artifact_strategy.LatestArtifactStrategy,
      resolved_channels=channel.union([
          example_gen_1.outputs['examples'],
          channel_union.outputs['output_data']
      ])).with_id('latest_artifacts_resolver')

  # Computes statistics over data for visualization and example validation.
  statistics_gen = StatisticsGen(
      examples=latest_artifacts_resolver.outputs['resolved_channels'])
  return pipeline.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=[
          example_gen_1, example_gen_2, channel_union,
          latest_artifacts_resolver, statistics_gen
      ])


def create_test_pipeline():
  return _create_pipeline(
      pipeline_name=_pipeline_name,
      pipeline_root=_pipeline_root,
      data_root_1=_data_root_1,
      data_root_2=_data_root_2)
