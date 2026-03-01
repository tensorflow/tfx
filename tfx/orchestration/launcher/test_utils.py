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
"""Common code shared by test code."""

import os
from typing import Any, Dict, List, Optional

from tfx import types
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import base_driver
from tfx.dsl.components.base import base_executor
from tfx.dsl.components.base import executor_spec
from tfx.dsl.io import fileio
from tfx.orchestration import data_types
from tfx.types import artifact_utils
from tfx.types import channel_utils
from tfx.types import component_spec


class _InputArtifact(types.Artifact):
  TYPE_NAME = 'InputArtifact'


class _OutputArtifact(types.Artifact):
  TYPE_NAME = 'OutputArtifact'


class _FakeDriver(base_driver.BaseDriver):
  """Fake driver for testing purpose only."""

  def pre_execution(  # pytype: disable=signature-mismatch  # overriding-parameter-type-checks
      self,
      input_dict: Dict[str, types.Channel],
      output_dict: Dict[str, types.Channel],
      exec_properties: Dict[str, Any],
      driver_args: data_types.DriverArgs,
      pipeline_info: data_types.PipelineInfo,
      component_info: data_types.ComponentInfo,
  ) -> data_types.ExecutionDecision:
    input_artifacts = channel_utils.unwrap_channel_dict(input_dict)
    output_artifacts = channel_utils.unwrap_channel_dict(output_dict)

    # Generating missing output artifact URIs
    for name, artifacts in output_artifacts.items():
      for idx, artifact in enumerate(artifacts):
        if not artifact.uri:
          suffix = str(idx + 1) if idx > 0 else ''
          artifact.uri = os.path.join(
              pipeline_info.pipeline_root, 'artifacts', name + suffix, 'data',
          )
          fileio.makedirs(os.path.dirname(artifact.uri))

    return data_types.ExecutionDecision(input_artifacts, output_artifacts,
                                        exec_properties, 123, False)


class _FakeExecutor(base_executor.BaseExecutor):
  """Fake executor for testing purpose only."""

  def Do(self, input_dict: Dict[str, List[types.Artifact]],
         output_dict: Dict[str, List[types.Artifact]],
         exec_properties: Dict[str, Any]) -> None:
    input_path = artifact_utils.get_single_uri(input_dict['input'])
    output_path = artifact_utils.get_single_uri(output_dict['output'])
    fileio.copy(input_path, output_path)


class _FakeComponentSpec(types.ComponentSpec):
  """Fake component spec for testing purpose only."""
  PARAMETERS = {}
  INPUTS = {'input': component_spec.ChannelParameter(type=_InputArtifact)}
  OUTPUTS = {'output': component_spec.ChannelParameter(type=_OutputArtifact)}


class _FakeComponent(base_component.BaseComponent):
  """Fake component for testing purpose only."""
  SPEC_CLASS = _FakeComponentSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(_FakeExecutor)
  DRIVER_CLASS = _FakeDriver

  def __init__(
      self,
      name: str,
      input_channel: types.Channel,
      output_channel: Optional[types.Channel] = None,
      custom_executor_spec: Optional[executor_spec.ExecutorSpec] = None):
    output_channel = output_channel or types.Channel(
        type=_OutputArtifact).set_artifacts([_OutputArtifact()])
    spec = _FakeComponentSpec(input=input_channel, output=output_channel)
    super().__init__(spec=spec, custom_executor_spec=custom_executor_spec)
    self._id = '{}.{}'.format(self.__class__.__name__, name)
