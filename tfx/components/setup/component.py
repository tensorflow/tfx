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
"""TFX Setup component definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from typing import Any, Dict, List, Text

from tfx import types
from tfx.components.base import base_component
from tfx.components.base import base_driver
from tfx.components.base import base_executor
from tfx.orchestration import data_types
from tfx.utils import channel


class SetupDriver(base_driver.BaseDriver):
  """Custom driver for Setup."""

  def pre_context(self, pipeline_info: data_types.PipelineInfo) -> None:
    """Create run context with run id as context name."""
    tf.logging.info('Run id: %s', pipeline_info.run_id)
    # TODO(b/139067056): create context and link it with executions.
    # TODO(b/136481432): store DAG in context and PipelineInfo.

  def pre_execution(
      self,
      input_dict: Dict[Text, channel.Channel],
      output_dict: Dict[Text, channel.Channel],
      exec_properties: Dict[Text, Any],
      driver_args: data_types.DriverArgs,
      pipeline_info: data_types.PipelineInfo,
      component_info: data_types.ComponentInfo,
  ) -> data_types.ExecutionDecision:
    assert not input_dict
    assert not output_dict
    assert not exec_properties
    execution_id = self._metadata_handler.register_execution(
        exec_properties={},
        pipeline_info=pipeline_info,
        component_info=component_info)
    self.pre_context(pipeline_info)
    # Setup won't be cached.
    return data_types.ExecutionDecision({}, {}, {}, execution_id, False)


class SetupExecutor(base_executor.BaseExecutor):
  """Custom executor for Setup."""

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    pass


class SetupSpec(base_component.ComponentSpec):
  """Setup component spec."""

  PARAMETERS = {}
  INPUTS = {}
  OUTPUTS = {}


class Setup(base_component.BaseComponent):
  """Setup component which runs before all other components."""

  SPEC_CLASS = SetupSpec
  DRIVER_CLASS = SetupDriver
  EXECUTOR_CLASS = SetupExecutor

  def __init__(self):
    """Constructs the Setup component."""
    super(Setup, self).__init__(spec=SetupSpec())
