# Copyright 2022 Google LLC. All Rights Reserved.
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
"""Tests for utils."""
import tensorflow as tf
from tfx import types
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import base_executor
from tfx.dsl.components.base import executor_spec
from tfx.dsl.experimental.node_execution_options import utils
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import channel
from tfx.types import component_spec
from tfx.types import standard_artifacts


class _BasicComponentSpec(types.ComponentSpec):
  PARAMETERS = {}
  INPUTS = {}
  OUTPUTS = {
      'examples':
          component_spec.ChannelParameter(type=standard_artifacts.Examples)
  }


class _BasicComponent(base_component.BaseComponent):

  SPEC_CLASS = _BasicComponentSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(base_executor.BaseExecutor)

  def __init__(self, component_spec_args):
    super().__init__(_BasicComponentSpec(**component_spec_args))


class UtilsTest(tf.test.TestCase):

  def test_execution_options(self):
    component = _BasicComponent(component_spec_args={
        'examples': channel.Channel(standard_artifacts.Examples)
    })
    component.node_execution_options = utils.NodeExecutionOptions(
        trigger_strategy=pipeline_pb2.NodeExecutionOptions
        .ALL_UPSTREAM_NODES_COMPLETED,
        success_optional=True,
        max_execution_retries=-1,
        execution_timeout_sec=100)
    self.assertEqual(
        component.node_execution_options,
        utils.NodeExecutionOptions(
            trigger_strategy=pipeline_pb2.NodeExecutionOptions
            .ALL_UPSTREAM_NODES_COMPLETED,
            success_optional=True,
            max_execution_retries=0,
            execution_timeout_sec=100))
    component.node_execution_options = None
    self.assertIsNone(component.node_execution_options)


if __name__ == '__main__':
  tf.test.main()
