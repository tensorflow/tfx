# Copyright 2023 Google LLC. All Rights Reserved.
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
"""Tests for tfx.dsl.components.common.job_cleanup."""

from absl.testing import absltest
from absl.testing import parameterized
from tfx.dsl.compiler import constants as compiler_constants
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import base_executor
from tfx.dsl.components.base import executor_spec
from tfx.dsl.components.common import job_cleanup
from tfx.types import artifact as tfx_artifact
from tfx.types import channel
from tfx.types import component_spec


class _OutputArtifact(tfx_artifact.Artifact):
  TYPE_NAME = 'OutputArtifact'


class _LaunchOnlySpec(component_spec.ComponentSpec):
  PARAMETERS = {}
  INPUTS = {}
  OUTPUTS = {
      compiler_constants.LAUNCH_ONLY_CHANNEL_NAME: (
          component_spec.ChannelParameter(type=_OutputArtifact)
      ),
  }


class _BasicComponent(base_component.BaseComponent):
  SPEC_CLASS = _LaunchOnlySpec

  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(base_executor.BaseExecutor)

  def __init__(self):
    output = channel.Channel(type=_OutputArtifact)
    spec = _LaunchOnlySpec(
        **{compiler_constants.LAUNCH_ONLY_CHANNEL_NAME: output}
    )
    super().__init__(spec)


class JobCleanupTest(parameterized.TestCase):

  def test_job_cleanup(self):
    launch_component = _BasicComponent().with_id('my_launch_component')

    cleanup = job_cleanup.JobCleanup(launch_component)

    self.assertEqual(
        cleanup.exec_properties['launcher_component'], launch_component.id
    )
    self.assertEqual(
        cleanup.inputs[compiler_constants.LAUNCH_ONLY_CHANNEL_NAME],
        launch_component.outputs[compiler_constants.LAUNCH_ONLY_CHANNEL_NAME],
    )

  @parameterized.named_parameters(
      ('no_channels', {}),
      (
          'multiple_channels',
          {
              'foo': component_spec.ChannelParameter(type=_OutputArtifact),
              compiler_constants.LAUNCH_ONLY_CHANNEL_NAME: (
                  component_spec.ChannelParameter(type=_OutputArtifact)
              ),
          },
      ),
  )
  def test_job_cleanup_throws_with_invalid_launch_component(self, channel_spec):
    class _InvalidLaunchOnlySpec(component_spec.ComponentSpec):
      PARAMETERS = {}
      INPUTS = {}
      OUTPUTS = channel_spec

    class _InvalidLaunchComponent(base_component.BaseComponent):
      SPEC_CLASS = _InvalidLaunchOnlySpec
      EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(
          base_executor.BaseExecutor
      )

      def __init__(self):
        if channel_spec:
          lo_channel = channel.Channel(type=_OutputArtifact)
          foo_channel = channel.Channel(type=_OutputArtifact)
          spec = _InvalidLaunchOnlySpec(**{
              compiler_constants.LAUNCH_ONLY_CHANNEL_NAME: lo_channel,
              'foo': foo_channel,
          })
        else:
          spec = _InvalidLaunchOnlySpec()
        super().__init__(spec)

    invalid_launch_component = _InvalidLaunchComponent().with_id(
        'my_invalid_launch_component'
    )

    with self.assertRaises(ValueError):
      job_cleanup.JobCleanup(invalid_launch_component)


if __name__ == '__main__':
  absltest.main()
