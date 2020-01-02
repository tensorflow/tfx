# Lint as: python3
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
"""Tests for tfx.orchestration.config.config_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tfx.components.base import executor_spec
from tfx.orchestration.config import config_utils
from tfx.orchestration.config import docker_component_config
from tfx.orchestration.config import pipeline_config
from tfx.orchestration.launcher import docker_component_launcher
from tfx.orchestration.launcher import in_process_component_launcher
from tfx.orchestration.launcher import test_utils
from tfx.types import channel_utils


class ConfigUtilsTest(tf.test.TestCase):

  def testFindComponentLaunchInfoReturnDefaultLaunchInfo(self):
    input_artifact = test_utils._InputArtifact()
    component = test_utils._FakeComponent(
        name='FakeComponent',
        input_channel=channel_utils.as_channel([input_artifact]))
    p_config = pipeline_config.PipelineConfig()

    (launcher_class,
     c_config) = config_utils.find_component_launch_info(p_config, component)

    self.assertEqual(in_process_component_launcher.InProcessComponentLauncher,
                     launcher_class)
    self.assertIsNone(c_config)

  def testFindComponentLaunchInfoReturnConfigOverride(self):
    input_artifact = test_utils._InputArtifact()
    component = test_utils._FakeComponent(
        name='FakeComponent',
        input_channel=channel_utils.as_channel([input_artifact]),
        custom_executor_spec=executor_spec.ExecutorContainerSpec(
            image='gcr://test', args=['{{input_dict["input"][0].uri}}']))
    default_config = docker_component_config.DockerComponentConfig()
    override_config = docker_component_config.DockerComponentConfig(name='test')
    p_config = pipeline_config.PipelineConfig(
        supported_launcher_classes=[
            docker_component_launcher.DockerComponentLauncher
        ],
        default_component_configs=[default_config],
        component_config_overrides={
            '_FakeComponent.FakeComponent': override_config
        })

    (launcher_class,
     c_config) = config_utils.find_component_launch_info(p_config, component)

    self.assertEqual(docker_component_launcher.DockerComponentLauncher,
                     launcher_class)
    self.assertEqual(override_config, c_config)

  def testFindComponentLaunchInfoFailWithNoLauncherClassFound(self):
    input_artifact = test_utils._InputArtifact()
    component = test_utils._FakeComponent(
        name='FakeComponent',
        input_channel=channel_utils.as_channel([input_artifact]))
    p_config = pipeline_config.PipelineConfig(supported_launcher_classes=[
        docker_component_launcher.DockerComponentLauncher
    ])

    with self.assertRaises(RuntimeError):
      # DockerComponentLauncher cannot launch class executor.
      config_utils.find_component_launch_info(p_config, component)


if __name__ == '__main__':
  tf.test.main()
