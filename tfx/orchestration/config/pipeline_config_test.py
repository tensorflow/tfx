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
"""Tests for tfx.orchestration.config.pipeline_config."""

import tensorflow as tf
from tfx.orchestration.config import docker_component_config
from tfx.orchestration.config import pipeline_config
from tfx.orchestration.launcher import in_process_component_launcher


class PipelineConfigTest(tf.test.TestCase):

  def testInitSucceed(self):
    # Init with default parameters
    pipeline_config.PipelineConfig()
    # Init with custom parameters
    pipeline_config.PipelineConfig(
        supported_launcher_classes=[
            in_process_component_launcher.InProcessComponentLauncher
        ],
        default_component_configs=[
            docker_component_config.DockerComponentConfig()
        ],
        component_config_overrides={
            'comp-1', docker_component_config.DockerComponentConfig()
        })

  def testInitFailWithDupLauncherClasses(self):
    with self.assertRaises(ValueError):
      pipeline_config.PipelineConfig(supported_launcher_classes=[
          in_process_component_launcher.InProcessComponentLauncher,
          in_process_component_launcher.InProcessComponentLauncher,
      ])

  def testInitFailWithDupDefaultComponentConfigClasses(self):
    with self.assertRaises(ValueError):
      pipeline_config.PipelineConfig(default_component_configs=[
          docker_component_config.DockerComponentConfig(),
          docker_component_config.DockerComponentConfig(),
      ])


if __name__ == '__main__':
  tf.test.main()
