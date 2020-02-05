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
"""E2E Tests for tfx.orchestration.launcher.docker_component_launcher."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow.compat.v1 as tf

from tfx.components.base import base_component
from tfx.components.base import executor_spec
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.beam import beam_dag_runner
from tfx.orchestration.config import docker_component_config
from tfx.orchestration.config import pipeline_config
from tfx.orchestration.launcher import docker_component_launcher
from tfx.types import component_spec


class _HelloWorldSpec(component_spec.ComponentSpec):
  INPUTS = {}
  OUTPUTS = {}
  PARAMETERS = {
      'name': component_spec.ExecutionParameter(type=str),
  }


class _HelloWorldComponent(base_component.BaseComponent):

  SPEC_CLASS = _HelloWorldSpec
  EXECUTOR_SPEC = executor_spec.ExecutorContainerSpec(
      # TODO(b/143965964): move the image to private repo if the test is flaky
      # due to docker hub.
      image='alpine:latest',
      command=['echo'],
      args=['hello {{exec_properties.name}}'])

  def __init__(self, name):
    super(_HelloWorldComponent, self).__init__(_HelloWorldSpec(name=name))


# TODO(hongyes): Add more complicated samples to pass inputs/outputs between
# containers.
def _create_pipeline(
    pipeline_name,
    pipeline_root,
    metadata_path,
    name,
):
  hello_world = _HelloWorldComponent(name=name)

  return pipeline.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=[hello_world],
      enable_cache=True,
      metadata_connection_config=metadata.sqlite_metadata_connection_config(
          metadata_path),
      additional_pipeline_args={},
  )


class DockerComponentLauncherE2eTest(tf.test.TestCase):

  def setUp(self):
    super(DockerComponentLauncherE2eTest, self).setUp()
    self._test_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    self._pipeline_name = 'docker_e2e_test'
    self._pipeline_root = os.path.join(self._test_dir, 'tfx', 'pipelines',
                                       self._pipeline_name)
    self._metadata_path = os.path.join(self._test_dir, 'tfx', 'metadata',
                                       self._pipeline_name, 'metadata.db')

  def testDockerComponentLauncherInBeam(self):

    beam_dag_runner.BeamDagRunner(
        config=pipeline_config.PipelineConfig(
            supported_launcher_classes=[
                docker_component_launcher.DockerComponentLauncher
            ],
            default_component_configs=[
                docker_component_config.DockerComponentConfig()
            ])).run(
                _create_pipeline(
                    pipeline_name=self._pipeline_name,
                    pipeline_root=self._pipeline_root,
                    metadata_path=self._metadata_path,
                    name='docker_e2e_test_in_beam'))

    metadata_config = metadata.sqlite_metadata_connection_config(
        self._metadata_path)
    with metadata.Metadata(metadata_config) as m:
      self.assertEqual(1, len(m.store.get_executions()))


if __name__ == '__main__':
  tf.test.main()
