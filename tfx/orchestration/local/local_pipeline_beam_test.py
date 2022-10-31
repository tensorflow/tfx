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
"""End to end test for running a beam-powered custom component in local mode.

The component and pipeline pattern in this file are provided only for testing
purposes and are not a recommended way to structure TFX pipelines. We recommend
consulting the TFX Component Tutorial (https://www.tensorflow.org/tfx/tutorials)
for a recommended pipeline topology.
"""

import os
import tempfile

import absl.testing.absltest
import apache_beam as beam
from apache_beam.options import pipeline_options

from tfx.dsl.component.experimental import annotations
from tfx.dsl.component.experimental.decorators import component
from tfx.dsl.placeholder import placeholder as ph
from tfx.orchestration import pipeline as pipeline_py
from tfx.orchestration.local import local_dag_runner
from tfx.orchestration.metadata import sqlite_metadata_connection_config


@component(use_beam=True)
def SimpleBeamPoweredComponent(
    beam_pipeline: annotations.BeamComponentParameter[beam.Pipeline] = None):
  with beam_pipeline as p:
    direct_num_workers = p.options.view_as(
        pipeline_options.DirectOptions).direct_num_workers
    direct_running_mode = p.options.view_as(
        pipeline_options.DirectOptions).direct_running_mode
    LocalDagRunnerTest.BEAM_ARG_VALUES[
        'direct_num_workers'] = direct_num_workers
    LocalDagRunnerTest.BEAM_ARG_VALUES[
        'direct_running_mode'] = direct_running_mode


class LocalDagRunnerTest(absl.testing.absltest.TestCase):

  # Global list of components names that have run, used to confirm
  # execution side-effects in local test.
  RAN_COMPONENTS = []
  # List of beam env vars from placeholders
  BEAM_ARG_VALUES = {}

  def setUp(self):
    super().setUp()
    self.__class__.RAN_COMPONENTS = []
    self.__class__.BEAM_ARG_VALUES = {}

  def _GetTestBeamComponentPipeline(
      self, num_workers_env_var_name,
      direct_running_mode_env_var_name) -> pipeline_py.Pipeline:

    # Construct component instances.
    dummy_beam_component = SimpleBeamPoweredComponent().with_id('Beam')

    # Construct and run pipeline
    temp_path = tempfile.mkdtemp()
    pipeline_root_path = os.path.join(temp_path, 'pipeline_root')
    metadata_path = os.path.join(temp_path, 'metadata.db')
    return pipeline_py.Pipeline(
        pipeline_name='test_pipeline',
        pipeline_root=pipeline_root_path,
        metadata_connection_config=sqlite_metadata_connection_config(
            metadata_path),
        components=[dummy_beam_component],
        beam_pipeline_args=[
            '--runner=DirectRunner',
            '--direct_running_mode=' +
            ph.environment_variable(direct_running_mode_env_var_name),
            ph.environment_variable(num_workers_env_var_name),
        ],
    )

  def testBeamComponentWithPlaceHolderArgs(self):
    # Set env vars for the placeholder
    direct_running_mode_env_var_name = 'DIRECT_RUNNING_MODE'
    direct_running_mode = 'multi_processing'
    direct_num_workers = 2
    num_workers_env_var_name = 'NUM_WORKERS'
    num_workers_env_var_value = f'--direct_num_workers={direct_num_workers}'

    os.environ[direct_running_mode_env_var_name] = direct_running_mode
    os.environ[num_workers_env_var_name] = num_workers_env_var_value

    local_dag_runner.LocalDagRunner().run(
        self._GetTestBeamComponentPipeline(
            num_workers_env_var_name, direct_running_mode_env_var_name))

    self.assertEqual(self.BEAM_ARG_VALUES['direct_num_workers'],
                     direct_num_workers)
    self.assertEqual(self.BEAM_ARG_VALUES['direct_running_mode'],
                     direct_running_mode)


if __name__ == '__main__':
  absl.testing.absltest.main()
