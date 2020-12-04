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
"""E2E Tests for tfx.examples.custom_components_hello_world."""

import os
from typing import Text

import tensorflow as tf
from tfx.dsl.io import fileio
from tfx.examples.custom_components.hello_world.example import taxi_pipeline_hello
from tfx.orchestration import metadata
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner


class TaxiPipelineHelloEndToEndTest(tf.test.TestCase):

  def setUp(self):
    super(TaxiPipelineHelloEndToEndTest, self).setUp()
    self._test_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    self._pipeline_name = 'hello_test'
    self._data_root = os.path.join(os.path.dirname(__file__), '..', 'data')
    self._pipeline_root = os.path.join(self._test_dir, 'tfx', 'pipelines',
                                       self._pipeline_name)
    self._metadata_path = os.path.join(self._test_dir, 'tfx', 'metadata',
                                       self._pipeline_name, 'metadata.db')

  def assertExecutedOnce(self, component: Text) -> None:
    """Check the component is executed exactly once."""
    component_path = os.path.join(self._pipeline_root, component)
    self.assertTrue(fileio.exists(component_path))
    execution_path = os.path.join(
        component_path, '.system', 'executor_execution')
    execution = fileio.listdir(execution_path)
    self.assertLen(execution, 1)

  def assertPipelineExecution(self) -> None:
    self.assertExecutedOnce('CsvExampleGen')
    self.assertExecutedOnce('HelloComponent')
    self.assertExecutedOnce('StatisticsGen')

  def testTaxiPipelineHello(self):
    BeamDagRunner().run(
        taxi_pipeline_hello._create_pipeline(
            pipeline_name=self._pipeline_name,
            data_root=self._data_root,
            pipeline_root=self._pipeline_root,
            metadata_path=self._metadata_path))

    self.assertTrue(fileio.exists(self._metadata_path))
    metadata_config = metadata.sqlite_metadata_connection_config(
        self._metadata_path)
    with metadata.Metadata(metadata_config) as m:
      artifact_count = len(m.store.get_artifacts())
      execution_count = len(m.store.get_executions())
      self.assertGreaterEqual(artifact_count, execution_count)

    self.assertPipelineExecution()

    # Run pipeline again.
    BeamDagRunner().run(
        taxi_pipeline_hello._create_pipeline(
            pipeline_name=self._pipeline_name,
            data_root=self._data_root,
            pipeline_root=self._pipeline_root,
            metadata_path=self._metadata_path))

    # Assert cache execution.
    with metadata.Metadata(metadata_config) as m:
      # Artifact count is unchanged.
      self.assertEqual(artifact_count, len(m.store.get_artifacts()))

    self.assertPipelineExecution()


if __name__ == '__main__':
  tf.test.main()
