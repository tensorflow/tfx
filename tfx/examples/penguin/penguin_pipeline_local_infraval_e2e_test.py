# Copyright 2020 Google LLC. All Rights Reserved.
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
"""E2E Tests for tfx.examples.penguin.penguin_pipeline_local_infraval."""

import os
from typing import Text
import unittest

import tensorflow as tf

from tfx.dsl.components.base import base_driver
from tfx.dsl.io import fileio
from tfx.examples.penguin import penguin_pipeline_local_infraval
from tfx.orchestration import metadata
from tfx.orchestration.local.local_dag_runner import LocalDagRunner


@unittest.skipIf(tf.__version__ < '2',
                 'Uses keras Model only compatible with TF 2.x')
class PenguinPipelineLocalInfravalEndToEndTest(tf.test.TestCase):

  def setUp(self):
    super(PenguinPipelineLocalInfravalEndToEndTest, self).setUp()
    self._test_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    self._pipeline_name = 'penguin_test'
    self._data_root = os.path.join(os.path.dirname(__file__), 'data')
    self._module_file = os.path.join(
        os.path.dirname(__file__), 'penguin_utils.py')
    self._serving_model_dir = os.path.join(self._test_dir, 'serving_model')
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

  def assertInfraValidatorPassed(self) -> None:
    infra_validator_path = os.path.join(self._pipeline_root, 'InfraValidator')
    blessing_path = os.path.join(self._pipeline_root, 'InfraValidator',
                                 'blessing')
    executions = fileio.listdir(blessing_path)
    self.assertGreaterEqual(len(executions), 1)
    for exec_id in executions:
      blessing_uri = base_driver._generate_output_uri(  # pylint: disable=protected-access
          infra_validator_path, 'blessing', exec_id)
      blessed = os.path.join(blessing_uri, 'INFRA_BLESSED')
      self.assertTrue(fileio.exists(blessed))

  def assertPipelineExecution(self) -> None:
    self.assertExecutedOnce('CsvExampleGen')
    self.assertExecutedOnce('Evaluator')
    self.assertExecutedOnce('ExampleValidator')
    self.assertExecutedOnce('InfraValidator')
    self.assertExecutedOnce('Pusher')
    self.assertExecutedOnce('SchemaGen')
    self.assertExecutedOnce('StatisticsGen')
    self.assertExecutedOnce('Trainer')
    self.assertExecutedOnce('Transform')

  def testPenguinPipelineLocal(self):
    LocalDagRunner().run(
        penguin_pipeline_local_infraval._create_pipeline(
            pipeline_name=self._pipeline_name,
            data_root=self._data_root,
            module_file=self._module_file,
            serving_model_dir=self._serving_model_dir,
            pipeline_root=self._pipeline_root,
            metadata_path=self._metadata_path,
            beam_pipeline_args=[]))

    self.assertTrue(fileio.exists(self._serving_model_dir))
    self.assertTrue(fileio.exists(self._metadata_path))
    expected_execution_count = 10  # 9 components + 1 resolver
    metadata_config = metadata.sqlite_metadata_connection_config(
        self._metadata_path)
    with metadata.Metadata(metadata_config) as m:
      artifact_count = len(m.store.get_artifacts())
      execution_count = len(m.store.get_executions())
      self.assertGreaterEqual(artifact_count, execution_count)
      self.assertEqual(expected_execution_count, execution_count)

    self.assertPipelineExecution()
    self.assertInfraValidatorPassed()

    # Runs pipeline the second time.
    LocalDagRunner().run(
        penguin_pipeline_local_infraval._create_pipeline(
            pipeline_name=self._pipeline_name,
            data_root=self._data_root,
            module_file=self._module_file,
            serving_model_dir=self._serving_model_dir,
            pipeline_root=self._pipeline_root,
            metadata_path=self._metadata_path,
            beam_pipeline_args=[]))

    # All executions but Evaluator and Pusher are cached.
    with metadata.Metadata(metadata_config) as m:
      # Artifact count is increased by 3 caused by Evaluator and Pusher.
      self.assertEqual(artifact_count + 3, len(m.store.get_artifacts()))
      artifact_count = len(m.store.get_artifacts())
      self.assertEqual(expected_execution_count * 2,
                       len(m.store.get_executions()))

    # Runs pipeline the third time.
    LocalDagRunner().run(
        penguin_pipeline_local_infraval._create_pipeline(
            pipeline_name=self._pipeline_name,
            data_root=self._data_root,
            module_file=self._module_file,
            serving_model_dir=self._serving_model_dir,
            pipeline_root=self._pipeline_root,
            metadata_path=self._metadata_path,
            beam_pipeline_args=[]))

    # Asserts cache execution.
    with metadata.Metadata(metadata_config) as m:
      # Artifact count is unchanged.
      self.assertEqual(artifact_count, len(m.store.get_artifacts()))
      self.assertEqual(expected_execution_count * 3,
                       len(m.store.get_executions()))


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
