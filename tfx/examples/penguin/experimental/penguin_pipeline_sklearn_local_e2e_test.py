# Copyright 2021 Google LLC. All Rights Reserved.
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
"""E2E Tests for tfx.examples.experimental.penguin_pipeline_sklearn_local."""

import os
import unittest

import tensorflow as tf
from tfx import v1 as tfx
from tfx.examples.penguin.experimental import penguin_pipeline_sklearn_local
from tfx.orchestration import metadata


@unittest.skipIf(tf.__version__ < '2',
                 'Uses keras Model only compatible with TF 2.x')
class PenguinPipelineSklearnLocalEndToEndTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._test_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    self._experimental_root = os.path.dirname(__file__)
    self._penguin_root = os.path.dirname(self._experimental_root)

    self._pipeline_name = 'sklearn_test'
    self._data_root = os.path.join(self._penguin_root, 'data')
    self._trainer_module_file = os.path.join(
        self._experimental_root, 'penguin_utils_sklearn.py')
    self._evaluator_module_file = os.path.join(
        self._experimental_root, 'sklearn_predict_extractor.py')
    self._serving_model_dir = os.path.join(self._test_dir, 'serving_model')
    self._pipeline_root = os.path.join(self._test_dir, 'tfx', 'pipelines',
                                       self._pipeline_name)
    self._metadata_path = os.path.join(self._test_dir, 'tfx', 'metadata',
                                       self._pipeline_name, 'metadata.db')

  def assertExecutedOnce(self, component: str) -> None:
    """Check the component is executed exactly once."""
    component_path = os.path.join(self._pipeline_root, component)
    self.assertTrue(tfx.dsl.io.fileio.exists(component_path))
    execution_path = os.path.join(
        component_path, '.system', 'executor_execution')
    execution = tfx.dsl.io.fileio.listdir(execution_path)
    self.assertLen(execution, 1)

  def assertPipelineExecution(self) -> None:
    self.assertExecutedOnce('CsvExampleGen')
    self.assertExecutedOnce('Evaluator')
    self.assertExecutedOnce('ExampleValidator')
    self.assertExecutedOnce('Pusher')
    self.assertExecutedOnce('SchemaGen')
    self.assertExecutedOnce('StatisticsGen')
    self.assertExecutedOnce('Trainer')

  def testPenguinPipelineSklearnLocal(self):
    tfx.orchestration.LocalDagRunner().run(
        penguin_pipeline_sklearn_local._create_pipeline(
            pipeline_name=self._pipeline_name,
            pipeline_root=self._pipeline_root,
            data_root=self._data_root,
            trainer_module_file=self._trainer_module_file,
            evaluator_module_file=self._evaluator_module_file,
            serving_model_dir=self._serving_model_dir,
            metadata_path=self._metadata_path,
            beam_pipeline_args=[]))

    self.assertTrue(tfx.dsl.io.fileio.exists(self._serving_model_dir))
    self.assertTrue(tfx.dsl.io.fileio.exists(self._metadata_path))
    expected_execution_count = 8  # 7 components + 1 resolver
    metadata_config = (
        tfx.orchestration.metadata.sqlite_metadata_connection_config(
            self._metadata_path))
    with metadata.Metadata(metadata_config) as m:
      artifact_count = len(m.store.get_artifacts())
      execution_count = len(m.store.get_executions())
      self.assertGreaterEqual(artifact_count, execution_count)
      self.assertEqual(expected_execution_count, execution_count)

    self.assertPipelineExecution()


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
