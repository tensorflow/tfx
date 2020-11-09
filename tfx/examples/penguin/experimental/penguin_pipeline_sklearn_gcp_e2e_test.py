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
"""E2E Tests for tfx.examples.experimental.penguin_pipeline_sklearn_gcp."""

import os
from typing import Text

import mock
import tensorflow as tf

from tfx.dsl.io import fileio
from tfx.examples.penguin.experimental import penguin_pipeline_sklearn_gcp
from tfx.extensions.google_cloud_ai_platform.pusher import executor as pusher_executor
from tfx.extensions.google_cloud_ai_platform.trainer import executor as trainer_executor
from tfx.orchestration import metadata
from tfx.orchestration.local.local_dag_runner import LocalDagRunner


class PenguinPipelineSklearnGcpEndToEndTest(tf.test.TestCase):

  def setUp(self):
    super(PenguinPipelineSklearnGcpEndToEndTest, self).setUp()
    self._test_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    self._experimental_root = os.path.dirname(__file__)
    self._penguin_root = os.path.dirname(self._experimental_root)

    self._pipeline_name = 'sklearn_test'
    self._data_root = os.path.join(self._penguin_root, 'data')
    self._module_file = os.path.join(self._experimental_root,
                                     'penguin_utils_sklearn.py')
    self._pipeline_root = os.path.join(self._test_dir, 'tfx', 'pipelines',
                                       self._pipeline_name)
    self._metadata_path = os.path.join(self._test_dir, 'tfx', 'metadata',
                                       self._pipeline_name, 'metadata.db')
    self._ai_platform_training_args = {
        'project': 'project_id',
        'region': 'us-central1',
    }
    self._ai_platform_serving_args = {
        'model_name': 'model_name',
        'project_id': 'project_id',
        'regions': ['us-central1'],
    }

  def assertExecutedOnce(self, component: Text) -> None:
    """Check the component is executed exactly once."""
    component_path = os.path.join(self._pipeline_root, component)
    self.assertTrue(fileio.exists(component_path))
    outputs = fileio.listdir(component_path)
    for output in outputs:
      execution = fileio.listdir(os.path.join(component_path, output))
      self.assertEqual(1, len(execution))

  def assertPipelineExecution(self) -> None:
    self.assertExecutedOnce('CsvExampleGen')
    self.assertExecutedOnce('Pusher')
    self.assertExecutedOnce('SchemaGen')
    self.assertExecutedOnce('StatisticsGen')
    self.assertExecutedOnce('Trainer')

  @mock.patch(
      'tfx.extensions.google_cloud_ai_platform.pusher.executor.discovery'
  )
  @mock.patch.object(trainer_executor, 'runner', autospec=True)
  @mock.patch.object(pusher_executor, 'runner', autospec=True)
  def testPenguinPipelineSklearnGcp(self, mock_pusher, mock_trainer, _):
    mock_pusher.get_service_name_and_api_version.return_value = ('ml', 'v1')
    mock_trainer.get_service_name_and_api_version.return_value = ('ml', 'v1')
    LocalDagRunner().run(
        penguin_pipeline_sklearn_gcp._create_pipeline(
            pipeline_name=self._pipeline_name,
            data_root=self._data_root,
            module_file=self._module_file,
            pipeline_root=self._pipeline_root,
            metadata_path=self._metadata_path,
            ai_platform_training_args=self._ai_platform_training_args,
            ai_platform_serving_args=self._ai_platform_serving_args,
            beam_pipeline_args=[]))

    self.assertTrue(fileio.exists(self._metadata_path))
    mock_trainer.start_aip_training.assert_called_once()
    mock_pusher.deploy_model_for_aip_prediction.assert_called_once()
    expected_execution_count = 6  # 6 components
    metadata_config = metadata.sqlite_metadata_connection_config(
        self._metadata_path)
    with metadata.Metadata(metadata_config) as m:
      artifact_count = len(m.store.get_artifacts())
      execution_count = len(m.store.get_executions())
      self.assertGreaterEqual(artifact_count, execution_count)
      self.assertEqual(expected_execution_count, execution_count)

    self.assertPipelineExecution()

    # Runs pipeline the second time.
    LocalDagRunner().run(
        penguin_pipeline_sklearn_gcp._create_pipeline(
            pipeline_name=self._pipeline_name,
            data_root=self._data_root,
            module_file=self._module_file,
            pipeline_root=self._pipeline_root,
            metadata_path=self._metadata_path,
            ai_platform_training_args=self._ai_platform_training_args,
            ai_platform_serving_args=self._ai_platform_serving_args,
            beam_pipeline_args=[]))

    # All executions but Evaluator and Pusher are cached.
    with metadata.Metadata(metadata_config) as m:
      self.assertEqual(artifact_count, len(m.store.get_artifacts()))
      artifact_count = len(m.store.get_artifacts())
      self.assertEqual(expected_execution_count * 2,
                       len(m.store.get_executions()))

    # Runs pipeline the third time.
    LocalDagRunner().run(
        penguin_pipeline_sklearn_gcp._create_pipeline(
            pipeline_name=self._pipeline_name,
            data_root=self._data_root,
            module_file=self._module_file,
            pipeline_root=self._pipeline_root,
            metadata_path=self._metadata_path,
            ai_platform_training_args=self._ai_platform_training_args,
            ai_platform_serving_args=self._ai_platform_serving_args,
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
