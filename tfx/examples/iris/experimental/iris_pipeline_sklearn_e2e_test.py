# Lint as: python3
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
"""E2E Tests for tfx.examples.experimental.iris_pipeline_sklearn."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Text

import mock
import tensorflow as tf

from tfx.examples.iris.experimental import iris_pipeline_sklearn
from tfx.extensions.google_cloud_ai_platform.pusher import executor
from tfx.orchestration import metadata
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner


class IrisPipelineSklearnEndToEndTest(tf.test.TestCase):

  def setUp(self):
    super(IrisPipelineSklearnEndToEndTest, self).setUp()
    self._test_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    self._experimental_root = os.path.dirname(__file__)
    self._iris_root = os.path.dirname(self._experimental_root)

    self._pipeline_name = 'sklearn_test'
    self._data_root = os.path.join(self._iris_root, 'data')
    self._module_file = os.path.join(
        self._experimental_root, 'iris_utils_sklearn.py')
    self._serving_model_dir = os.path.join(self._test_dir, 'serving_model')
    self._pipeline_root = os.path.join(self._test_dir, 'tfx', 'pipelines',
                                       self._pipeline_name)
    self._metadata_path = os.path.join(self._test_dir, 'tfx', 'metadata',
                                       self._pipeline_name, 'metadata.db')
    self._ai_platform_serving_args = {
        'model_name': 'model_name',
        'project_id': 'project_id',
        'regions': ['us-central1'],
    }
    self._executor = executor.Executor()

  def assertExecutedOnce(self, component: Text) -> None:
    """Check the component is executed exactly once."""
    component_path = os.path.join(self._pipeline_root, component)
    self.assertTrue(tf.io.gfile.exists(component_path))
    outputs = tf.io.gfile.listdir(component_path)
    for output in outputs:
      execution = tf.io.gfile.listdir(os.path.join(component_path, output))
      self.assertEqual(1, len(execution))

  def assertPipelineExecution(self) -> None:
    self.assertExecutedOnce('CsvExampleGen')
    self.assertExecutedOnce('Pusher')
    self.assertExecutedOnce('SchemaGen')
    self.assertExecutedOnce('StatisticsGen')
    self.assertExecutedOnce('Trainer')
    self.assertExecutedOnce('Transform')

  @mock.patch.object(executor, 'runner', autospec=True)
  def testIrisPipelineSklearn(self, mock_runner):
    BeamDagRunner().run(
        iris_pipeline_sklearn._create_pipeline(
            pipeline_name=self._pipeline_name,
            data_root=self._data_root,
            module_file=self._module_file,
            serving_model_dir=self._serving_model_dir,
            pipeline_root=self._pipeline_root,
            metadata_path=self._metadata_path,
            ai_platform_serving_args=self._ai_platform_serving_args,
            direct_num_workers=1))

    self.assertTrue(tf.io.gfile.exists(self._serving_model_dir))
    self.assertTrue(tf.io.gfile.exists(self._metadata_path))
    mock_runner.deploy_model_for_aip_prediction.assert_called_once()
    expected_execution_count = 8  # 8 components
    metadata_config = metadata.sqlite_metadata_connection_config(
        self._metadata_path)
    with metadata.Metadata(metadata_config) as m:
      artifact_count = len(m.store.get_artifacts())
      execution_count = len(m.store.get_executions())
      self.assertGreaterEqual(artifact_count, execution_count)
      self.assertEqual(expected_execution_count, execution_count)

    self.assertPipelineExecution()

    # Runs pipeline the second time.
    BeamDagRunner().run(
        iris_pipeline_sklearn._create_pipeline(
            pipeline_name=self._pipeline_name,
            data_root=self._data_root,
            module_file=self._module_file,
            serving_model_dir=self._serving_model_dir,
            pipeline_root=self._pipeline_root,
            metadata_path=self._metadata_path,
            ai_platform_serving_args=self._ai_platform_serving_args,
            direct_num_workers=1))

    # All executions but Evaluator and Pusher are cached.
    with metadata.Metadata(metadata_config) as m:
      self.assertEqual(artifact_count, len(m.store.get_artifacts()))
      artifact_count = len(m.store.get_artifacts())
      self.assertEqual(expected_execution_count * 2,
                       len(m.store.get_executions()))

    # Runs pipeline the third time.
    BeamDagRunner().run(
        iris_pipeline_sklearn._create_pipeline(
            pipeline_name=self._pipeline_name,
            data_root=self._data_root,
            module_file=self._module_file,
            serving_model_dir=self._serving_model_dir,
            pipeline_root=self._pipeline_root,
            metadata_path=self._metadata_path,
            ai_platform_serving_args=self._ai_platform_serving_args,
            direct_num_workers=1))

    # Asserts cache execution.
    with metadata.Metadata(metadata_config) as m:
      # Artifact count is unchanged.
      self.assertEqual(artifact_count, len(m.store.get_artifacts()))
      self.assertEqual(expected_execution_count * 3,
                       len(m.store.get_executions()))


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
