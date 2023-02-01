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
import unittest

from absl.testing import parameterized
import tensorflow as tf

from tfx.dsl.io import fileio
from tfx.examples.penguin import penguin_pipeline_local_infraval
from tfx.orchestration import metadata
from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from tfx.utils import path_utils

from ml_metadata.proto import metadata_store_pb2

_OUTPUT_EVENT_TYPES = [
    metadata_store_pb2.Event.OUTPUT,
    metadata_store_pb2.Event.DECLARED_OUTPUT,
]


@unittest.skipIf(tf.__version__ < '2',
                 'Uses keras Model only compatible with TF 2.x')
class PenguinPipelineLocalInfravalEndToEndTest(
    tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._test_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    self._pipeline_name = 'penguin_test'
    self._data_root = os.path.join(os.path.dirname(__file__), 'data')
    self._schema_path = os.path.join(
        os.path.dirname(__file__), 'schema', 'user_provided', 'schema.pbtxt')
    self._module_file = os.path.join(
        os.path.dirname(__file__), 'penguin_utils_keras.py')
    self._serving_model_dir = os.path.join(self._test_dir, 'serving_model')
    self._pipeline_root = os.path.join(self._test_dir, 'tfx', 'pipelines',
                                       self._pipeline_name)
    self._metadata_path = os.path.join(self._test_dir, 'tfx', 'metadata',
                                       self._pipeline_name, 'metadata.db')

  def _assertFileExists(self, path):
    self.assertTrue(fileio.exists(path), f'{path} does not exist.')

  def _assertExecutedOnce(self, component: str) -> None:
    """Check the component is executed exactly once."""
    component_path = os.path.join(self._pipeline_root, component)
    self.assertTrue(fileio.exists(component_path))
    execution_path = os.path.join(
        component_path, '.system', 'executor_execution')
    execution = fileio.listdir(execution_path)
    self.assertLen(execution, 1)

  def _get_latest_output_artifact(self, component_name, output_key):
    metadata_config = metadata.sqlite_metadata_connection_config(
        self._metadata_path)
    with metadata.Metadata(metadata_config) as m:
      [exec_type_name] = [
          exec_type.name for exec_type in m.store.get_execution_types()
          if component_name in exec_type.name]
      executions = m.store.get_executions_by_type(exec_type_name)
      events = m.store.get_events_by_execution_ids([e.id for e in executions])
      output_artifact_ids = [event.artifact_id for event in events
                             if event.type in _OUTPUT_EVENT_TYPES]
      output_artifacts = m.store.get_artifacts_by_id(output_artifact_ids)
      self.assertNotEmpty(output_artifacts)
      return max(output_artifacts, key=lambda a: a.create_time_since_epoch)

  def _assertInfraValidatorPassed(self):
    blessing = self._get_latest_output_artifact('InfraValidator', 'blessing')
    self._assertFileExists(os.path.join(blessing.uri, 'INFRA_BLESSED'))

  def _assertPushedModelHasWarmup(self):
    pushed_model = self._get_latest_output_artifact('Pusher', 'pushed_model')
    self._assertFileExists(path_utils.warmup_file_path(pushed_model.uri))

  def _assertPipelineExecution(self):
    self._assertExecutedOnce('CsvExampleGen')
    self._assertExecutedOnce('Evaluator')
    self._assertExecutedOnce('ExampleValidator')
    self._assertExecutedOnce('ImportSchemaGen')
    self._assertExecutedOnce('InfraValidator')
    self._assertExecutedOnce('Pusher')
    self._assertExecutedOnce('StatisticsGen')
    self._assertExecutedOnce('Trainer')
    self._assertExecutedOnce('Transform')

  @parameterized.named_parameters(
      ('_withoutWarmup', False),
      ('_withWarmup', True))
  def testPenguinPipelineLocal(self, make_warmup):
    LocalDagRunner().run(
        penguin_pipeline_local_infraval._create_pipeline(
            pipeline_name=self._pipeline_name,
            data_root=self._data_root,
            module_file=self._module_file,
            accuracy_threshold=0.1,
            serving_model_dir=self._serving_model_dir,
            pipeline_root=self._pipeline_root,
            metadata_path=self._metadata_path,
            user_provided_schema_path=self._schema_path,
            beam_pipeline_args=[],
            make_warmup=make_warmup))

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

    self._assertPipelineExecution()
    self._assertInfraValidatorPassed()

    # Runs pipeline the second time.
    LocalDagRunner().run(
        penguin_pipeline_local_infraval._create_pipeline(
            pipeline_name=self._pipeline_name,
            data_root=self._data_root,
            module_file=self._module_file,
            accuracy_threshold=0.1,
            serving_model_dir=self._serving_model_dir,
            pipeline_root=self._pipeline_root,
            metadata_path=self._metadata_path,
            user_provided_schema_path=self._schema_path,
            beam_pipeline_args=[],
            make_warmup=make_warmup))

    # All executions but Evaluator and Pusher are cached.
    with metadata.Metadata(metadata_config) as m:
      # Artifact count is increased by 3 caused by Evaluator and Pusher.
      self.assertLen(m.store.get_artifacts(), artifact_count + 3)
      artifact_count = len(m.store.get_artifacts())
      self.assertLen(m.store.get_executions(), expected_execution_count * 2)

    # Runs pipeline the third time.
    LocalDagRunner().run(
        penguin_pipeline_local_infraval._create_pipeline(
            pipeline_name=self._pipeline_name,
            data_root=self._data_root,
            module_file=self._module_file,
            accuracy_threshold=0.1,
            serving_model_dir=self._serving_model_dir,
            pipeline_root=self._pipeline_root,
            metadata_path=self._metadata_path,
            user_provided_schema_path=self._schema_path,
            beam_pipeline_args=[],
            make_warmup=make_warmup))

    # Asserts cache execution.
    with metadata.Metadata(metadata_config) as m:
      # Artifact count is unchanged.
      self.assertLen(m.store.get_artifacts(), artifact_count)
      self.assertLen(m.store.get_executions(), expected_execution_count * 3)


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
