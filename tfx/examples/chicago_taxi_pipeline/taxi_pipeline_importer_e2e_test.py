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
"""E2E Tests for tfx.examples.chicago_taxi_pipeline.taxi_pipeline_importer."""
import os

from absl.testing import parameterized
import tensorflow as tf
from tfx.dsl.io import fileio
from tfx.examples.chicago_taxi_pipeline import taxi_pipeline_importer
from tfx.examples.chicago_taxi_pipeline import test_utils
from tfx.orchestration import metadata
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.orchestration.portable.beam_dag_runner import BeamDagRunner as IrBeamDagRunner
from tfx_bsl.version import __version__ as tfx_bsl_version


class TaxiPipelineImporterEndToEndTest(test_utils.TaxiTest,
                                       parameterized.TestCase):

  def setUp(self):
    super(TaxiPipelineImporterEndToEndTest, self).setUp()
    self._test_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    self._pipeline_name = 'beam_test'
    self._data_root = os.path.join(os.path.dirname(__file__), 'data', 'simple')
    self._user_schema_path = os.path.join(
        os.path.dirname(__file__), 'data', 'user_provided_schema')
    self._module_file = os.path.join(os.path.dirname(__file__), 'taxi_utils.py')
    self._serving_model_dir = os.path.join(self._test_dir, 'serving_model')
    self._pipeline_root = os.path.join(self._test_dir, 'tfx', 'pipelines',
                                       self._pipeline_name)
    self._metadata_path = os.path.join(self._test_dir, 'tfx', 'metadata',
                                       self._pipeline_name, 'metadata.db')

  @parameterized.parameters((BeamDagRunner), (IrBeamDagRunner))
  def testTaxiPipelineWithImporter(self, runner_class):
    runner_class().run(
        taxi_pipeline_importer._create_pipeline(
            pipeline_name=self._pipeline_name,
            data_root=self._data_root,
            user_schema_path=self._user_schema_path,
            module_file=self._module_file,
            serving_model_dir=self._serving_model_dir,
            pipeline_root=self._pipeline_root,
            metadata_path=self._metadata_path,
            beam_pipeline_args=[]))

    self.assertTrue(fileio.exists(self._serving_model_dir))
    self.assertTrue(fileio.exists(self._metadata_path))
    metadata_config = metadata.sqlite_metadata_connection_config(
        self._metadata_path)
    with metadata.Metadata(metadata_config) as m:
      artifact_count = len(m.store.get_artifacts())
      execution_count = len(m.store.get_executions())
      self.assertGreaterEqual(artifact_count, execution_count)
      self.assertEqual(10, execution_count)

    self.assertComponentsExecuted(self._pipeline_root, [
        'CsvExampleGen', 'Evaluator', 'ExampleValidator', 'Pusher', 'SchemaGen',
        'StatisticsGen', 'Trainer', 'Transform'
    ])

    # Runs the pipeline again.
    runner_class().run(
        taxi_pipeline_importer._create_pipeline(
            pipeline_name=self._pipeline_name,
            data_root=self._data_root,
            user_schema_path=self._user_schema_path,
            module_file=self._module_file,
            serving_model_dir=self._serving_model_dir,
            pipeline_root=self._pipeline_root,
            metadata_path=self._metadata_path,
            beam_pipeline_args=[]))

    # All executions but Evaluator and Pusher are cached.
    # Note that Resolver will always execute.
    with metadata.Metadata(metadata_config) as m:
      # Artifact count is increased by 3 caused by Evaluator and Pusher.
      self.assertLen(m.store.get_artifacts(), artifact_count + 3)
      artifact_count = len(m.store.get_artifacts())
      self.assertLen(m.store.get_executions(), 20)

    # Runs the pipeline the third time.
    runner_class().run(
        taxi_pipeline_importer._create_pipeline(
            pipeline_name=self._pipeline_name,
            data_root=self._data_root,
            user_schema_path=self._user_schema_path,
            module_file=self._module_file,
            serving_model_dir=self._serving_model_dir,
            pipeline_root=self._pipeline_root,
            metadata_path=self._metadata_path,
            beam_pipeline_args=[]))

    # Asserts cache execution.
    with metadata.Metadata(metadata_config) as m:
      # Artifact count is unchanged.
      self.assertLen(m.store.get_artifacts(), artifact_count)
      self.assertLen(m.store.get_executions(), 30)


if __name__ == '__main__':
  tf.test.main()
