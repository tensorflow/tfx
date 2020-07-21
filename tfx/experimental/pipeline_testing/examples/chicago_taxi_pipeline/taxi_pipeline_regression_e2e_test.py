# Lint as: python2, python3
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
"""E2E Tests for taxi pipeline beam with stub executors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from tfx.examples.chicago_taxi_pipeline import taxi_pipeline_beam
from tfx.experimental.pipeline_testing import pipeline_recorder_utils
from tfx.experimental.pipeline_testing import stub_component_launcher
from tfx.orchestration import metadata
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.orchestration.config import pipeline_config

class TaxiPipelineRegressionEndToEndTest(tf.test.TestCase):

  def setUp(self):
    super(TaxiPipelineRegressionEndToEndTest, self).setUp()
    self._test_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    self._pipeline_name = 'beam_test'
    # This example assumes that the taxi data and taxi utility function are stored in
    # ~/tfx/experimental/pipeline_testing/examples/chicago_taxi_pipeline.
    # Feel free to customize this as needed.
    taxi_root = os.path.dirname(__file__)
    self._data_root = os.path.join(taxi_root, 'data', 'simple')
    self._module_file = os.path.join(taxi_root, 'taxi_utils.py')
    self._serving_model_dir = os.path.join(self._test_dir, 'serving_model')
    self._pipeline_root = os.path.join(self._test_dir, 'tfx', 'pipelines',
                                       self._pipeline_name)
    # Metadata path for recording successful pipeline run.
    self._metadata_db_uri = os.path.join(self._test_dir, 'tfx', 'record',
                                         'metadata.db')
    # Metadata path for stub pipeline runs.
    self._metadata_path = os.path.join(self._test_dir, 'tfx', 'metadata',
                                       self._pipeline_name, 'metadata.db')
    self._output_dir = os.path.join(self._test_dir, 'testdata')

  def testTaxiPipelineBeam(self):
    # Runs the pipeline and record to self._output_dir
    record_taxi_pipeline = taxi_pipeline_beam._create_pipeline(  # pylint:disable=protected-access, unexpected-keyword-arg
        pipeline_name=self._pipeline_name,
        data_root=self._data_root,
        module_file=self._module_file,
        serving_model_dir=self._serving_model_dir,
        pipeline_root=self._pipeline_root,
        metadata_path=self._metadata_db_uri,
        beam_pipeline_args=[])
    BeamDagRunner().run(record_taxi_pipeline)
    pipeline_recorder_utils.record_pipeline(
        output_dir=self._output_dir,
        metadata_db_uri=self._metadata_db_uri,
        host=None,
        port=None,
        pipeline_name=self._pipeline_name,
        run_id=None)

    # Run pipeline with stub executors first time.
    taxi_pipeline = taxi_pipeline_beam._create_pipeline(  # pylint:disable=protected-access, unexpected-keyword-arg
        pipeline_name=self._pipeline_name,
        data_root=self._data_root,
        module_file=self._module_file,
        serving_model_dir=self._serving_model_dir,
        pipeline_root=self._pipeline_root,
        metadata_path=self._metadata_path,
        beam_pipeline_args=[])

    model_resolver_id = 'ResolverNode.latest_blessed_model_resolver'
    self._component_ids = [component.id
                           for component in taxi_pipeline.components
                           if component.id != model_resolver_id]

    my_launcher = stub_component_launcher.get_stub_launcher_class(
        test_data_dir=self._output_dir,
        stubbed_component_ids=self._component_ids,
        stubbed_component_map={})
    my_pipeline_config = pipeline_config.PipelineConfig(
        supported_launcher_classes=[
            my_launcher,
        ])
    BeamDagRunner(config=my_pipeline_config).run(taxi_pipeline)

    self.assertTrue(tf.io.gfile.exists(self._metadata_path))
    metadata_config = metadata.sqlite_metadata_connection_config(
        self._metadata_path)
    with metadata.Metadata(metadata_config) as m:
      artifact_count = len(m.store.get_artifacts())
      execution_count = len(m.store.get_executions())
      self.assertGreaterEqual(artifact_count, execution_count)
      self.assertLen(taxi_pipeline.components, execution_count)

    # Run pipeline with stub executors second time.
    BeamDagRunner(config=my_pipeline_config).run(taxi_pipeline)

    # All executions but Evaluator and Pusher are cached.
    # Note that Resolver will always execute.
    with metadata.Metadata(metadata_config) as m:
      self.assertLen(m.store.get_artifacts(), artifact_count)
      artifact_count = len(m.store.get_artifacts())
      self.assertLen(m.store.get_executions(),
                     len(taxi_pipeline.components) * 2)

    # Runs pipeline with stub executors for the third time.
    BeamDagRunner(config=my_pipeline_config).run(taxi_pipeline)
    # Asserts cache execution.
    with metadata.Metadata(metadata_config) as m:
      # Artifact count is unchanged.
      self.assertLen(m.store.get_artifacts(), artifact_count)
      self.assertLen(m.store.get_executions(),
                     len(taxi_pipeline.components) * 3)

if __name__ == '__main__':
  tf.test.main()
