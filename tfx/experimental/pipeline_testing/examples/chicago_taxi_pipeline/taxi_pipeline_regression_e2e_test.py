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
"""E2E Tests for taxi pipeline beam with stub executors."""

import os
from typing import Text

from absl import logging
import tensorflow as tf
from tfx.dsl.compiler import compiler
from tfx.dsl.io import fileio
from tfx.examples.chicago_taxi_pipeline import taxi_pipeline_beam
from tfx.experimental.pipeline_testing import executor_verifier_utils
from tfx.experimental.pipeline_testing import pipeline_mock
from tfx.experimental.pipeline_testing import pipeline_recorder_utils
from tfx.orchestration import metadata
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

from ml_metadata.proto import metadata_store_pb2


class TaxiPipelineRegressionEndToEndTest(tf.test.TestCase):

  def setUp(self):
    super(TaxiPipelineRegressionEndToEndTest, self).setUp()
    self._test_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    self._pipeline_name = 'beam_stub_test'
    # This example assumes that the taxi data and taxi utility function are
    # stored in tfx/examples/chicago_taxi_pipeline. Feel free to customize this
    # as needed.
    taxi_root = os.path.dirname(taxi_pipeline_beam.__file__)
    self._data_root = os.path.join(taxi_root, 'data', 'simple')
    self._module_file = os.path.join(taxi_root, 'taxi_utils.py')
    self._serving_model_dir = os.path.join(self._test_dir, 'serving_model')
    self._pipeline_root = os.path.join(self._test_dir, 'tfx', 'pipelines',
                                       self._pipeline_name)
    # Metadata path for recording successful pipeline run.
    self._recorded_mlmd_path = os.path.join(self._test_dir, 'tfx', 'record',
                                            'metadata.db')
    # Metadata path for stub pipeline runs.
    self._metadata_path = os.path.join(self._test_dir, 'tfx', 'metadata',
                                       self._pipeline_name, 'metadata.db')
    self._recorded_output_dir = os.path.join(self._test_dir, 'testdata')

    # Runs the pipeline and record to self._recorded_output_dir
    record_taxi_pipeline = taxi_pipeline_beam._create_pipeline(  # pylint:disable=protected-access
        pipeline_name=self._pipeline_name,
        data_root=self._data_root,
        module_file=self._module_file,
        serving_model_dir=self._serving_model_dir,
        pipeline_root=self._pipeline_root,
        metadata_path=self._recorded_mlmd_path,
        beam_pipeline_args=[])

    BeamDagRunner().run(record_taxi_pipeline)

    pipeline_recorder_utils.record_pipeline(
        output_dir=self._recorded_output_dir,
        metadata_db_uri=self._recorded_mlmd_path,
        pipeline_name=self._pipeline_name)

    self.taxi_pipeline = taxi_pipeline_beam._create_pipeline(  # pylint:disable=protected-access
        pipeline_name=self._pipeline_name,
        data_root=self._data_root,
        module_file=self._module_file,
        serving_model_dir=self._serving_model_dir,
        pipeline_root=self._pipeline_root,
        metadata_path=self._metadata_path,
        beam_pipeline_args=[])

  def assertDirectoryEqual(self, dir1: Text, dir2: Text):
    self.assertTrue(executor_verifier_utils.compare_dirs(dir1, dir2))

  def _verify_file_path(self, output_uri: Text, artifact_uri: Text):
    self.assertTrue(
        executor_verifier_utils.verify_file_dir(output_uri, artifact_uri))

  def _veryify_root_dir(self, output_uri: str, unused_artifact_uri: str):
    self.assertTrue(fileio.exists(output_uri))

  def _verify_evaluation(self, output_uri: Text, expected_uri: Text):
    self.assertTrue(
        executor_verifier_utils.compare_eval_results(output_uri, expected_uri,
                                                     .5))

  def _verify_schema(self, output_uri: Text, expected_uri: Text):
    self.assertTrue(
        executor_verifier_utils.compare_file_sizes(output_uri, expected_uri,
                                                   .5))

  def _verify_examples(self, output_uri: Text, expected_uri: Text):
    self.assertTrue(
        executor_verifier_utils.compare_file_sizes(output_uri, expected_uri,
                                                   .5))

  def _verify_model(self, output_uri: Text, expected_uri: Text):
    self.assertTrue(
        executor_verifier_utils.compare_model_file_sizes(
            output_uri, expected_uri, .5))

  def _verify_anomalies(self, output_uri: Text, expected_uri: Text):
    self.assertTrue(
        executor_verifier_utils.compare_anomalies(output_uri, expected_uri))

  def testStubbedTaxiPipelineBeam(self):
    pipeline_ir = compiler.Compiler().compile(self.taxi_pipeline)

    logging.info('Replacing with test_data_dir:%s', self._recorded_output_dir)
    pipeline_mock.replace_executor_with_stub(pipeline_ir,
                                             self._recorded_output_dir, [])

    BeamDagRunner().run(pipeline_ir)

    self.assertTrue(fileio.exists(self._metadata_path))

    metadata_config = metadata.sqlite_metadata_connection_config(
        self._metadata_path)

    # Verify that recorded files are successfully copied to the output uris.
    with metadata.Metadata(metadata_config) as m:
      artifacts = m.store.get_artifacts()
      artifact_count = len(artifacts)
      executions = m.store.get_executions()
      execution_count = len(executions)
      # Artifact count is greater by 3 due to extra artifacts produced by
      # Evaluator(blessing and evaluation), Trainer(model and model_run) and
      # Transform(example, graph, cache) minus Resolver which doesn't generate
      # new artifact.
      self.assertEqual(artifact_count, execution_count + 3)
      self.assertLen(self.taxi_pipeline.components, execution_count)

      for execution in executions:
        component_id = pipeline_recorder_utils.get_component_id_from_execution(
            m, execution)
        if component_id.startswith('ResolverNode'):
          continue
        eid = [execution.id]
        events = m.store.get_events_by_execution_ids(eid)
        output_events = [
            x for x in events if x.type == metadata_store_pb2.Event.OUTPUT
        ]
        for event in output_events:
          steps = event.path.steps
          self.assertTrue(steps[0].HasField('key'))
          name = steps[0].key
          artifacts = m.store.get_artifacts_by_id([event.artifact_id])
          for idx, artifact in enumerate(artifacts):
            self.assertDirectoryEqual(
                artifact.uri,
                os.path.join(self._recorded_output_dir, component_id, name,
                             str(idx)))

    # Calls verifier for pipeline output artifacts, excluding the resolver node.
    BeamDagRunner().run(self.taxi_pipeline)
    pipeline_outputs = executor_verifier_utils.get_pipeline_outputs(
        self.taxi_pipeline.metadata_connection_config, self._pipeline_name)

    verifier_map = {
        'model': self._verify_model,
        'model_run': self._verify_model,
        'examples': self._verify_examples,
        'schema': self._verify_schema,
        'anomalies': self._verify_anomalies,
        'evaluation': self._verify_evaluation,
        # A subdirectory of updated_analyzer_cache has changing name.
        'updated_analyzer_cache': self._veryify_root_dir,
    }

    # List of components to verify. ResolverNode is ignored because it
    # doesn't have an executor.
    verify_component_ids = [
        component.id
        for component in self.taxi_pipeline.components
        if not component.id.startswith('ResolverNode')
    ]

    for component_id in verify_component_ids:
      logging.info('Verifying %s', component_id)
      for key, artifact_dict in pipeline_outputs[component_id].items():
        for idx, artifact in artifact_dict.items():
          recorded_uri = os.path.join(self._recorded_output_dir, component_id,
                                      key, str(idx))
          verifier_map.get(key, self._verify_file_path)(artifact.uri,
                                                        recorded_uri)


if __name__ == '__main__':
  tf.test.main()
