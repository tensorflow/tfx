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
"""E2E Tests for IMDB Sentiment Analysis example with stub executors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Text

from absl import logging
import tensorflow as tf

from tfx.dsl.compiler import compiler
from tfx.dsl.io import fileio
from tfx.examples.imdb import imdb_pipeline_native_keras
from tfx.experimental.pipeline_testing import executor_verifier_utils
from tfx.experimental.pipeline_testing import pipeline_mock
from tfx.experimental.pipeline_testing import pipeline_recorder_utils
from tfx.orchestration import metadata
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from ml_metadata.proto import metadata_store_pb2


class ImdbStubPipelineRegressionEndToEndTest(tf.test.TestCase):

  def setUp(self):
    super(ImdbStubPipelineRegressionEndToEndTest, self).setUp()
    self._test_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    self._pipeline_name = 'imdb_stub_test'
    # This example assumes that the imdb data and imdb utility function are
    # stored in tfx/examples/imdb. Feel free to customize this as needed.
    imdb_root = os.path.dirname(imdb_pipeline_native_keras.__file__)
    self._data_root = os.path.join(imdb_root, 'data')
    self._module_file = os.path.join(imdb_root, 'imdb_utils_native_keras.py')
    self._serving_model_dir = os.path.join(self._test_dir, 'serving_model')
    self._pipeline_root = os.path.join(self._test_dir, 'pipelines',
                                       self._pipeline_name)
    # Metadata path for recording successful pipeline run.
    self._recorded_mlmd_path = os.path.join(self._test_dir, 'record',
                                            'metadata.db')
    # Metadata path for stub pipeline
    self._metadata_path = os.path.join(self._test_dir, 'metadata',
                                       self._pipeline_name, 'metadata.db')
    self._recorded_output_dir = os.path.join(self._test_dir, 'testdata')

    record_imdb_pipeline = imdb_pipeline_native_keras._create_pipeline(  # pylint:disable=protected-access
        pipeline_name=self._pipeline_name,
        data_root=self._data_root,
        module_file=self._module_file,
        serving_model_dir=self._serving_model_dir,
        pipeline_root=self._pipeline_root,
        metadata_path=self._recorded_mlmd_path,
        beam_pipeline_args=[])

    BeamDagRunner().run(record_imdb_pipeline)

    pipeline_recorder_utils.record_pipeline(
        output_dir=self._recorded_output_dir,
        metadata_db_uri=self._recorded_mlmd_path,
        pipeline_name=self._pipeline_name)

    # Run pipeline with stub executors.
    self.imdb_pipeline = imdb_pipeline_native_keras._create_pipeline(  # pylint:disable=protected-access
        pipeline_name=self._pipeline_name,
        data_root=self._data_root,
        module_file=self._module_file,
        serving_model_dir=self._serving_model_dir,
        pipeline_root=self._pipeline_root,
        metadata_path=self._metadata_path,
        beam_pipeline_args=[])

  def _verify_file_path(self, output_uri: Text, artifact_uri: Text):
    self.assertTrue(
        executor_verifier_utils.verify_file_dir(output_uri, artifact_uri))

  def _veryify_root_dir(self, output_uri: str, unused_artifact_uri: str):
    self.assertTrue(fileio.exists(output_uri))

  def _verify_evaluation(self, output_uri: Text, expected_uri: Text):
    self.assertTrue(
        executor_verifier_utils.compare_eval_results(output_uri, expected_uri,
                                                     1.0, ['accuracy']))

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

  def assertDirectoryEqual(self, dir1: Text, dir2: Text):
    self.assertTrue(executor_verifier_utils.compare_dirs(dir1, dir2))

  def testStubbedImdbPipelineBeam(self):
    pipeline_ir = compiler.Compiler().compile(self.imdb_pipeline)

    pipeline_mock.replace_executor_with_stub(pipeline_ir,
                                             self._recorded_output_dir, [])

    BeamDagRunner().run(pipeline_ir)

    self.assertTrue(fileio.exists(self._metadata_path))

    metadata_config = metadata.sqlite_metadata_connection_config(
        self._metadata_path)

    # Verify that recorded files are successfully copied to the output uris.
    with metadata.Metadata(metadata_config) as m:
      for execution in m.store.get_executions():
        component_id = pipeline_recorder_utils.get_component_id_from_execution(
            m, execution)
        if component_id.startswith('Resolver'):
          continue
        eid = [execution.id]
        events = m.store.get_events_by_execution_ids(eid)
        output_events = [
            x for x in events if x.type == metadata_store_pb2.Event.OUTPUT
        ]
        for event in output_events:
          steps = event.path.steps
          assert steps[0].HasField('key')
          name = steps[0].key
          artifacts = m.store.get_artifacts_by_id([event.artifact_id])
          for idx, artifact in enumerate(artifacts):
            self.assertDirectoryEqual(
                artifact.uri,
                os.path.join(self._recorded_output_dir, component_id, name,
                             str(idx)))

    # Calls verifier for pipeline output artifacts, excluding the resolver node.
    BeamDagRunner().run(self.imdb_pipeline)
    pipeline_outputs = executor_verifier_utils.get_pipeline_outputs(
        self.imdb_pipeline.metadata_connection_config,
        self._pipeline_name)

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

    # List of components to verify. Resolver is ignored because it
    # doesn't have an executor.
    verify_component_ids = [
        component.id
        for component in self.imdb_pipeline.components
        if not component.id.startswith('Resolver')
    ]

    for component_id in verify_component_ids:
      for key, artifact_dict in pipeline_outputs[component_id].items():
        for idx, artifact in artifact_dict.items():
          logging.info('Verifying %s', component_id)
          recorded_uri = os.path.join(self._recorded_output_dir, component_id,
                                      key, str(idx))
          verifier_map.get(key, self._verify_file_path)(artifact.uri,
                                                        recorded_uri)


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
