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
"""E2E Tests for IMDB Sentiment Analysis example with stub executors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import filecmp

import tensorflow as tf
from typing import Text

from tfx.examples.imdb import imdb_pipeline_native_keras
from tfx.experimental.pipeline_testing import pipeline_recorder_utils
from tfx.experimental.pipeline_testing import stub_component_launcher
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.orchestration.config import pipeline_config
from tfx.orchestration import metadata
from ml_metadata.proto import metadata_store_pb2

class ImdbStubPipelineRegressionEndToEndTest(tf.test.TestCase):

  def setUp(self):
    super(ImdbPipelineRegressionEndToEndTest, self).setUp()
    self._test_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    self._pipeline_name = 'imdb_stub_test'
    # This example assumes that the imdb data and imdb utility function are stored in
    # tfx/examples/imdb. Feel free to customize this as needed.
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

  def assertDirectoryEqual(self, dir1: Text, dir2: Text):
    """Recursively comparing contents of two directories."""

    dir_cmp = filecmp.dircmp(dir1, dir2)
    self.assertEmpty(dir_cmp.left_only)
    self.assertEmpty(dir_cmp.right_only)
    self.assertEmpty(dir_cmp.funny_files)

    _, mismatch, errors = filecmp.cmpfiles(
        dir1, dir2, dir_cmp.common_files, shallow=False)
    self.assertEmpty(mismatch)
    self.assertEmpty(errors)

    for common_dir in dir_cmp.common_dirs:
      new_dir1 = os.path.join(dir1, common_dir)
      new_dir2 = os.path.join(dir2, common_dir)
      self.assertDirectoryEqual(new_dir1, new_dir2)

  def testImdbPipelineBeam(self):
    # Runs the pipeline and record to self._recorded_output_dir
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
        host=None,
        port=None,
        pipeline_name=self._pipeline_name,
        run_id=None)

    # Run pipeline with stub executors.
    imdb_pipeline = imdb_pipeline_native_keras._create_pipeline(  # pylint:disable=protected-access
        pipeline_name=self._pipeline_name,
        data_root=self._data_root,
        module_file=self._module_file,
        serving_model_dir=self._serving_model_dir,
        pipeline_root=self._pipeline_root,
        metadata_path=self._metadata_path,
        beam_pipeline_args=[])

    model_resolver_id = 'ResolverNode.latest_blessed_model_resolver'
    stubbed_component_ids = [component.id
                             for component in imdb_pipeline.components
                             if component.id != model_resolver_id]

    stub_launcher = stub_component_launcher.get_stub_launcher_class(
        test_data_dir=self._recorded_output_dir,
        stubbed_component_ids=stubbed_component_ids,
        stubbed_component_map={})
    stub_pipeline_config = pipeline_config.PipelineConfig(
        supported_launcher_classes=[
            stub_launcher,
        ])
    BeamDagRunner(config=stub_pipeline_config).run(imdb_pipeline)

    self.assertTrue(tf.io.gfile.exists(self._metadata_path))

    metadata_config = metadata.sqlite_metadata_connection_config(
        self._metadata_path)

    # Verify that recorded files are successfully copied to the output uris.
    with metadata.Metadata(metadata_config) as m:
      for execution in m.store.get_executions():
        component_id = execution.properties[
            metadata._EXECUTION_TYPE_KEY_COMPONENT_ID].string_value  # pylint: disable=protected-access
        if component_id == 'ResolverNode.latest_blessed_model_resolver':
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
          artifacts = m.store.get_artifacts_by_id(
              [event.artifact_id])
          for artifact in artifacts:
            self.assertDirectoryEqual(artifact.uri, os.path.join(
                self._recorded_output_dir,
                component_id,
                name))
if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
