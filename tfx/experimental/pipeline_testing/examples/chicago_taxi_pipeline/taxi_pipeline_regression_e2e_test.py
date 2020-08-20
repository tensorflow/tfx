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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import filecmp
import os

from typing import Text
import tensorflow as tf

from tfx.examples.chicago_taxi_pipeline import taxi_pipeline_beam
from tfx.experimental.pipeline_testing import pipeline_recorder_utils
from tfx.experimental.pipeline_testing import stub_component_launcher
from tfx.orchestration import metadata
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.orchestration.config import pipeline_config

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

  def testTaxiPipelineBeam(self):
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
        host=None,
        port=None,
        pipeline_name=self._pipeline_name,
        run_id=None)

    # Run pipeline with stub executors.
    taxi_pipeline = taxi_pipeline_beam._create_pipeline(  # pylint:disable=protected-access
        pipeline_name=self._pipeline_name,
        data_root=self._data_root,
        module_file=self._module_file,
        serving_model_dir=self._serving_model_dir,
        pipeline_root=self._pipeline_root,
        metadata_path=self._metadata_path,
        beam_pipeline_args=[])

    model_resolver_id = 'ResolverNode.latest_blessed_model_resolver'
    stubbed_component_ids = [
        component.id
        for component in taxi_pipeline.components
        if component.id != model_resolver_id
    ]

    stub_launcher = stub_component_launcher.get_stub_launcher_class(
        test_data_dir=self._recorded_output_dir,
        stubbed_component_ids=stubbed_component_ids,
        stubbed_component_map={})
    stub_pipeline_config = pipeline_config.PipelineConfig(
        supported_launcher_classes=[
            stub_launcher,
        ])
    BeamDagRunner(config=stub_pipeline_config).run(taxi_pipeline)

    self.assertTrue(tf.io.gfile.exists(self._metadata_path))

    metadata_config = metadata.sqlite_metadata_connection_config(
        self._metadata_path)

    # Verify that recorded files are successfully copied to the output uris.
    with metadata.Metadata(metadata_config) as m:
      artifacts = m.store.get_artifacts()
      artifact_count = len(artifacts)
      executions = m.store.get_executions()
      execution_count = len(executions)
      # artifact count is greater by 2 due to two artifacts produced by both
      # Evaluator(blessing and evaluation) and Trainer(model and model_run)
      self.assertEqual(artifact_count, execution_count + 2)
      self.assertLen(taxi_pipeline.components, execution_count)

      for execution in executions:
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
          self.assertTrue(steps[0].HasField('key'))
          name = steps[0].key
          artifacts = m.store.get_artifacts_by_id([event.artifact_id])
          for idx, artifact in enumerate(artifacts):
            self.assertDirectoryEqual(
                artifact.uri,
                os.path.join(self._recorded_output_dir, component_id, name,
                             str(idx)))


if __name__ == '__main__':
  tf.test.main()
