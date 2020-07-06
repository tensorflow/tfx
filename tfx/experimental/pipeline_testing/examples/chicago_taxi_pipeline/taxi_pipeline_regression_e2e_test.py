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
"""E2E Tests for taxi pipeline beam with dummy executors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Text

import tensorflow as tf

from tfx.examples.chicago_taxi_pipeline import taxi_pipeline_beam
from tfx.experimental.pipeline_testing import dummy_component_launcher
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
    taxi_root = os.path.join(os.environ['HOME'],
                             "tfx/tfx/examples/chicago_taxi_pipeline")
    self._data_root = os.path.join(taxi_root, 'data', 'simple')
    self._module_file = os.path.join(taxi_root, 'taxi_utils.py')
    self._serving_model_dir = os.path.join(self._test_dir, 'serving_model')
    self._pipeline_root = os.path.join(self._test_dir, 'tfx', 'pipelines',
                                       self._pipeline_name)
    self._metadata_path = os.path.join(self._test_dir, 'tfx', 'metadata',
                                       self._pipeline_name, 'metadata.db')

  def assertExecutedOnce(self, component: Text) -> None:
    """Check the component is executed exactly once.
    Pipeline root is <test_dir>/tfx/pipelines/beam_test/
    Execution outputs are saved in <component.id>/<key>/<execution.id>
    """
    component_path = os.path.join(self._pipeline_root, component)
    self.assertTrue(tf.io.gfile.exists(component_path))
    outputs = tf.io.gfile.listdir(component_path)
    for output in outputs:
      execution = tf.io.gfile.listdir(os.path.join(component_path, output))
      self.assertLen(execution, 1)

  def testTaxiPipelineBeam(self):
    record_dir = os.path.join(os.environ['HOME'],
                              'tfx/tfx/experimental/pipeline_testing/',
                              'examples/chicago_taxi_pipeline/testdata')

    component_ids = ['CsvExampleGen', \
                    'StatisticsGen', 'SchemaGen', \
                    'ExampleValidator', 'Transform', \
                    'Trainer', 'Evaluator', 'Pusher']

    my_launcher = dummy_component_launcher.create_dummy_launcher_class(
        record_dir,
        component_ids,
        {})
    taxi_pipeline = taxi_pipeline_beam._create_pipeline(  # pylint:disable=protected-access, unexpected-keyword-arg
                pipeline_name=self._pipeline_name,
                data_root=self._data_root,
                module_file=self._module_file,
                serving_model_dir=self._serving_model_dir,
                pipeline_root=self._pipeline_root,
                metadata_path=self._metadata_path,
                direct_num_workers=1)

    BeamDagRunner(config=pipeline_config.PipelineConfig(
        supported_launcher_classes=[
            my_launcher,
        ],
        )).run(taxi_pipeline)

    self.assertTrue(tf.io.gfile.exists(self._metadata_path))
    metadata_config = metadata.sqlite_metadata_connection_config(
        self._metadata_path)
    with metadata.Metadata(metadata_config) as m:
      artifact_count = len(m.store.get_artifacts())
      execution_count = len(m.store.get_executions())
      self.assertGreaterEqual(artifact_count, execution_count)
      self.assertLen(taxi_pipeline.components, execution_count)

    for component_id in component_ids:
      self.assertExecutedOnce(component_id)

    # Runs pipeline the second time.
    BeamDagRunner(config=pipeline_config.PipelineConfig(
        supported_launcher_classes=[
            my_launcher,
        ],
        )).run(
            taxi_pipeline_beam._create_pipeline(  # pylint:disable=protected-access, unexpected-keyword-arg
                pipeline_name=self._pipeline_name,
                data_root=self._data_root,
                module_file=self._module_file,
                serving_model_dir=self._serving_model_dir,
                pipeline_root=self._pipeline_root,
                metadata_path=self._metadata_path,
                direct_num_workers=1))

    # All executions but Evaluator and Pusher are cached.
    # Note that Resolver will always execute.
    with metadata.Metadata(metadata_config) as m:
      # Artifact count is increased by 3 caused by Evaluator and Pusher.
      self.assertLen(m.store.get_artifacts(), artifact_count)
      artifact_count = len(m.store.get_artifacts())
      self.assertLen(m.store.get_executions(),
                     len(taxi_pipeline.components) * 2)

    # Runs pipeline the third time.
    BeamDagRunner(config=pipeline_config.PipelineConfig(
        supported_launcher_classes=[
            my_launcher,
        ],
        )).run(
            taxi_pipeline_beam._create_pipeline(  # pylint:disable=protected-access, unexpected-keyword-arg
                pipeline_name=self._pipeline_name,
                data_root=self._data_root,
                module_file=self._module_file,
                serving_model_dir=self._serving_model_dir,
                pipeline_root=self._pipeline_root,
                metadata_path=self._metadata_path,
                direct_num_workers=1))

    # Asserts cache execution.
    with metadata.Metadata(metadata_config) as m:
      # Artifact count is unchanged.
      self.assertLen(m.store.get_artifacts(), artifact_count)
      self.assertLen(m.store.get_executions(),
                     len(taxi_pipeline.components) * 3)


if __name__ == '__main__':
  tf.test.main()
