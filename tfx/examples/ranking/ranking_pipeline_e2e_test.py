# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Tests for tfx.examples.ranking.ranking_pipeline."""
import os
import unittest

import tensorflow as tf
from tfx.examples.ranking import ranking_pipeline
from tfx.orchestration import metadata
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

try:
  import struct2tensor  # pylint: disable=g-import-not-at-top
except ImportError:
  struct2tensor = None


@unittest.skipIf(struct2tensor is None,
                 'Cannot import required modules. This can happen when'
                 ' struct2tensor is not available.')
class RankingPipelineTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._test_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    self._pipeline_name = 'tf_ranking_test'
    self._data_root = os.path.join(os.path.dirname(__file__),
                                   'testdata', 'input')
    self._tfx_root = os.path.join(self._test_dir, 'tfx')
    self._module_file = os.path.join(os.path.dirname(__file__),
                                     'ranking_utils.py')
    self._serving_model_dir = os.path.join(self._test_dir, 'serving_model')
    self._metadata_path = os.path.join(self._tfx_root, 'metadata',
                                       self._pipeline_name, 'metadata.db')
    print('TFX ROOT: ', self._tfx_root)

  def assertExecutedOnce(self, component) -> None:
    """Check the component is executed exactly once."""
    component_path = os.path.join(self._pipeline_root, component)
    self.assertTrue(tf.io.gfile.exists(component_path))
    outputs = tf.io.gfile.listdir(component_path)
    for output in outputs:
      execution = tf.io.gfile.listdir(os.path.join(component_path, output))
      self.assertEqual(1, len(execution))

  def testPipeline(self):
    BeamDagRunner().run(
        ranking_pipeline._create_pipeline(
            pipeline_name=self._pipeline_name,
            pipeline_root=self._tfx_root,
            data_root=self._data_root,
            module_file=self._module_file,
            serving_model_dir=self._serving_model_dir,
            metadata_path=self._metadata_path,
            beam_pipeline_args=['--direct_num_workers=1']))
    self.assertTrue(tf.io.gfile.exists(self._serving_model_dir))
    self.assertTrue(tf.io.gfile.exists(self._metadata_path))

    metadata_config = metadata.sqlite_metadata_connection_config(
        self._metadata_path)
    with metadata.Metadata(metadata_config) as m:
      artifact_count = len(m.store.get_artifacts())
      execution_count = len(m.store.get_executions())
      self.assertGreaterEqual(artifact_count, execution_count)
      self.assertEqual(9, execution_count)


if __name__ == '__main__':
  tf.test.main()
