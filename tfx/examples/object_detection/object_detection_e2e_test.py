# Copyright 2022 Google LLC. All Rights Reserved.
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
"""Tests for tfx.examples.object_detection.object_detection_e2e."""

import os
import tensorflow as tf

from tfx.examples.object_detection import object_detection_pipeline
from tfx.orchestration.local.local_dag_runner import LocalDagRunner


class ObjectDetectionE2ETest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._test_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    self._pipeline_name = 'tf_object_detection'
    self._data_root = os.path.join(os.path.dirname(__file__), 'data')
    self._tfx_root = os.path.join(self._test_dir, 'tfx')
    self._metadata_path = os.path.join(self._tfx_root, 'metadata',
                                       self._pipeline_name, 'metadata.db')
    self._module_file = os.path.join(
        os.path.dirname(__file__), 'object_detection_utils.py')

    print('TFX ROOT: ', self._tfx_root)

  def testPipeline(self):
    LocalDagRunner().run(
        object_detection_pipeline._create_pipeline(
            pipeline_name=self._pipeline_name,
            pipeline_root=self._tfx_root,
            data_root=self._data_root,
            metadata_path=self._metadata_path,
            module_file=self._module_file,
            beam_pipeline_args=['--direct_num_workers=1']))
    self.assertTrue(tf.io.gfile.exists(self._metadata_path))

if __name__ == '__main__':
  tf.test.main()
