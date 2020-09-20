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
"""Tests for tfx.orchestration.experimental.kubernetes.examples.taxi_pipeline_kubernetes."""

import os
import tensorflow as tf
from tfx.orchestration.experimental.kubernetes.examples import taxi_pipeline_kubernetes


class TaxiPipelineKubernetesTest(tf.test.TestCase):

  def setUp(self):
    super(TaxiPipelineKubernetesTest, self).setUp()
    self._test_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

  def testTaxiPipelineCheckDagConstruction(self):
    logical_pipeline = taxi_pipeline_kubernetes.create_pipeline(
        pipeline_name='Test',
        pipeline_root=self._test_dir,
        data_root=self._test_dir,
        module_file=self._test_dir,
        serving_model_dir=self._test_dir,
        beam_pipeline_args=[])
    self.assertEqual(9, len(logical_pipeline.components))


if __name__ == '__main__':
  tf.test.main()
