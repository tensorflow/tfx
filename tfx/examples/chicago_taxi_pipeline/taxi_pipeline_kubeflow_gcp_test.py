# Lint as: python2, python3
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
"""Tests for tfx.examples.chicago_taxi_pipeline.taxi_pipeline_kubeflow_gcp."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tfx.examples.chicago_taxi_pipeline import taxi_pipeline_kubeflow_gcp
from tfx.orchestration.kubeflow.kubeflow_dag_runner import KubeflowDagRunner


class TaxiPipelineKubeflowTest(tf.test.TestCase):

  def setUp(self):
    super(TaxiPipelineKubeflowTest, self).setUp()
    self._tmp_dir = os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR',
                                   self.get_temp_dir())
    self._olddir = os.getcwd()
    os.chdir(self._tmp_dir)

  def tearDown(self):
    super(TaxiPipelineKubeflowTest, self).tearDown()
    os.chdir(self._olddir)

  def testTaxiPipelineConstructionAndDefinitionFileExists(self):
    logical_pipeline = taxi_pipeline_kubeflow_gcp.create_pipeline(
        pipeline_name=taxi_pipeline_kubeflow_gcp._pipeline_name,
        pipeline_root=taxi_pipeline_kubeflow_gcp._pipeline_root,
        module_file=taxi_pipeline_kubeflow_gcp._module_file,
        ai_platform_training_args=taxi_pipeline_kubeflow_gcp
        ._ai_platform_training_args,
        ai_platform_serving_args=taxi_pipeline_kubeflow_gcp
        ._ai_platform_serving_args)
    self.assertEqual(8, len(logical_pipeline.components))

    KubeflowDagRunner().run(logical_pipeline)
    file_path = os.path.join(self._tmp_dir,
                             'chicago_taxi_pipeline_kubeflow_gcp.tar.gz')
    self.assertTrue(tf.io.gfile.exists(file_path))


if __name__ == '__main__':
  tf.test.main()
