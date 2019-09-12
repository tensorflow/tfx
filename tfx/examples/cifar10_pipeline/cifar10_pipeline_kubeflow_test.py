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
"""Tests for tfx.examples.cifar10_pipeline.cifar10_pipeline_kubeflow."""

import os
import tensorflow as tf
from tfx.examples.cifar10_pipeline import cifar10_pipeline_kubeflow
from tfx.orchestration.kubeflow.kubeflow_dag_runner import KubeflowDagRunner


class Cifar10PipelineKubeflowTest(tf.test.TestCase):

    def setUp(self):
        super(Cifar10PipelineKubeflowTest, self).setUp()
        self._tmp_dir = os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR',
                                       self.get_temp_dir())
        self._olddir = os.getcwd()
        os.chdir(self._tmp_dir)

    def tearDown(self):
        super(Cifar10PipelineKubeflowTest, self).tearDown()
        os.chdir(self._olddir)

    def testCifar10PipelineConstructionAndDefinitionFileExists(self):
        logical_pipeline = cifar10_pipeline_kubeflow._create_pipeline(
            pipeline_name=cifar10_pipeline_kubeflow._pipeline_name,
            pipeline_root=cifar10_pipeline_kubeflow._pipeline_root,
            data_root=os.path.join(cifar10_pipeline_kubeflow._pipeline_root, 'data'),
            module_file=cifar10_pipeline_kubeflow._module_file,
            serving_model_dir=cifar10_pipeline_kubeflow._serving_model_dir)
        self.assertEqual(9, len(logical_pipeline.components))

        KubeflowDagRunner().run(logical_pipeline)
        file_path = os.path.join(self._tmp_dir,
                                 'cifar10.tar.gz')
        self.assertTrue(tf.gfile.Exists(file_path))


if __name__ == '__main__':
    tf.test.main()
