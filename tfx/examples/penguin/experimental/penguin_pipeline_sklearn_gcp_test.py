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
"""Tests for tfx.examples.experimental.penguin_pipeline_sklearn_gcp."""

import os
from unittest import mock

import tensorflow as tf
from tfx.dsl.io import fileio
from tfx.examples.penguin.experimental import penguin_pipeline_sklearn_gcp
from tfx.orchestration.kubeflow.kubeflow_dag_runner import KubeflowDagRunner
from tfx.utils import test_case_utils


class PenguinPipelineSklearnGcpTest(test_case_utils.TfxTest):

  def setUp(self):
    super(PenguinPipelineSklearnGcpTest, self).setUp()
    self.enter_context(test_case_utils.change_working_dir(self.tmp_dir))

    self._experimental_root = os.path.dirname(__file__)
    self._penguin_root = os.path.dirname(self._experimental_root)

    self._pipeline_name = 'sklearn_test'
    self._data_root = os.path.join(self._penguin_root, 'data')
    self._trainer_module_file = os.path.join(
        self._experimental_root, 'penguin_utils_sklearn.py')
    self._evaluator_module_file = os.path.join(
        self._experimental_root, 'sklearn_predict_extractor.py')
    self._pipeline_root = os.path.join(self.tmp_dir, 'tfx', 'pipelines',
                                       self._pipeline_name)
    self._ai_platform_training_args = {
        'project': 'project_id',
        'region': 'us-central1',
    }
    self._ai_platform_serving_args = {
        'model_name': 'model_name',
        'project_id': 'project_id',
        'regions': ['us-central1'],
    }

  @mock.patch('tfx.components.util.udf_utils.UserModuleFilePipDependency.'
              'resolve')
  def testPipelineConstruction(self, resolve_mock):
    # Avoid actually performing user module packaging because relative path is
    # not valid with respect to temporary directory.
    resolve_mock.side_effect = lambda pipeline_root: None

    logical_pipeline = penguin_pipeline_sklearn_gcp._create_pipeline(
        pipeline_name=self._pipeline_name,
        pipeline_root=self._pipeline_root,
        data_root=self._data_root,
        trainer_module_file=self._trainer_module_file,
        evaluator_module_file=self._evaluator_module_file,
        ai_platform_training_args=self._ai_platform_training_args,
        ai_platform_serving_args=self._ai_platform_serving_args,
        beam_pipeline_args=[])
    self.assertEqual(8, len(logical_pipeline.components))

    KubeflowDagRunner().run(logical_pipeline)
    file_path = os.path.join(self.tmp_dir, 'sklearn_test.tar.gz')
    self.assertTrue(fileio.exists(file_path))


if __name__ == '__main__':
  tf.test.main()
