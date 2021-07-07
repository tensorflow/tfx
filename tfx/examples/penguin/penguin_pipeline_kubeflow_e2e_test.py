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
"""Tests for tfx.examples.penguin.penguin_pipeline_kubeflow_gcp."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from unittest import mock

import tensorflow as tf
from tfx.dsl.io import fileio
from tfx.examples.penguin import penguin_pipeline_kubeflow
from tfx.orchestration import test_utils
from tfx.orchestration.kubeflow import test_utils as kubeflow_test_utils
from tfx.v1 import orchestration


class PenguinPipelineKubeflowTest(kubeflow_test_utils.BaseKubeflowTest):

  @classmethod
  def setUpClass(cls):
    # Initializes the port-forward process to talk MLMD.
    super().setUpClass()

  @mock.patch('tfx.components.util.udf_utils.UserModuleFilePipDependency.'
              'resolve')
  def testPenguinPipelineConstructionAndDefinitionFileExists(
      self, resolve_mock):
    # Avoid actually performing user module packaging because a placeholder
    # GCS bucket is used.
    resolve_mock.side_effect = lambda pipeline_root: None

    kubeflow_pipeline = penguin_pipeline_kubeflow.create_pipeline(
        pipeline_name=penguin_pipeline_kubeflow._pipeline_name,
        pipeline_root=penguin_pipeline_kubeflow._pipeline_root,
        data_root=penguin_pipeline_kubeflow._data_root,
        module_file=penguin_pipeline_kubeflow._module_file,
        enable_tuning=False,
        ai_platform_training_args=penguin_pipeline_kubeflow
        ._ai_platform_training_args,
        ai_platform_serving_args=penguin_pipeline_kubeflow
        ._ai_platform_serving_args,
        beam_pipeline_args=penguin_pipeline_kubeflow
        ._beam_pipeline_args_by_runner['DirectRunner'],
        use_aip_component=False,
        serving_model_dir=penguin_pipeline_kubeflow._serving_model_dir)
    self.assertEqual(9, len(kubeflow_pipeline.components))

    orchestration.experimental.KubeflowDagRunner().run(kubeflow_pipeline)
    file_path = os.path.join(self.tmp_dir, 'penguin_kubeflow.tar.gz')
    self.assertTrue(fileio.exists(file_path))

  def testEndToEndPipelineRun(self):
    """End-to-end test for pipeline with RuntimeParameter."""
    pipeline_name = 'kubeflow-e2e-test-parameter-{}'.format(
        test_utils.random_id())
    kubeflow_pipeline = penguin_pipeline_kubeflow.create_pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=self._pipeline_root(pipeline_name),
        data_root=self._penguin_data_root,
        module_file=self._penguin_dependency_file,
        enable_tuning=False,
        ai_platform_training_args=penguin_pipeline_kubeflow
        ._ai_platform_training_args,
        ai_platform_serving_args=penguin_pipeline_kubeflow
        ._ai_platform_serving_args,
        beam_pipeline_args=penguin_pipeline_kubeflow
        ._beam_pipeline_args_by_runner['DirectRunner'],
        use_aip_component=False,
        serving_model_dir=self._serving_model_dir)

    # TODO(b/190231146): test run parameters after relevant support available.
    parameters = {
        'train-steps': 10,
        'eval-steps': 5,
    }
    self._compile_and_run_pipeline(
        pipeline=kubeflow_pipeline, parameters=parameters)
    self.assertTrue(fileio.exists(self._serving_model_dir))


if __name__ == '__main__':
  tf.test.main()
