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
"""E2E tests for tfx.examples.penguin.penguin_pipeline_kubeflow."""

import os

import tensorflow as tf
from tfx.dsl.io import fileio
from tfx.examples.penguin import penguin_pipeline_kubeflow
from tfx.orchestration.kubeflow import test_utils as kubeflow_test_utils
from tfx.orchestration.kubeflow.v2.e2e_tests import base_test_case
from tfx.utils import io_utils


class PenguinPipelineKubeflowV2Test(base_test_case.BaseKubeflowV2Test):

  def setUp(self):
    super().setUp()
    penguin_examples_dir = os.path.join(self._REPO_BASE, 'tfx', 'examples',
                                        'penguin')
    penguin_test_data_root = os.path.join(penguin_examples_dir, 'data')
    penguin_test_schema_file = os.path.join(penguin_examples_dir, 'schema',
                                            'user_provided', 'schema.pbtxt')
    self._penguin_module_file = os.path.join(penguin_examples_dir,
                                             'penguin_utils_cloud_tuner.py')
    self._penguin_data_root = os.path.join(self._test_data_dir, 'data')
    self._penguin_schema_file = os.path.join(self._test_data_dir,
                                             'schema.pbtxt')
    io_utils.copy_dir(penguin_test_data_root, self._penguin_data_root)
    io_utils.copy_file(
        penguin_test_schema_file, self._penguin_schema_file, overwrite=True)

  def testEndToEndPipelineRun(self):
    """E2E test for pipeline with runtime parameter."""
    pipeline_name = 'kubeflow-v2-e2e-test-{}'.format(self._test_id)
    kubeflow_pipeline = penguin_pipeline_kubeflow.create_pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=self._pipeline_root(pipeline_name),
        data_root=self._penguin_data_root,
        module_file=self._penguin_module_file,
        enable_tuning=False,
        enable_cache=True,
        user_provided_schema_path=self._penguin_schema_file,
        ai_platform_training_args=penguin_pipeline_kubeflow
        ._ai_platform_training_args,
        ai_platform_serving_args=penguin_pipeline_kubeflow
        ._ai_platform_serving_args,
        beam_pipeline_args=penguin_pipeline_kubeflow
        ._beam_pipeline_args_by_runner['DirectRunner'],
        use_cloud_component=False,
        use_aip=False,
        use_vertex=False,
        serving_model_dir=self._serving_model_dir)

    self._run_pipeline(
        pipeline=kubeflow_pipeline,
        parameter_values={
            'train-args': {
                'num_steps': 100
            },
            'eval-args': {
                'num_steps': 50
            }
        })
    self.assertTrue(fileio.exists(self._serving_model_dir))


class PenguinPipelineKubeflowTest(kubeflow_test_utils.BaseKubeflowTest):

  def setUp(self):
    super().setUp()
    penguin_examples_dir = os.path.join(self._REPO_BASE, 'tfx', 'examples',
                                        'penguin')
    penguin_test_data_root = os.path.join(penguin_examples_dir, 'data')
    penguin_test_schema_file = os.path.join(penguin_examples_dir, 'schema',
                                            'user_provided', 'schema.pbtxt')
    self._penguin_module_file = os.path.join(penguin_examples_dir,
                                             'penguin_utils_cloud_tuner.py')
    self._penguin_data_root = os.path.join(self._test_data_dir, 'data')
    self._penguin_schema_file = os.path.join(self._test_data_dir,
                                             'schema.pbtxt')

    io_utils.copy_dir(penguin_test_data_root, self._penguin_data_root)
    io_utils.copy_file(
        penguin_test_schema_file, self._penguin_schema_file, overwrite=True)

  def testEndToEndPipelineRun(self):
    """End-to-end test for pipeline with RuntimeParameter."""
    pipeline_name = 'kubeflow-v1-e2e-test-{}'.format(self._test_id)
    kubeflow_pipeline = penguin_pipeline_kubeflow.create_pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=self._pipeline_root(pipeline_name),
        data_root=self._penguin_data_root,
        module_file=self._penguin_module_file,
        enable_tuning=False,
        enable_cache=True,
        user_provided_schema_path=self._penguin_schema_file,
        ai_platform_training_args=penguin_pipeline_kubeflow
        ._ai_platform_training_args,
        ai_platform_serving_args=penguin_pipeline_kubeflow
        ._ai_platform_serving_args,
        beam_pipeline_args=penguin_pipeline_kubeflow
        ._beam_pipeline_args_by_runner['DirectRunner'],
        use_cloud_component=False,
        use_aip=False,
        use_vertex=False,
        serving_model_dir=self._serving_model_dir)

    parameters = {
        'train-args': '{"num_steps": 100}',
        'eval-args': '{"num_steps": 50}',
    }
    self._compile_and_run_pipeline(
        pipeline=kubeflow_pipeline, parameters=parameters)
    self.assertTrue(fileio.exists(self._serving_model_dir))


if __name__ == '__main__':
  tf.test.main()
