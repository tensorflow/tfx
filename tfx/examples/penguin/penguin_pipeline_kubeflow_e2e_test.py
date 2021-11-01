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
import subprocess

import tensorflow as tf
from tfx.dsl.io import fileio
from tfx.examples.penguin import penguin_pipeline_kubeflow
from tfx.orchestration import test_utils
from tfx.orchestration.kubeflow import test_utils as kubeflow_test_utils


class PenguinPipelineKubeflowTest(kubeflow_test_utils.BaseKubeflowTest):

  def setUp(self):
    super().setUp()

    penguin_examples_dir = os.path.join(self._REPO_BASE, 'tfx', 'examples',
                                        'penguin')
    # The location of the penguin test data and schema. The input files are
    # copied to a test-local location for each invocation, and cleaned up at the
    # end of test.
    penguin_test_data_root = os.path.join(penguin_examples_dir, 'data')
    penguin_test_schema_file = os.path.join(penguin_examples_dir, 'schema',
                                            'user_provided', 'schema.pbtxt')

    # The location of the user module for penguin. Will be packaged and copied
    # to under the pipeline root before pipeline execution.
    self._penguin_dependency_file = os.path.join(
        penguin_examples_dir, 'penguin_utils_cloud_tuner.py')

    # TODO(b/174289068): Create test data handling utilities.
    subprocess.run(
        ['gsutil', 'cp', '-r', penguin_test_data_root, self._testdata_root],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ['gsutil', 'cp', penguin_test_schema_file, self._testdata_root],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    self._penguin_data_root = os.path.join(self._testdata_root, 'data')
    self._penguin_schema_file = os.path.join(self._testdata_root,
                                             'schema.pbtxt')

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
        user_provided_schema_path=self._penguin_schema_file,
        ai_platform_training_args=penguin_pipeline_kubeflow
        ._ai_platform_training_args,
        ai_platform_serving_args=penguin_pipeline_kubeflow
        ._ai_platform_serving_args,
        beam_pipeline_args=penguin_pipeline_kubeflow
        ._beam_pipeline_args_by_runner['DirectRunner'],
        use_aip_component=False,
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
