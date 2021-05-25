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
"""Tests for tfx.tools.cli.handler.kubeflow_v2_dag_runner_patcher."""

import os
from unittest import mock

import tensorflow as tf
from tfx.orchestration import pipeline as tfx_pipeline
from tfx.orchestration.kubeflow.v2 import kubeflow_v2_dag_runner
from tfx.tools.cli.handler import kubeflow_v2_dag_runner_patcher
from tfx.utils import test_case_utils


class KubeflowV2DagRunnerPatcherTest(test_case_utils.TfxTest):

  def setUp(self):
    super().setUp()
    self.enter_context(test_case_utils.change_working_dir(self.tmp_dir))

  def testPatcherBuildImageFn(self):
    given_image_name = 'foo/bar'
    built_image_name = 'foo/bar@sha256:1234567890'

    mock_build_image_fn = mock.MagicMock(return_value=built_image_name)
    patcher = kubeflow_v2_dag_runner_patcher.KubeflowV2DagRunnerPatcher(
        call_real_run=True, build_image_fn=mock_build_image_fn)
    runner_config = kubeflow_v2_dag_runner.KubeflowV2DagRunnerConfig(
        default_image=given_image_name)
    runner = kubeflow_v2_dag_runner.KubeflowV2DagRunner(config=runner_config)
    pipeline = tfx_pipeline.Pipeline('dummy', 'dummy_root')
    with patcher.patch() as context:
      runner.run(pipeline)
    self.assertIn(patcher.OUTPUT_FILE_PATH, context)

    mock_build_image_fn.assert_called_once_with(given_image_name)
    self.assertEqual(runner_config.default_image, built_image_name)

  def testPatcherSavePipelineFn(self):
    pipeline_name = 'dummy'
    pipeline_dir = '/foo/pipeline'
    mock_prepare_dir_fn = mock.MagicMock(return_value=pipeline_dir)
    patcher = kubeflow_v2_dag_runner_patcher.KubeflowV2DagRunnerPatcher(
        call_real_run=False, prepare_dir_fn=mock_prepare_dir_fn)
    runner = kubeflow_v2_dag_runner.KubeflowV2DagRunner(
        config=kubeflow_v2_dag_runner.KubeflowV2DagRunnerConfig())
    pipeline = tfx_pipeline.Pipeline(pipeline_name, 'dummy_root')
    with patcher.patch() as context:
      runner.run(pipeline)

    mock_prepare_dir_fn.assert_called_once_with(pipeline_name)
    self.assertEqual(
        context[patcher.OUTPUT_FILE_PATH],
        os.path.join(pipeline_dir,
                     kubeflow_v2_dag_runner_patcher.OUTPUT_FILENAME))


if __name__ == '__main__':
  tf.test.main()
