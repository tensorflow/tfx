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
"""Tests for tfx.tools.cli.handler.kubeflow_dag_runner_patcher."""

import os
from unittest import mock

import tensorflow as tf
from tfx.orchestration import pipeline as tfx_pipeline
from tfx.orchestration.kubeflow import kubeflow_dag_runner
from tfx.tools.cli.handler import kubeflow_dag_runner_patcher
from tfx.utils import test_case_utils


class KubeflowDagRunnerPatcherTest(test_case_utils.TfxTest):

  def setUp(self):
    super().setUp()
    self.enter_context(test_case_utils.change_working_dir(self.tmp_dir))

  def testPatcher(self):
    given_image_name = 'foo/bar'
    built_image_name = 'foo/bar@sha256:1234567890'

    mock_build_image_fn = mock.MagicMock(return_value=built_image_name)
    patcher = kubeflow_dag_runner_patcher.KubeflowDagRunnerPatcher(
        call_real_run=True,
        build_image_fn=mock_build_image_fn,
        use_temporary_output_file=True)
    runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
        tfx_image=given_image_name)
    runner = kubeflow_dag_runner.KubeflowDagRunner(config=runner_config)
    pipeline = tfx_pipeline.Pipeline('dummy', 'dummy_root')
    with patcher.patch() as context:
      runner.run(pipeline)
    self.assertTrue(context[patcher.USE_TEMPORARY_OUTPUT_FILE])
    self.assertIn(patcher.OUTPUT_FILE_PATH, context)

    mock_build_image_fn.assert_called_once_with(given_image_name)
    self.assertEqual(runner_config.tfx_image, built_image_name)

  def testPatcherWithOutputFile(self):
    output_filename = 'foo.tar.gz'
    patcher = kubeflow_dag_runner_patcher.KubeflowDagRunnerPatcher(
        call_real_run=False,
        build_image_fn=None,
        use_temporary_output_file=True)
    runner = kubeflow_dag_runner.KubeflowDagRunner(
        output_filename=output_filename)
    pipeline = tfx_pipeline.Pipeline('dummy', 'dummy_root')
    with patcher.patch() as context:
      runner.run(pipeline)
    self.assertFalse(context[patcher.USE_TEMPORARY_OUTPUT_FILE])
    self.assertEqual(
        os.path.basename(context[patcher.OUTPUT_FILE_PATH]), output_filename)
    self.assertEqual(runner._output_filename, output_filename)


if __name__ == '__main__':
  tf.test.main()
