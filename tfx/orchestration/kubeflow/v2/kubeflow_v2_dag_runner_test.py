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
"""Tests for Kubeflow V2 runner."""

import datetime
import json
import os
from unittest import mock


import tensorflow as tf
from tfx import version
from tfx.dsl.components.base import base_component
from tfx.orchestration import pipeline as tfx_pipeline
from tfx.orchestration.kubeflow.v2 import kubeflow_v2_dag_runner
from tfx.orchestration.kubeflow.v2 import test_utils
from tfx.utils import telemetry_utils
from tfx.utils import test_case_utils

_TEST_DIR = 'testdir'

_TEST_FILE_NAME = 'test_pipeline_1.json'

_ILLEGALLY_NAMED_PIPELINE = tfx_pipeline.Pipeline(
    pipeline_name='ThisIsIllegal', pipeline_root='/some/path', components=[])


class KubeflowV2DagRunnerTest(test_case_utils.TfxTest):

  def setUp(self):
    super().setUp()
    self.enter_context(test_case_utils.change_working_dir(self.tmp_dir))

  def _compare_against_testdata(
      self, runner: kubeflow_v2_dag_runner.KubeflowV2DagRunner,
      pipeline: tfx_pipeline.Pipeline, golden_file: str):
    """Compiles and compare the actual JSON output against a golden file."""
    actual_output = runner.run(pipeline=pipeline, write_out=True)

    expected_json = json.loads(test_utils.get_text_from_test_data(golden_file))
    expected_json['pipelineSpec']['sdkVersion'] = 'tfx-{}'.format(
        version.__version__)
    if 'labels' in expected_json:
      expected_json['labels']['tfx_version'] = telemetry_utils._normalize_label(
          version.__version__)

    self.assertDictEqual(actual_output, expected_json)

    with open(os.path.join(_TEST_DIR, _TEST_FILE_NAME)) as pipeline_json_file:
      actual_json = json.load(pipeline_json_file)

    self.assertDictEqual(actual_json, expected_json)

  @mock.patch('sys.version_info')
  @mock.patch(
      'tfx.orchestration.kubeflow.v2.kubeflow_v2_dag_runner._get_current_time')
  def testCompileTwoStepPipeline(self, fake_now, fake_sys_version):
    fake_now.return_value = datetime.date(2020, 1, 1)
    fake_sys_version.major = 3
    fake_sys_version.minor = 7
    runner = kubeflow_v2_dag_runner.KubeflowV2DagRunner(
        output_dir=_TEST_DIR,
        output_filename=_TEST_FILE_NAME,
        config=kubeflow_v2_dag_runner.KubeflowV2DagRunnerConfig(
            display_name='my-pipeline',
            default_image='gcr.io/my-tfx:latest'))

    self._compare_against_testdata(
        runner=runner,
        pipeline=test_utils.two_step_pipeline(),
        golden_file='expected_two_step_pipeline_job.json')

  @mock.patch.object(base_component.BaseComponent, '_resolve_pip_dependencies')
  @mock.patch('sys.version_info')
  @mock.patch(
      'tfx.orchestration.kubeflow.v2.kubeflow_v2_dag_runner._get_current_time')
  def testCompileFullTaxiPipeline(self, fake_now, fake_sys_version,
                                  moke_resolve_dependencies):
    fake_now.return_value = datetime.date(2020, 1, 1)
    fake_sys_version.major = 3
    fake_sys_version.minor = 7
    moke_resolve_dependencies.return_value = None

    runner = kubeflow_v2_dag_runner.KubeflowV2DagRunner(
        output_dir=_TEST_DIR,
        output_filename=_TEST_FILE_NAME,
        config=kubeflow_v2_dag_runner.KubeflowV2DagRunnerConfig(
            display_name='my-pipeline',
            default_image='tensorflow/tfx:latest'))

    self._compare_against_testdata(
        runner=runner,
        pipeline=test_utils.full_taxi_pipeline(),
        golden_file='expected_full_taxi_pipeline_job.json')
    moke_resolve_dependencies.assert_called()

if __name__ == '__main__':
  tf.test.main()
