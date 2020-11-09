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
import shutil
import tempfile
from typing import Text

# Standard Imports

import mock
import tensorflow as tf
from tfx import version
from tfx.orchestration import pipeline as tfx_pipeline
from tfx.orchestration.kubeflow.v2 import kubeflow_v2_dag_runner
from tfx.orchestration.kubeflow.v2 import test_utils
from tfx.utils import telemetry_utils

_TEST_DIR = 'testdir'

_TEST_FILE_NAME = 'test_pipeline_1.json'

_ILLEGALLY_NAMED_PIPELINE = tfx_pipeline.Pipeline(
    pipeline_name='ThisIsIllegal', pipeline_root='/some/path', components=[])

# Sample versions used for testing SDK to docker version transformation.
_SAMPLE_RELEASED_VERSION = '0.25.0'
_SAMPLE_RC_VERSION = '0.25.0-rc1'
_SAMPLE_NIGHTLY_VERSION = '0.25.0.dev20201101'
_SAMPLE_DEV_VERSION = '0.26.0.dev'
_EXPECTED_RELEASED_DOCKER_VERSION = '0.25.0'
_EXPECTED_RC_DOCKER_VERSION = '0.25.0rc1'
_EXPECTED_NIGHTLY_DOCKER_VERSION = '0.25.0.dev20201101'
_EXPECTED_DEV_DOCKER_VERSION = 'latest'


class KubeflowV2DagRunnerTest(tf.test.TestCase):

  def setUp(self):
    super(KubeflowV2DagRunnerTest, self).setUp()
    self.test_dir = tempfile.mkdtemp()
    os.chdir(self.test_dir)

  def tearDown(self):
    super(KubeflowV2DagRunnerTest, self).tearDown()
    shutil.rmtree(self.test_dir)

  def _compare_against_testdata(
      self, runner: kubeflow_v2_dag_runner.KubeflowV2DagRunner,
      pipeline: tfx_pipeline.Pipeline, golden_file: Text):
    """Compiles and compare the actual JSON output against a golden file."""
    actual_output = runner.run(pipeline=pipeline, write_out=True)

    expected_json = json.loads(test_utils.get_text_from_test_data(golden_file))
    expected_json['pipelineSpec']['sdkVersion'] = version.__version__
    if 'labels' in expected_json:
      expected_json['labels']['tfx_version'] = telemetry_utils._normalize_label(
          version.__version__)

    self.assertDictEqual(actual_output, expected_json)

    with open(os.path.join(_TEST_DIR, _TEST_FILE_NAME)) as pipeline_json_file:
      actual_json = json.load(pipeline_json_file)

    self.assertDictEqual(actual_json, expected_json)

  def testImageVersionFormatting(self):
    self.assertEqual(
        kubeflow_v2_dag_runner.get_image_version(_SAMPLE_RELEASED_VERSION),
        _EXPECTED_RELEASED_DOCKER_VERSION)
    self.assertEqual(
        kubeflow_v2_dag_runner.get_image_version(_SAMPLE_RC_VERSION),
        _EXPECTED_RC_DOCKER_VERSION)
    self.assertEqual(
        kubeflow_v2_dag_runner.get_image_version(_SAMPLE_NIGHTLY_VERSION),
        _EXPECTED_NIGHTLY_DOCKER_VERSION)
    self.assertEqual(
        kubeflow_v2_dag_runner.get_image_version(_SAMPLE_DEV_VERSION),
        _EXPECTED_DEV_DOCKER_VERSION)

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
            project_id='my-project',
            display_name='my-pipeline',
            default_image='gcr.io/my-tfx:latest'))

    self._compare_against_testdata(
        runner=runner,
        pipeline=test_utils.two_step_pipeline(),
        golden_file='expected_two_step_pipeline_job.json')

  @mock.patch('sys.version_info')
  @mock.patch(
      'tfx.orchestration.kubeflow.v2.kubeflow_v2_dag_runner._get_current_time')
  def testCompileFullTaxiPipeline(self, fake_now, fake_sys_version):
    fake_now.return_value = datetime.date(2020, 1, 1)
    fake_sys_version.major = 3
    fake_sys_version.minor = 7
    runner = kubeflow_v2_dag_runner.KubeflowV2DagRunner(
        output_dir=_TEST_DIR,
        output_filename=_TEST_FILE_NAME,
        config=kubeflow_v2_dag_runner.KubeflowV2DagRunnerConfig(
            project_id='my-project',
            display_name='my-pipeline',
            default_image='tensorflow/tfx:latest'))

    self._compare_against_testdata(
        runner=runner,
        pipeline=test_utils.full_taxi_pipeline(),
        golden_file='expected_full_taxi_pipeline_job.json')


if __name__ == '__main__':
  tf.test.main()
