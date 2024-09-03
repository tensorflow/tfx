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

import collections
import datetime
import json
import os
from unittest import mock

from absl.testing import parameterized
from tfx import version
from tfx.dsl.components.base import base_component
from tfx.orchestration import pipeline as tfx_pipeline
from tfx.orchestration.kubeflow.v2 import kubeflow_v2_dag_runner
from tfx.orchestration.kubeflow.v2 import test_utils
from tfx.utils import telemetry_utils
from tfx.utils import test_case_utils
import yaml

_TEST_DIR = 'testdir'

_TEST_FILE_NAME = 'test_pipeline_1.json'
_TEST_YAML_FILE_NAME = 'test_pipeline_1.yaml'

_ILLEGALLY_NAMED_PIPELINE = tfx_pipeline.Pipeline(
    pipeline_name='ThisIsIllegal', pipeline_root='/some/path', components=[])


class KubeflowV2DagRunnerTest(test_case_utils.TfxTest, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(test_case_utils.change_working_dir(self.tmp_dir))
    VersionInfo = collections.namedtuple(
        'VersionInfo', ['major', 'minor', 'micro']
    )
    self.enter_context(mock.patch('sys.version_info', new=VersionInfo(3, 7, 0)))

  def _compare_against_testdata(
      self,
      runner: kubeflow_v2_dag_runner.KubeflowV2DagRunner,
      pipeline: tfx_pipeline.Pipeline,
      golden_file: str,
      use_legacy_data: bool = False,
      use_yaml_file: bool = False,
  ):
    """Compiles and compares the actual JSON/YAML output against a golden file."""
    actual_output = runner.run(pipeline=pipeline, write_out=True)

    expected_json = json.loads(
        test_utils.get_text_from_test_data(
            golden_file, use_legacy_data=use_legacy_data
        )
    )
    expected_json['pipelineSpec']['sdkVersion'] = 'tfx-{}'.format(
        version.__version__)
    if 'labels' in expected_json:
      expected_json['labels']['tfx_version'] = telemetry_utils._normalize_label(
          version.__version__)

    self.assertDictEqual(actual_output, expected_json)

    if use_yaml_file:
      with open(
          os.path.join(_TEST_DIR, _TEST_YAML_FILE_NAME)
      ) as pipeline_yaml_file:
        actual_json = yaml.safe_load(pipeline_yaml_file)
        expected_json = expected_json['pipelineSpec']
    else:
      with open(os.path.join(_TEST_DIR, _TEST_FILE_NAME)) as pipeline_json_file:
        actual_json = json.load(pipeline_json_file)

    self.assertDictEqual(actual_json, expected_json)

  @parameterized.named_parameters(
      dict(
          testcase_name='use_pipeline_spec_2_1_and_json_file',
          use_pipeline_spec_2_1=True,
          use_yaml_file=False,
      ),
      dict(
          testcase_name='use_pipeline_spec_2_0_and_json_file',
          use_pipeline_spec_2_1=False,
          use_yaml_file=False,
      ),
      dict(
          testcase_name='use_pipeline_spec_2_1_and_yaml_file',
          use_pipeline_spec_2_1=True,
          use_yaml_file=True,
      ),
      dict(
          testcase_name='use_pipeline_spec_2_0_and_yaml_file',
          use_pipeline_spec_2_1=False,
          use_yaml_file=True,
      ),
  )
  @mock.patch(
      'tfx.orchestration.kubeflow.v2.kubeflow_v2_dag_runner._get_current_time'
  )
  def testCompileTwoStepPipeline(
      self, fake_now, use_pipeline_spec_2_1, use_yaml_file=False
  ):
    fake_now.return_value = datetime.date(2020, 1, 1)
    output_filename = _TEST_YAML_FILE_NAME if use_yaml_file else _TEST_FILE_NAME
    runner = kubeflow_v2_dag_runner.KubeflowV2DagRunner(
        output_dir=_TEST_DIR,
        output_filename=output_filename,
        config=kubeflow_v2_dag_runner.KubeflowV2DagRunnerConfig(
            display_name='my-pipeline',
            default_image='gcr.io/my-tfx:latest',
            use_pipeline_spec_2_1=use_pipeline_spec_2_1,
        ),
    )

    self._compare_against_testdata(
        runner=runner,
        pipeline=test_utils.two_step_pipeline(),
        golden_file='expected_two_step_pipeline_job.json',
        use_legacy_data=not (use_pipeline_spec_2_1),
        use_yaml_file=use_yaml_file,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='use_pipeline_spec_2_1_and_json_file',
          use_pipeline_spec_2_1=True,
          use_yaml_file=False,
      ),
      dict(
          testcase_name='use_pipeline_spec_2_0_and_json_file',
          use_pipeline_spec_2_1=False,
          use_yaml_file=False,
      ),
      dict(
          testcase_name='use_pipeline_spec_2_1_and_yaml_file',
          use_pipeline_spec_2_1=True,
          use_yaml_file=True,
      ),
      dict(
          testcase_name='use_pipeline_spec_2_0_and_yaml_file',
          use_pipeline_spec_2_1=False,
          use_yaml_file=True,
      ),
  )
  @mock.patch(
      'tfx.orchestration.kubeflow.v2.kubeflow_v2_dag_runner._get_current_time'
  )
  def testCompileTwoStepPipelineWithMultipleImages(
      self, fake_now, use_pipeline_spec_2_1, use_yaml_file=False
  ):
    fake_now.return_value = datetime.date(2020, 1, 1)
    images = {
        kubeflow_v2_dag_runner._DEFAULT_IMAGE_PATH_KEY: 'gcr.io/my-tfx:latest',
        'BigQueryExampleGen': 'gcr.io/big-query:1.0.0',
    }
    output_filename = _TEST_YAML_FILE_NAME if use_yaml_file else _TEST_FILE_NAME
    runner = kubeflow_v2_dag_runner.KubeflowV2DagRunner(
        output_dir=_TEST_DIR,
        output_filename=output_filename,
        config=kubeflow_v2_dag_runner.KubeflowV2DagRunnerConfig(
            display_name='my-pipeline',
            default_image=images,
            use_pipeline_spec_2_1=use_pipeline_spec_2_1,
        ),
    )

    self._compare_against_testdata(
        runner=runner,
        pipeline=test_utils.two_step_pipeline(),
        golden_file='expected_two_step_pipeline_job_with_multiple_images.json',
        use_legacy_data=not use_pipeline_spec_2_1,
        use_yaml_file=use_yaml_file,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='use_pipeline_spec_2_1_and_json_file',
          use_pipeline_spec_2_1=True,
          use_yaml_file=False,
      ),
      dict(
          testcase_name='use_pipeline_spec_2_0_and_json_file',
          use_pipeline_spec_2_1=False,
          use_yaml_file=False,
      ),
      dict(
          testcase_name='use_pipeline_spec_2_1_and_yaml_file',
          use_pipeline_spec_2_1=True,
          use_yaml_file=True,
      ),
      dict(
          testcase_name='use_pipeline_spec_2_0_and_yaml_file',
          use_pipeline_spec_2_1=False,
          use_yaml_file=True,
      ),
  )
  @mock.patch('tfx.version')
  @mock.patch(
      'tfx.orchestration.kubeflow.v2.kubeflow_v2_dag_runner._get_current_time'
  )
  def testCompileTwoStepPipelineWithoutDefaultImage(
      self,
      fake_now,
      fake_tfx_version,
      use_pipeline_spec_2_1,
      use_yaml_file=False,
  ):
    fake_now.return_value = datetime.date(2020, 1, 1)
    fake_tfx_version.__version__ = '1.13.0.dev'
    images = {
        'BigQueryExampleGen': 'gcr.io/big-query:1.0.0',
    }
    output_filename = _TEST_YAML_FILE_NAME if use_yaml_file else _TEST_FILE_NAME
    runner = kubeflow_v2_dag_runner.KubeflowV2DagRunner(
        output_dir=_TEST_DIR,
        output_filename=output_filename,
        config=kubeflow_v2_dag_runner.KubeflowV2DagRunnerConfig(
            display_name='my-pipeline',
            default_image=images,
            use_pipeline_spec_2_1=use_pipeline_spec_2_1,
        ),
    )

    self._compare_against_testdata(
        runner=runner,
        pipeline=test_utils.two_step_pipeline(),
        golden_file='expected_two_step_pipeline_job_without_default_image.json',
        use_legacy_data=not use_pipeline_spec_2_1,
        use_yaml_file=use_yaml_file,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='use_pipeline_spec_2_1_and_json_file',
          use_pipeline_spec_2_1=True,
          use_yaml_file=False,
      ),
      dict(
          testcase_name='use_pipeline_spec_2_0_and_json_file',
          use_pipeline_spec_2_1=False,
          use_yaml_file=False,
      ),
      dict(
          testcase_name='use_pipeline_spec_2_1_and_yaml_file',
          use_pipeline_spec_2_1=True,
          use_yaml_file=True,
      ),
      dict(
          testcase_name='use_pipeline_spec_2_0_and_yaml_file',
          use_pipeline_spec_2_1=False,
          use_yaml_file=True,
      ),
  )
  @mock.patch.object(base_component.BaseComponent, '_resolve_pip_dependencies')
  @mock.patch(
      'tfx.orchestration.kubeflow.v2.kubeflow_v2_dag_runner._get_current_time'
  )
  def testCompileFullTaxiPipeline(
      self,
      fake_now,
      moke_resolve_dependencies,
      use_pipeline_spec_2_1,
      use_yaml_file=False,
  ):
    fake_now.return_value = datetime.date(2020, 1, 1)
    moke_resolve_dependencies.return_value = None

    output_filename = _TEST_YAML_FILE_NAME if use_yaml_file else _TEST_FILE_NAME
    runner = kubeflow_v2_dag_runner.KubeflowV2DagRunner(
        output_dir=_TEST_DIR,
        output_filename=output_filename,
        config=kubeflow_v2_dag_runner.KubeflowV2DagRunnerConfig(
            display_name='my-pipeline',
            default_image='tensorflow/tfx:latest',
            use_pipeline_spec_2_1=use_pipeline_spec_2_1,
        ),
    )

    self._compare_against_testdata(
        runner=runner,
        pipeline=test_utils.full_taxi_pipeline(),
        golden_file='expected_full_taxi_pipeline_job.json',
        use_legacy_data=not use_pipeline_spec_2_1,
        use_yaml_file=use_yaml_file,
    )
    moke_resolve_dependencies.assert_called()
