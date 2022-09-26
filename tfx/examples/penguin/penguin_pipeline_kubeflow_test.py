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
"""Tests for tfx.examples.penguin.penguin_pipeline_kubeflow."""

import os
from unittest import mock

from absl.testing import parameterized
import tensorflow as tf
from tfx.dsl.io import fileio
from tfx.examples.penguin import penguin_pipeline_kubeflow
from tfx.utils import test_case_utils
from tfx.v1 import orchestration


class PenguinPipelineKubeflowTest(test_case_utils.TfxTest,
                                  parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(test_case_utils.change_working_dir(self.tmp_dir))

  @parameterized.named_parameters(
      dict(testcase_name=' Local', use_aip=False, use_vertex=False),
      dict(testcase_name=' AIP', use_aip=True, use_vertex=False),
      dict(testcase_name=' Vertex', use_aip=False, use_vertex=True))
  @mock.patch('tfx.components.util.udf_utils.UserModuleFilePipDependency.'
              'resolve')
  def testPenguinPipelineConstructionAndDefinitionFileExists(
      self, resolve_mock, use_aip, use_vertex):
    # Avoid actually performing user module packaging because a placeholder
    # GCS bucket is used.
    resolve_mock.side_effect = lambda pipeline_root: None

    kubeflow_pipeline = penguin_pipeline_kubeflow.create_pipeline(
        pipeline_name=penguin_pipeline_kubeflow._pipeline_name,
        pipeline_root=penguin_pipeline_kubeflow._pipeline_root,
        data_root=penguin_pipeline_kubeflow._data_root,
        module_file=penguin_pipeline_kubeflow._module_file,
        enable_tuning=False,
        enable_cache=True,
        user_provided_schema_path=penguin_pipeline_kubeflow
        ._user_provided_schema,
        ai_platform_training_args=penguin_pipeline_kubeflow
        ._ai_platform_training_args,
        ai_platform_serving_args=penguin_pipeline_kubeflow
        ._ai_platform_serving_args,
        beam_pipeline_args=penguin_pipeline_kubeflow
        ._beam_pipeline_args_by_runner['DirectRunner'],
        use_cloud_component=False,
        use_aip=use_aip,
        use_vertex=use_vertex,
        serving_model_dir=penguin_pipeline_kubeflow._serving_model_dir)
    self.assertLen(kubeflow_pipeline.components, 9)

    if use_vertex:
      v2_dag_runner = orchestration.experimental.KubeflowV2DagRunner(
          config=orchestration.experimental.KubeflowV2DagRunnerConfig(),
          output_dir=self.tmp_dir,
          output_filename=penguin_pipeline_kubeflow._pipeline_definition_file)
      v2_dag_runner.run(kubeflow_pipeline)
      file_path = os.path.join(
          self.tmp_dir, penguin_pipeline_kubeflow._pipeline_definition_file)
      self.assertTrue(fileio.exists(file_path))
    else:
      v1_dag_runner = orchestration.experimental.KubeflowDagRunner(
          config=orchestration.experimental.KubeflowDagRunnerConfig(
              kubeflow_metadata_config=orchestration.experimental
              .get_default_kubeflow_metadata_config()))
      v1_dag_runner.run(kubeflow_pipeline)
      file_path = os.path.join(self.tmp_dir, 'penguin-kubeflow.tar.gz')
      self.assertTrue(fileio.exists(file_path))


if __name__ == '__main__':
  tf.test.main()
