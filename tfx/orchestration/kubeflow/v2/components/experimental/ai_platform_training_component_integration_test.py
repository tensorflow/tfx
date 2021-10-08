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
"""Integration tests for AI Platform Training component."""

import os

import tensorflow as tf
from tfx.dsl.component.experimental import placeholders
from tfx.dsl.components.common import importer
from tfx.orchestration import pipeline
from tfx.orchestration import test_utils
from tfx.orchestration.kubeflow.v2.components.experimental import ai_platform_training_component
from tfx.orchestration.kubeflow.v2.e2e_tests import base_test_case
from tfx.types import standard_artifacts
from tfx.types.experimental import simple_artifacts

_PIPELINE_NAME_PREFIX = 'aip-training-component-pipeline-{}'


class AiPlatformTrainingComponentIntegrationTest(
    base_test_case.BaseKubeflowV2Test):
  """Integration tests of AiPlatformTrainingComponent on managed pipeline."""

  _TEST_DATA_BUCKET = os.environ.get('CAIP_E2E_DATA_BUCKET')
  _TRAINING_IMAGE = os.environ.get('CAIP_TRAINING_COMPONENT_TEST_IMAGE')

  def testSuccessfulExecution(self):
    example_importer = importer.Importer(
        artifact_type=simple_artifacts.File,
        reimport=False,
        source_uri=f'gs://{self._TEST_DATA_BUCKET}/ai-platform-training/mnist'
    ).with_id('examples')

    train = ai_platform_training_component.create_ai_platform_training(
        name='simple_aip_training',
        project_id=self._GCP_PROJECT_ID,
        region=self._GCP_REGION,
        image_uri=self._TRAINING_IMAGE,
        args=[
            '--dataset',
            placeholders.InputUriPlaceholder('examples'),
            '--model-dir',
            placeholders.OutputUriPlaceholder('model'),
            '--lr',
            placeholders.InputValuePlaceholder('learning_rate'),
        ],
        scale_tier='BASIC',
        inputs={'examples': example_importer.outputs['result']},
        outputs={'model': standard_artifacts.Model},
        parameters={'learning_rate': '0.001'})

    pipeline_name = _PIPELINE_NAME_PREFIX.format(test_utils.random_id())
    aip_training_pipeline = pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=self._pipeline_root(pipeline_name),
        components=[example_importer, train],
    )

    self._run_pipeline(aip_training_pipeline)


if __name__ == '__main__':
  tf.test.main()
