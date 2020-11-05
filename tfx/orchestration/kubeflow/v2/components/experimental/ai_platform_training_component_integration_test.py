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

import tensorflow as tf
from tfx.components.common_nodes import importer_node
from tfx.dsl.component.experimental import placeholders
from tfx.orchestration import pipeline
from tfx.orchestration.kubeflow.v2 import test_utils
from tfx.orchestration.kubeflow.v2.components.experimental import ai_platform_training_component
from tfx.types import standard_artifacts
from tfx.types.experimental import simple_artifacts


_PIPELINE_NAME = 'aip_training_component_pipeline'

_EXAMPLE_IMPORTER = importer_node.ImporterNode(
    instance_name='examples',
    artifact_type=simple_artifacts.File,
    reimport=False,
    source_uri='gs://tfx-oss-testing-bucket/sample-data/mnist'
)
_TRAIN = ai_platform_training_component.create_ai_platform_training(
    name='simple_aip_training',
    project_id='tfx-oss-testing',
    region='us-central1',
    image_uri='gcr.io/tfx-oss-testing/caip-training:tfx-test',
    args=[
        '--dataset',
        placeholders.InputUriPlaceholder('examples'),
        '--model-dir',
        placeholders.OutputUriPlaceholder('model'),
        '--lr',
        placeholders.InputValuePlaceholder(
            'learning_rate'),
    ],
    scale_tier='BASIC',
    inputs={'examples': _EXAMPLE_IMPORTER.outputs['result']},
    outputs={'model': standard_artifacts.Model},
    parameters={'learning_rate': '0.001'})


class AiPlatformTrainingComponentIntegrationTest(
    test_utils.BaseAIPlatformPipelinesTest):

  def testSuccessfulExecution(self):
    aip_training_pipeline = pipeline.Pipeline(
        pipeline_name=_PIPELINE_NAME,
        pipeline_root=self._pipeline_root(_PIPELINE_NAME),
        components=[_EXAMPLE_IMPORTER, _TRAIN],
    )

    job_name = self._run_pipeline(aip_training_pipeline)

    self._check_job_status(job_name)


if __name__ == '__main__':
  tf.test.main()
