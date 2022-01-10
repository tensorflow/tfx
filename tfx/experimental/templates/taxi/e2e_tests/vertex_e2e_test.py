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
"""E2E test using Kubeflow V2 runner for taxi template."""

import os

from absl import logging
import tensorflow as tf
from tfx.experimental.templates import container_based_test_case


class TaxiTemplateKubeflowV2E2ETest(
    container_based_test_case.BaseVertexEndToEndTest):

  def _generate_pipeline_name(self, random_id: str):
    return f'taxi-template-vertex-e2e-{random_id}'

  def testPipeline(self):
    self._copyTemplate('taxi')

    # Uncomment all variables in config.
    self._uncommentMultiLineVariables(
        os.path.join('pipeline', 'configs.py'), [
            'GOOGLE_CLOUD_REGION',
            'BIG_QUERY_WITH_DIRECT_RUNNER_BEAM_PIPELINE_ARGS',
            'BIG_QUERY_QUERY', 'DATAFLOW_BEAM_PIPELINE_ARGS',
            'GCP_AI_PLATFORM_TRAINING_ARGS', 'GCP_AI_PLATFORM_SERVING_ARGS'
        ])

    # Prepare data.
    self._prepare_data()
    self._replaceFileContent('kubeflow_v2_runner.py', [
        ('_DATA_PATH = \'gs://{}/tfx-template/data/taxi/\'.format(configs.GCS_BUCKET_NAME)',
         '_DATA_PATH = \'gs://{{}}/{}/{}\'.format(configs.GCS_BUCKET_NAME)'
         .format(self._DATA_DIRECTORY_NAME, self._pipeline_name)),
    ])
    self._replaceFileContent(
        os.path.join('pipeline', 'configs.py'),
        [('GCS_BUCKET_NAME = GOOGLE_CLOUD_PROJECT + \'-kubeflowpipelines-default\'',
          f'GCS_BUCKET_NAME = \'{self._BUCKET_NAME}\'')])

    # Create a pipeline with only one component.
    self._create_pipeline()

    # Update the pipeline to include all components.
    updated_pipeline_file = self._addAllComponents()
    logging.info('Updated %s to add all components to the pipeline.',
                 updated_pipeline_file)
    self._update_pipeline()
    self._run_pipeline()


if __name__ == '__main__':
  tf.test.main()
