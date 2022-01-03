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
"""E2E test using kubeflow orchestrator for taxi template."""

import os

from absl import logging
import tensorflow as tf
from tfx.experimental.templates import container_based_test_case
from tfx.orchestration.kubeflow import test_utils as kubeflow_test_utils


class TaxiTemplateKubeflowE2ETest(
    container_based_test_case.BaseKubeflowEndToEndTest):

  def tearDown(self):
    super().tearDown()
    self._delete_caip_model()

  def _generate_pipeline_name(self, random_id: str):
    return f'taxi-template-kubeflow-e2e-test-{random_id}'

  # retry is handled by kubeflow_test_utils.delete_ai_platform_model().
  def _delete_caip_model(self):
    model_name = self._pipeline_name.replace('-', '_')
    kubeflow_test_utils.delete_ai_platform_model(model_name)

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
    self._replaceFileContent(
        os.path.join('pipeline', 'configs.py'), [
            ('GOOGLE_CLOUD_REGION = \'\'',
             'GOOGLE_CLOUD_REGION = \'{}\''.format(self._GCP_REGION)),
        ])

    # Prepare data
    self._prepare_data()
    self._replaceFileContent('kubeflow_runner.py', [
        ('DATA_PATH = \'gs://{}/tfx-template/data/taxi/\'.format(configs.GCS_BUCKET_NAME)',
         'DATA_PATH = \'gs://{{}}/{}/{}\'.format(configs.GCS_BUCKET_NAME)'
         .format(self._DATA_DIRECTORY_NAME, self._pipeline_name)),
    ])

    self._compile_pipeline()
    self._check_telemetry_label()

    # Create a pipeline with only one component.
    self._create_pipeline()
    self._run_pipeline()

    # Update the pipeline to include all components.
    updated_pipeline_file = self._addAllComponents()
    logging.info('Updated %s to add all components to the pipeline.',
                 updated_pipeline_file)
    self._update_pipeline()
    self._run_pipeline()

    # Enable BigQuery
    self._uncomment(
        os.path.join('pipeline', 'pipeline.py'), [
            'query: str,',
            'example_gen = tfx.extensions.google_cloud_big_query.BigQueryExampleGen(',
            '    query=query)'
        ])
    self._uncomment('kubeflow_runner.py', [
        'query=configs.BIG_QUERY_QUERY',
        'beam_pipeline_args=configs\n',
        '.BIG_QUERY_WITH_DIRECT_RUNNER_BEAM_PIPELINE_ARGS,',
    ])
    logging.info('Added BigQueryExampleGen to pipeline.')
    self._update_pipeline()
    self._run_pipeline()

    # TODO(b/173065862) Re-enable Dataflow tests after timeout is resolved.
    #     # Enable Dataflow
    #     self._comment('kubeflow_runner.py', [
    #         'beam_pipeline_args=configs\n',
    #         '.BIG_QUERY_WITH_DIRECT_RUNNER_BEAM_PIPELINE_ARGS',
    #     ])
    #     self._uncomment('kubeflow_runner.py', [
    #         'beam_pipeline_args=configs.DATAFLOW_BEAM_PIPELINE_ARGS',
    #     ])
    #     logging.info('Added Dataflow to pipeline.')
    #     self._update_pipeline()
    #     self._run_pipeline()

    #     # Enable CAIP extension.
    #     self._comment('kubeflow_runner.py', [
    #         'beam_pipeline_args=configs.DATAFLOW_BEAM_PIPELINE_ARGS',
    #     ])
    self._uncomment('kubeflow_runner.py', [
        'ai_platform_training_args=configs.GCP_AI_PLATFORM_TRAINING_ARGS,',
        'ai_platform_serving_args=configs.GCP_AI_PLATFORM_SERVING_ARGS,',
    ])
    logging.info('Using CAIP trainer and pusher.')
    self._update_pipeline()
    self._run_pipeline()


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.test.main()
