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

import datetime
import os

from absl import logging
from google.cloud import storage
from google.cloud import aiplatform

import tensorflow as tf
from tfx.experimental.templates import test_utils
from tfx.orchestration import test_utils as orchestration_test_utils
from tfx.orchestration.kubeflow.v2 import vertex_client_utils
from tfx.utils import docker_utils
from tfx.utils import retry


class TaxiTemplateKubeflowV2E2ETest(test_utils.BaseEndToEndTest):

  _POLLING_INTERVAL_IN_SECONDS = 30
  _TIME_OUT = datetime.timedelta(hours=2)

  _DATA_DIRECTORY_NAME = 'template_data'

  # The following environment variables need to be set prior to calling the test
  # in this file. All variables are required and do not have a default.

  # The base container image name to use when building the image used in tests.
  _BASE_CONTAINER_IMAGE = os.environ['KFP_E2E_BASE_CONTAINER_IMAGE']

  # The src path to use to build docker image
  _REPO_BASE = os.environ['KFP_E2E_SRC']

  _GCP_PROJECT_ID = os.environ['KFP_E2E_GCP_PROJECT_ID']
  _GCP_REGION = os.environ['KFP_E2E_GCP_REGION']
  _BUCKET_NAME = os.environ['KFP_E2E_BUCKET_NAME']

  def setUp(self):
    super().setUp()
    random_id = orchestration_test_utils.random_id()
    if ':' not in self._BASE_CONTAINER_IMAGE:
      self._base_container_image = '{}:{}'.format(self._BASE_CONTAINER_IMAGE,
                                                  random_id)
      self._prepare_base_container_image()
    else:
      self._base_container_image = self._BASE_CONTAINER_IMAGE
    # Overriding the pipeline name to
    self._pipeline_name = 'taxi-template-vertex-e2e-{}'.format(random_id)
    self._target_container_image = 'gcr.io/{}/{}'.format(
        self._GCP_PROJECT_ID, self._pipeline_name)

  def tearDown(self):
    super().tearDown()
    self._delete_target_container_image()
    self._delete_base_container_image()
    self._delete_pipeline_data()

  # TODO(b/174289068): Refactor duplicated cleanup routines.
  @retry.retry(ignore_eventual_failure=True)
  def _delete_pipeline_data(self):
    path = 'tfx_pipeline_output/{}'.format(self._pipeline_name)
    orchestration_test_utils.delete_gcs_files(self._GCP_PROJECT_ID,
                                              self._BUCKET_NAME, path)
    path = '{}/{}'.format(self._DATA_DIRECTORY_NAME, self._pipeline_name)
    orchestration_test_utils.delete_gcs_files(self._GCP_PROJECT_ID,
                                              self._BUCKET_NAME, path)

  @retry.retry(ignore_eventual_failure=True)
  def _delete_base_container_image(self):
    if self._base_container_image == self._BASE_CONTAINER_IMAGE:
      return  # Didn't generate a base image for the test.
    docker_utils.delete_image(self._base_container_image)

  @retry.retry(ignore_eventual_failure=True)
  def _delete_target_container_image(self):
    docker_utils.delete_image(self._target_container_image)

  def _prepare_base_container_image(self):
    orchestration_test_utils.build_docker_image(self._base_container_image,
                                                self._REPO_BASE)

  def _create_pipeline(self):
    self._runCli([
        'pipeline',
        'create',
        '--engine',
        'vertex',
        '--pipeline-path',
        'kubeflow_v2_runner.py',
        '--build-image',
        '--build-base-image',
        self._base_container_image,
    ])

  def _update_pipeline(self):
    self._runCli([
        'pipeline',
        'update',
        '--engine',
        'vertex',
        '--pipeline_path',
        'kubeflow_v2_runner.py',
        '--build-image',
    ])

  def _run_pipeline(self):
    result = self._runCli([
        'run',
        'create',
        '--engine',
        'vertex',
        '--pipeline_name',
        self._pipeline_name,
        '--project',
        self._GCP_PROJECT_ID,
        '--region',
        self._GCP_REGION,
    ])
    run_id = self._parse_run_id(result)
    self._wait_until_completed(run_id)

  def _parse_run_id(self, output: str):
    run_id_lines = [
        line for line in output.split('\n')
        if line.startswith('| ')
    ][1:]  # Skip header line.
    self.assertLen(run_id_lines, 1)
    return run_id_lines[0].split('|')[1].strip()

  def _wait_until_completed(self, run_id: str):
    aiplatform.init(
        project=self._GCP_PROJECT_ID,
        location=self._GCP_REGION,
    )
    vertex_client_utils.poll_job_status(run_id, self._TIME_OUT,
                                        self._POLLING_INTERVAL_IN_SECONDS)

  def _prepare_data(self):
    """Uploads the csv data from local to GCS location."""
    gcs_client = storage.Client(project=self._GCP_PROJECT_ID)
    bucket = gcs_client.bucket(self._BUCKET_NAME)
    blob = bucket.blob('{}/{}/data.csv'.format(self._DATA_DIRECTORY_NAME,
                                               self._pipeline_name))
    blob.upload_from_filename('data/data.csv')

  def testPipeline(self):
    self._copyTemplate('taxi')
    os.environ['VERTEX_HOME'] = os.path.join(self._temp_dir, 'vertex')

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
