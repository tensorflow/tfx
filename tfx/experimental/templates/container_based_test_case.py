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
"""Base classes for container based e2e tests."""

import datetime
import os
import subprocess
import tarfile

from absl import logging
from google.cloud import aiplatform
import kfp
from tfx.dsl.io import fileio
from tfx.experimental.templates import test_utils
from tfx.orchestration import test_utils as orchestration_test_utils
from tfx.orchestration.kubeflow import test_utils as kubeflow_test_utils
from tfx.orchestration.kubeflow.v2 import vertex_client_utils
from tfx.utils import docker_utils
from tfx.utils import io_utils
from tfx.utils import retry
from tfx.utils import telemetry_utils
from tfx.utils import test_case_utils
import yaml


class BaseContainerBasedEndToEndTest(test_utils.BaseEndToEndTest):
  """Common utilities for kubeflow/vertex engine."""

  _POLLING_INTERVAL_IN_SECONDS = 30
  _TIME_OUT = datetime.timedelta(hours=2)

  _DATA_DIRECTORY_NAME = 'template_data'

  # The following environment variables need to be set prior to calling the test
  # in this file. All variables are required and do not have a default.

  # The base container image name to use when building the image used in tests.
  _BASE_CONTAINER_IMAGE = os.environ['KFP_E2E_BASE_CONTAINER_IMAGE']

  # The src path to use to build docker image
  _REPO_BASE = os.environ['KFP_E2E_SRC']

  # The project id to use to run tests.
  _GCP_PROJECT_ID = os.environ['KFP_E2E_GCP_PROJECT_ID']

  # The GCP region in which the end-to-end test is run.
  _GCP_REGION = os.environ['KFP_E2E_GCP_REGION']

  # The GCP bucket to use to write output artifacts.
  _BUCKET_NAME = os.environ['KFP_E2E_BUCKET_NAME']

  def setUp(self):
    super().setUp()
    random_id = orchestration_test_utils.random_id()
    self._pipeline_name = self._generate_pipeline_name(random_id)
    logging.info('Pipeline: %s', self._pipeline_name)

    if ':' not in self._BASE_CONTAINER_IMAGE:
      self._base_container_image = '{}:{}'.format(self._BASE_CONTAINER_IMAGE,
                                                  random_id)
      self._prepare_base_container_image()
    else:
      self._base_container_image = self._BASE_CONTAINER_IMAGE

    self._target_container_image = 'gcr.io/{}/{}'.format(
        self._GCP_PROJECT_ID, self._pipeline_name)

  def tearDown(self):
    super().tearDown()
    self._delete_target_container_image()
    self._delete_base_container_image()
    self._delete_pipeline_data()

  def _prepare_base_container_image(self):
    orchestration_test_utils.build_docker_image(self._base_container_image,
                                                self._REPO_BASE)

  def _prepare_data(self):
    io_utils.copy_file(
        'data/data.csv',
        f'gs://{self._BUCKET_NAME}/{self._DATA_DIRECTORY_NAME}/'
        + f'{self._pipeline_name}/data.csv')

  @retry.retry(ignore_eventual_failure=True)
  def _delete_pipeline_data(self):
    path = f'gs://{self._BUCKET_NAME}/tfx_pipeline_output/{self._pipeline_name}'
    io_utils.delete_dir(path)
    path = (f'gs://{self._BUCKET_NAME}/{self._DATA_DIRECTORY_NAME}/'
            f'{self._pipeline_name}')
    io_utils.delete_dir(path)

  @retry.retry(ignore_eventual_failure=True)
  def _delete_base_container_image(self):
    if self._base_container_image == self._BASE_CONTAINER_IMAGE:
      return  # Didn't generate a base image for the test.
    docker_utils.delete_image(self._base_container_image)

  @retry.retry(ignore_eventual_failure=True)
  def _delete_target_container_image(self):
    docker_utils.delete_image(self._target_container_image)


class BaseKubeflowEndToEndTest(BaseContainerBasedEndToEndTest):
  """Common utilities for kubeflow engine."""

  _RETRY_LIMIT = 3

  # This default bucket name is valid for KFP marketplace deployment since KFP
  # version 0.5.0.
  _BUCKET_NAME = (
      BaseContainerBasedEndToEndTest._GCP_PROJECT_ID +
      '-kubeflowpipelines-default')

  def setUp(self):
    super().setUp()
    self._namespace = 'kubeflow'
    self._endpoint = self._get_endpoint(self._namespace)
    self._kfp_client = kfp.Client(host=self._endpoint)
    logging.info('ENDPOINT: %s', self._endpoint)
    self.enter_context(
        test_case_utils.override_env_var(
            'KUBEFLOW_HOME', os.path.join(self._temp_dir, 'kubeflow')))

  def tearDown(self):
    super().tearDown()
    self._delete_runs()
    self._delete_pipeline()

  def _get_endpoint(self, namespace):
    cmd = 'kubectl describe configmap inverse-proxy-config -n {}'.format(
        namespace)
    output = subprocess.check_output(cmd.split())
    for line in output.decode('utf-8').split('\n'):
      if line.endswith('googleusercontent.com'):
        return line

  def _get_kfp_runs(self):
    # CLI uses experiment_name which is the same as pipeline_name.
    experiment_id = self._kfp_client.get_experiment(
        experiment_name=self._pipeline_name).id
    response = self._kfp_client.list_runs(experiment_id=experiment_id)
    return response.runs

  @retry.retry(ignore_eventual_failure=True)
  def _delete_runs(self):
    for run in self._get_kfp_runs():
      self._kfp_client._run_api.delete_run(id=run.id)  # pylint: disable=protected-access

  @retry.retry(ignore_eventual_failure=True)
  def _delete_pipeline(self):
    self._runCli([
        'pipeline', 'delete', '--engine', 'kubeflow', '--pipeline_name',
        self._pipeline_name
    ])

  def _parse_run_id(self, output: str):
    run_id_lines = [
        line for line in output.split('\n')
        if '| {} |'.format(self._pipeline_name) in line
    ]
    self.assertLen(run_id_lines, 1)
    return run_id_lines[0].split('|')[2].strip()

  def _wait_until_completed(self, run_id: str):
    end_state = kubeflow_test_utils.poll_kfp_with_retry(
        self._endpoint, run_id, self._RETRY_LIMIT, self._TIME_OUT,
        self._POLLING_INTERVAL_IN_SECONDS)
    self.assertEqual(end_state.lower(), kubeflow_test_utils.KFP_SUCCESS_STATUS)

  def _create_pipeline(self):
    self._runCli([
        'pipeline',
        'create',
        '--engine',
        'kubeflow',
        '--pipeline_path',
        'kubeflow_runner.py',
        '--endpoint',
        self._endpoint,
        '--build-image',
        '--build-base-image',
        self._base_container_image,
    ])

  def _compile_pipeline(self):
    self._runCli([
        'pipeline',
        'compile',
        '--engine',
        'kubeflow',
        '--pipeline_path',
        'kubeflow_runner.py',
    ])

  def _update_pipeline(self):
    self._runCli([
        'pipeline',
        'update',
        '--engine',
        'kubeflow',
        '--pipeline_path',
        'kubeflow_runner.py',
        '--endpoint',
        self._endpoint,
        '--build-image',
    ])

  def _run_pipeline(self):
    result = self._runCli([
        'run',
        'create',
        '--engine',
        'kubeflow',
        '--pipeline_name',
        self._pipeline_name,
        '--endpoint',
        self._endpoint,
    ])
    run_id = self._parse_run_id(result)
    self._wait_until_completed(run_id)
    kubeflow_test_utils.print_failure_log_for_run(self._endpoint, run_id,
                                                  self._namespace)

  def _check_telemetry_label(self):
    file_path = os.path.join(self._project_dir,
                             '{}.tar.gz'.format(self._pipeline_name))
    self.assertTrue(fileio.exists(file_path))

    with tarfile.TarFile.open(file_path).extractfile(
        'pipeline.yaml') as pipeline_file:
      self.assertIsNotNone(pipeline_file)
      pipeline = yaml.safe_load(pipeline_file)
      metadata = [
          c['metadata'] for c in pipeline['spec']['templates'] if 'dag' not in c
      ]
      for m in metadata:
        self.assertEqual('tfx-template',
                         m['labels'][telemetry_utils.LABEL_KFP_SDK_ENV])


class BaseVertexEndToEndTest(BaseContainerBasedEndToEndTest):
  """Common utilities for vertex engine."""

  def setUp(self):
    super().setUp()
    self.enter_context(
        test_case_utils.override_env_var('VERTEX_HOME',
                                         os.path.join(self._temp_dir,
                                                      'vertex')))

  def _parse_run_id(self, output: str):
    run_id_lines = [
        line for line in output.split('\n') if line.startswith('| ')
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
