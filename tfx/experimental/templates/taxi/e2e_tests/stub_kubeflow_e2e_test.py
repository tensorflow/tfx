# Lint as: python3
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
"""E2E test using kubeflow orchestrator with stub executors for taxi template."""

import datetime
import os
import subprocess
import tarfile
import urllib.request
import time

from absl import logging
import kfp
import tensorflow as tf
from typing import *
from tfx.experimental.pipeline_testing import pipeline_recorder_utils
from tfx.experimental.templates.taxi.e2e_tests import test_utils
from tfx.orchestration import test_utils as orchestration_test_utils
from tfx.orchestration.kubeflow import test_utils as kubeflow_test_utils
from tfx.utils import telemetry_utils
from tfx.utils import io_utils
import yaml
from google.cloud import storage

# os.environ['KFP_E2E_BASE_CONTAINER_IMAGE'] = 'gcr.io/tfx-intern-sujip-gke/tensorflow:latest'
# os.environ['KFP_E2E_SRC'] = '/home/jupyter/tfx'
# os.environ['KFP_E2E_GCP_PROJECT_ID']= 'tfx-intern-sujip-gke'
#os.environ['KFP_E2E_GCP_REGION'] = '
# export KFP_E2E_GCP_REGION='us-central1-a'
# export KFP_E2E_BUCKET_NAME='tfx-intern-sujip-gke-kubeflowpipelines-default'
# export KFP_E2E_BASE_CONTAINER_IMAGE='gcr.io/tfx-intern-sujip-gke/tensorflow'
# export KFP_E2E_SRC='/home/jupyter/tfx'
# export KFP_E2E_GCP_PROJECT_ID='tfx-intern-sujip-gke'
# export KFP_E2E_TEST_DATA_ROOT='gs://tfx-intern-sujip-gke-kubeflowpipelines-default/tfx-template/data'
# export PYTHONPATH=/home/jupyter/tfx
# export PATH=/home/jupyter/.local/bin:$PATH

class StubKubeflowE2ETest(test_utils.BaseEndToEndTest):

  _POLLING_INTERVAL_IN_SECONDS = 10
  _MAX_POLLING_COUNT = 20 * 6  # 20 min.

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
  #_GCP_REGION = os.environ['KFP_E2E_GCP_REGION']

  # The GCP bucket to use to write output artifacts.
  # This default bucket name is valid for KFP marketplace deployment since KFP
  # version 0.5.0.
  _BUCKET_NAME = _GCP_PROJECT_ID + '-kubeflowpipelines-default'

  _KFP_E2E_TEST_FORWARDING_PORT_BEGIN = 17330
  _KFP_E2E_TEST_FORWARDING_PORT_END = 17335
  _MAX_ATTEMPTS = 5

  def setUp(self):
    super().setUp()
    random_id = orchestration_test_utils.random_id()
    self._pipeline_name = 'taxi-template-kubeflow-e2e-test-' + random_id
    logging.info('Pipeline: %s', self._pipeline_name)
    self._endpoint = self._get_endpoint()
    self._kfp_client = kfp.Client(host=self._endpoint)
    logging.info('ENDPOINT: %s', self._endpoint)

    self._base_container_image = '{}:{}'.format(self._BASE_CONTAINER_IMAGE,
                                                random_id)
    self._target_container_image = 'gcr.io/{}/{}:{}'.format(
        self._GCP_PROJECT_ID, 'taxi-template-kubeflow-e2e-test', random_id)
    self._record_dir = "gs://{}/recorded_testdata".format(self._BUCKET_NAME)
    self._port_forwarding_process = self._setup_mlmd_port_forward()
    self._prepare_base_container_image()
    self._prepare_skaffold()

  def tearDown(self):
    super(StubKubeflowE2ETest, self).tearDown()
    logging.info('Killing the GRPC port-forwarding process.')
    self._port_forwarding_process.kill()
    io_utils.delete_dir(self._record_dir)
    self._cleanup_kfp()

  def _get_grpc_port(self) -> Text:
    """Get the port number used by MLMD gRPC server."""
    get_grpc_port_command = [
        'kubectl', '-n', 'default', 'get', 'configmap',
        'metadata-grpc-configmap', '-o',
        'jsonpath={.data.METADATA_GRPC_SERVICE_PORT}'
    ]

    grpc_port = subprocess.check_output(get_grpc_port_command)
    return grpc_port.decode('utf-8')

  def _setup_mlmd_port_forward(self) -> subprocess.Popen:
    """Uses port forward to talk to MLMD gRPC server."""
    grpc_port = self._get_grpc_port()

    is_bind = False

    for port in range(self._KFP_E2E_TEST_FORWARDING_PORT_BEGIN,self._KFP_E2E_TEST_FORWARDING_PORT_END):
      print("port", port)
      grpc_forward_command = [
          'kubectl', 'port-forward', 'deployment/metadata-grpc-deployment',
          '-n', 'default', ('%s:%s' % (port, grpc_port))
      ]
      # Begin port forwarding.
      proc = subprocess.Popen(grpc_forward_command)
      try:
        # Wait while port forward to pod is being established
        poll_grpc_port_command = ['lsof', '-i', ':%s' % port]
        result = subprocess.run(  # pylint: disable=subprocess-run-check
            poll_grpc_port_command,
            stdout=subprocess.PIPE)
        for _ in range(self._MAX_ATTEMPTS):
          if (result.returncode == 0 and
              'kubectl' in result.stdout.decode('utf-8')):
            is_bind = True
            break
          logging.info(
              'Waiting while gRPC port-forward is being established...')
          time.sleep(5)
          result = subprocess.run(  # pylint: disable=subprocess-run-check
              poll_grpc_port_command,
              stdout=subprocess.PIPE)

      except:  # pylint: disable=bare-except
        # Kill the process in case unexpected error occurred.
        proc.kill()
      
      if is_bind:
        self._port = port
        break

    if not is_bind:
      raise RuntimeError('Failed to establish gRPC port-forward to cluster in '
                         'the specified range: port %s to %s' %
                         (self._KFP_E2E_TEST_FORWARDING_PORT_BEGIN,
                          self._KFP_E2E_TEST_FORWARDING_PORT_END))

    return proc

  def _cleanup_with_retry(self, method):
    max_num_trial = 3
    for _ in range(max_num_trial):
      try:
        method()
      except Exception as err:  # pylint:disable=broad-except
        logging.info(err)
      else:
        break

  def _cleanup_kfp(self):
    self._cleanup_with_retry(self._delete_base_container_image)
    self._cleanup_with_retry(self._delete_target_container_image)
    self._cleanup_with_retry(self._delete_pipeline)
    self._cleanup_with_retry(self._delete_pipeline_data)
    self._cleanup_with_retry(self._delete_runs)

  def _get_kfp_runs(self):
    # CLI uses experiment_name which is the same as pipeline_name.
    experiment_id = self._kfp_client.get_experiment(
        experiment_name=self._pipeline_name).id
    response = self._kfp_client.list_runs(experiment_id=experiment_id)
    return response.runs

  def _delete_runs(self):
    for run in self._get_kfp_runs():
      self._kfp_client._run_api.delete_run(id=run.id)  # pylint: disable=protected-access

  def _delete_pipeline(self):
    self._runCli([
        'pipeline', 'delete', '--engine', 'kubeflow', '--pipeline_name',
        self._pipeline_name
    ])

  def _delete_pipeline_data(self):
    path = 'tfx_pipeline_output/{}'.format(self._pipeline_name)
    orchestration_test_utils.delete_gcs_files(self._GCP_PROJECT_ID,
                                              self._BUCKET_NAME, path)
    path = '{}/{}'.format(self._DATA_DIRECTORY_NAME, self._pipeline_name)
    orchestration_test_utils.delete_gcs_files(self._GCP_PROJECT_ID,
                                              self._BUCKET_NAME, path)

  def _delete_base_container_image(self):
    subprocess.check_output([
        'gcloud', 'container', 'images', 'delete', self._base_container_image
    ])

  def _delete_target_container_image(self):
    subprocess.check_output([
        'gcloud', 'container', 'images', 'delete', self._target_container_image
    ])

  def _get_endpoint(self):  # pylint: disable=inconsistent-return-statements
    output = subprocess.check_output(
        'kubectl describe configmap inverse-proxy-config -n default'.split())
    for line in output.decode('utf-8').split('\n'):
      if line.endswith('googleusercontent.com'):
        return line

  def _prepare_data(self):
    client = storage.Client(project=self._GCP_PROJECT_ID)
    bucket = client.bucket(self._BUCKET_NAME)
    blob = bucket.blob('{}/{}/data.csv'.format(self._DATA_DIRECTORY_NAME,
                                               self._pipeline_name))
    blob.upload_from_filename('data/data.csv')

  def _prepare_base_container_image(self):
    orchestration_test_utils.build_and_push_docker_image(
        self._base_container_image, self._REPO_BASE)

  def _prepare_skaffold(self):
    self._skaffold = os.path.join(self._temp_dir, 'skaffold')
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/skaffold/releases/latest/skaffold-linux-amd64',
        self._skaffold)
    os.chmod(self._skaffold, 0o775)

  def _create_pipeline(self):
    result = self._runCli([
        'pipeline',
        'create',
        '--engine',
        'kubeflow',
        '--pipeline_path',
        'kubeflow_dag_runner.py',
        '--endpoint',
        self._endpoint,
        '--build-target-image',
        self._target_container_image,
        '--skaffold-cmd',
        self._skaffold,
        '--build-base-image',
        self._base_container_image,
    ])
    self.assertEqual(0, result.exit_code)

  def _update_pipeline(self):
    result = self._runCli([
        'pipeline',
        'update',
        '--engine',
        'kubeflow',
        '--pipeline_path',
        'kubeflow_dag_runner.py',
        '--endpoint',
        self._endpoint,
        '--skaffold-cmd',
        self._skaffold,
    ])
    self.assertEqual(0, result.exit_code)

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
    self.assertEqual(0, result.exit_code)
    self._wait_until_completed(self._parse_run_id(result.output))

  def _parse_run_id(self, output: str):
    run_id_lines = [
        line for line in output.split('\n')
        if '| {} |'.format(self._pipeline_name) in line
    ]
    self.assertLen(run_id_lines, 1)
    return run_id_lines[0].split('|')[2].strip()

  def _wait_until_completed(self, run_id: str):
    # This timeout will never expire. polling_count * interval == 20min.
    timeout = datetime.timedelta(hours=1)
    end_state = kubeflow_test_utils.poll_kfp_with_retry(
        self._endpoint, run_id, self._MAX_POLLING_COUNT, timeout,
        self._POLLING_INTERVAL_IN_SECONDS)
    self.assertEqual(end_state.lower(), kubeflow_test_utils.KFP_SUCCESS_STATUS)

  def _check_telemetry_label(self):
    file_path = os.path.join(self._project_dir,
                             '{}.tar.gz'.format(self._pipeline_name))
    self.assertTrue(tf.io.gfile.exists(file_path))

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

  def testPipeline(self):
    print("port:", self._port)
    self._copyTemplate()
    os.environ['KUBEFLOW_HOME'] = os.path.join(self._temp_dir, 'kubeflow')

    # Uncomment all variables in config.
    self._uncommentMultiLineVariables(
        os.path.join('pipeline', 'configs.py'), [
            'GOOGLE_CLOUD_REGION',
            'BIG_QUERY_WITH_DIRECT_RUNNER_BEAM_PIPELINE_ARGS',
            'BIG_QUERY_QUERY', 'DATAFLOW_BEAM_PIPELINE_ARGS',
            'GCP_AI_PLATFORM_TRAINING_ARGS', 'GCP_AI_PLATFORM_SERVING_ARGS'
        ])

    # Prepare data
    self._prepare_data()
    self._replaceFileContent('kubeflow_dag_runner.py', [
        ('DATA_PATH = \'gs://{}/tfx-template/data/\'.format(configs.GCS_BUCKET_NAME)',  # pylint: disable=line-too-long
         'DATA_PATH = \'gs://{{}}/{}/{}\'.format(configs.GCS_BUCKET_NAME)'
         .format(self._DATA_DIRECTORY_NAME, self._pipeline_name)),
    ])

    # Create a pipeline with only one component.
    updated_pipeline_file = self._addAllComponents()
    logging.info('Updated %s to add all components to the pipeline.',
                 updated_pipeline_file)

    self._create_pipeline()
    self._run_pipeline()
#     self._check_telemetry_label()
    logging.info("Successfully ran the pipeline.")
    pipeline_recorder_utils.record_pipeline(
        output_dir=self._record_dir,
        metadata_db_uri=None,
        host='localhost',
        port=self._port,
        pipeline_name=self._pipeline_name,
        run_id=None)
    logging.info("Pipeline has been recorded.")

    # Enable stub executors
    self._uncomment('kubeflow_dag_runner.py',
                    ['supported_launcher_classes=[',
                     '    stub_component_launcher.StubComponentLauncher',
                     '],'])
    # Update the pipeline to use stub executors.

#     self._update_pipeline()
    self._create_pipeline()
    self._run_pipeline()


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.test.main()