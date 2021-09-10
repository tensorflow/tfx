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
"""E2E Kubeflow tests for CLI."""

import codecs
import locale
import os
import subprocess
import sys
from typing import Optional

import absl
from google.cloud import storage
import kfp
import kfp_server_api
import tensorflow as tf
from tfx.dsl.io import fileio
from tfx.tools.cli import labels
from tfx.tools.cli import pip_utils
from tfx.tools.cli.e2e import test_utils
from tfx.utils import retry
from tfx.utils import test_case_utils


class CliKubeflowEndToEndTest(test_case_utils.TfxTest):

  def _get_endpoint(self, config: str) -> str:
    lines = config.decode('utf-8').split('\n')
    for line in lines:
      if line.endswith('googleusercontent.com'):
        return line

  def setUp(self):
    super().setUp()

    # List of packages installed.
    self._pip_list = pip_utils.get_package_names()

    # Check if Kubeflow is installed before running E2E tests.
    if labels.KUBEFLOW_PACKAGE_NAME not in self._pip_list:
      sys.exit('Kubeflow not installed.')

    # Change the encoding for Click since Python 3 is configured to use ASCII as
    # encoding for the environment.
    if codecs.lookup(locale.getpreferredencoding()).name == 'ascii':
      os.environ['LANG'] = 'en_US.utf-8'

    # Testdata path.
    self._testdata_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'testdata')
    self._testdata_dir_updated = self.tmp_dir
    fileio.makedirs(self._testdata_dir_updated)

    self.enter_context(test_case_utils.change_working_dir(self.tmp_dir))

    # Generate a unique pipeline name. Uses tmp_dir as a random seed.
    self._pipeline_name = ('cli-kubeflow-e2e-test-' +
                           test_utils.generate_random_id(self.tmp_dir))
    absl.logging.info('Pipeline name is %s' % self._pipeline_name)
    self._pipeline_name_v2 = self._pipeline_name + '_v2'

    orig_pipeline_path = os.path.join(self._testdata_dir,
                                      'test_pipeline_kubeflow_1.py')
    self._pipeline_path = os.path.join(self._testdata_dir_updated,
                                       'test_pipeline_kubeflow_1.py')
    self._pipeline_path_v2 = os.path.join(self._testdata_dir_updated,
                                          'test_pipeline_kubeflow_2.py')

    test_utils.copy_and_change_pipeline_name(orig_pipeline_path,
                                             self._pipeline_path,
                                             'chicago_taxi_pipeline_kubeflow',
                                             self._pipeline_name)
    self.assertTrue(fileio.exists(self._pipeline_path))
    test_utils.copy_and_change_pipeline_name(orig_pipeline_path,
                                             self._pipeline_path_v2,
                                             'chicago_taxi_pipeline_kubeflow',
                                             self._pipeline_name_v2)
    self.assertTrue(fileio.exists(self._pipeline_path_v2))

    # Endpoint URL
    self._endpoint = self._get_endpoint(
        subprocess.check_output(
            'kubectl describe configmap inverse-proxy-config -n kubeflow'.split(
            )))
    absl.logging.info('ENDPOINT: ' + self._endpoint)

    try:
      # Create a kfp client for cleanup after running commands.
      self._client = kfp.Client(host=self._endpoint)
    except kfp_server_api.rest.ApiException as err:
      absl.logging.info(err)

  def tearDown(self):
    super().tearDown()
    self._cleanup_kfp_server(self._pipeline_name)

  def _cleanup_kfp_server(self, pipeline_name):
    self._delete_experiment(pipeline_name)
    self._delete_pipeline(pipeline_name)
    self._delete_pipeline_output(pipeline_name)

  @retry.retry(ignore_eventual_failure=True)
  def _delete_pipeline(self, pipeline_name: str):
    pipeline_id = self._get_kfp_pipeline_id(pipeline_name)
    if pipeline_id is not None:
      self._client.delete_pipeline(pipeline_id)
      absl.logging.info('Deleted pipeline : {}'.format(pipeline_name))

  @retry.retry(ignore_eventual_failure=True)
  def _delete_experiment(self, pipeline_name: str):
    experiment_id = self._get_kfp_experiment_id(pipeline_name)
    if experiment_id is not None:
      self._delete_all_runs(experiment_id)
      self._client._experiment_api.delete_experiment(experiment_id)
      absl.logging.info('Deleted experiment : {}'.format(pipeline_name))

  def _get_kfp_pipeline_id(self, pipeline_name: str) -> Optional[str]:
    return self._client.get_pipeline_id(pipeline_name)

  def _get_kfp_experiment_id(self, pipeline_name: str) -> Optional[str]:
    try:
      experiment = self._client.get_experiment(experiment_name=pipeline_name)
    except ValueError:
      return None
    return experiment.id

  @retry.retry(ignore_eventual_failure=True)
  def _delete_pipeline_output(self, pipeline_name: str) -> None:
    """Deletes output produced by the named pipeline.

    Args:
      pipeline_name: The name of the pipeline.
    """
    if ('KFP_E2E_GCP_PROJECT_ID' not in os.environ) or ('KFP_E2E_BUCKET_NAME'
                                                        not in os.environ):
      return
    client = storage.Client(project=os.environ['KFP_E2E_GCP_PROJECT_ID'])
    bucket = client.get_bucket(os.environ['KFP_E2E_BUCKET_NAME'])
    prefix = 'test_output/{}'.format(pipeline_name)
    absl.logging.info(
        'Deleting output under GCS bucket prefix: {}'.format(prefix))
    blobs = list(bucket.list_blobs(prefix=prefix))
    bucket.delete_blobs(blobs)

  def _delete_all_runs(self, experiment_id: str):
    try:
      # Get all runs related to the experiment_id.
      response = self._client.list_runs(experiment_id=experiment_id)
      if response and response.runs:
        for run in response.runs:
          self._client._run_api.delete_run(id=run.id)
    except kfp_server_api.rest.ApiException as err:
      absl.logging.info(err)

  def _valid_create_and_check(self, pipeline_path: str,
                              pipeline_name: str) -> None:
    result = test_utils.run_cli([
        'pipeline', 'create', '--engine', 'kubeflow', '--pipeline_path',
        pipeline_path, '--endpoint', self._endpoint
    ])
    absl.logging.info('[CLI] %s', result)
    self.assertIn('Creating pipeline', result)
    self.assertIn('Pipeline "{}" created successfully.'.format(pipeline_name),
                  result)

  def _run_pipeline_using_kfp_client(self, pipeline_name: str):
    pipeline_id = self._get_kfp_pipeline_id(pipeline_name)
    experiment_id = self._get_kfp_experiment_id(pipeline_name)
    absl.logging.info(
        'Creating a run directly using kfp API for "%s"(pipeline_id=%s,'
        ' experiment_id=%s)', pipeline_name, pipeline_id, experiment_id)
    return self._client.run_pipeline(
        experiment_id=experiment_id,
        job_name=pipeline_name,
        pipeline_id=pipeline_id)

  def testPipelineCreate(self):
    # Create a pipeline.
    self._valid_create_and_check(self._pipeline_path, self._pipeline_name)

    # Test pipeline create when pipeline already exists.
    with self.assertRaises(subprocess.CalledProcessError) as cm:
      test_utils.run_cli([
          'pipeline', 'create', '--engine', 'kubeflow', '--pipeline_path',
          self._pipeline_path, '--endpoint', self._endpoint
      ])
    self.assertIn('Pipeline "{}" already exists.'.format(self._pipeline_name),
                  cm.exception.output)

  def testPipelineUpdate(self):
    # Try pipeline update when pipeline does not exist.
    with self.assertRaises(subprocess.CalledProcessError) as cm:
      test_utils.run_cli([
          'pipeline', 'update', '--engine', 'kubeflow', '--pipeline_path',
          self._pipeline_path, '--endpoint', self._endpoint
      ])
    self.assertIn('Cannot find pipeline "{}".'.format(self._pipeline_name),
                  cm.exception.output)

    # Now update an existing pipeline.
    self._valid_create_and_check(self._pipeline_path, self._pipeline_name)

    result = test_utils.run_cli([
        'pipeline', 'update', '--engine', 'kubeflow', '--pipeline_path',
        self._pipeline_path, '--endpoint', self._endpoint
    ])
    self.assertIn('Updating pipeline', result)
    self.assertIn(
        'Pipeline "{}" updated successfully.'.format(self._pipeline_name),
        result)

  def testPipelineCompile(self):
    # Invalid DSL path
    pipeline_path = os.path.join(self._testdata_dir, 'test_pipeline_flink.py')
    with self.assertRaises(subprocess.CalledProcessError) as cm:
      test_utils.run_cli([
          'pipeline', 'compile', '--engine', 'kubeflow', '--pipeline_path',
          pipeline_path
      ])
    self.assertIn('Invalid pipeline path: {}'.format(pipeline_path),
                  cm.exception.output)

    # Wrong Runner.
    pipeline_path = os.path.join(self._testdata_dir,
                                 'test_pipeline_airflow_1.py')
    with self.assertRaises(subprocess.CalledProcessError) as cm:
      test_utils.run_cli([
          'pipeline', 'compile', '--engine', 'kubeflow', '--pipeline_path',
          pipeline_path
      ])

    # Successful compilation.
    result = test_utils.run_cli([
        'pipeline', 'compile', '--engine', 'kubeflow', '--pipeline_path',
        self._pipeline_path
    ])
    absl.logging.info('pipeline compile result:%s', result)
    self.assertIn('Compiling pipeline', result)
    self.assertIn('Pipeline compiled successfully', result)

  def testPipelineDelete(self):
    # Try deleting a non existent pipeline.
    with self.assertRaises(subprocess.CalledProcessError) as cm:
      test_utils.run_cli([
          'pipeline', 'delete', '--engine', 'kubeflow', '--pipeline_name',
          self._pipeline_name, '--endpoint', self._endpoint
      ])
    self.assertIn('Cannot find pipeline "{}".'.format(self._pipeline_name),
                  cm.exception.output)

    # Create a pipeline.
    self._valid_create_and_check(self._pipeline_path, self._pipeline_name)

    # Now delete the pipeline.
    result = test_utils.run_cli([
        'pipeline', 'delete', '--engine', 'kubeflow', '--pipeline_name',
        self._pipeline_name, '--endpoint', self._endpoint
    ])
    self.assertIn('Deleting pipeline', result)
    self.assertIn(
        'Pipeline {} deleted successfully.'.format(self._pipeline_name), result)

  def testPipelineList(self):
    # Create pipelines.
    self._valid_create_and_check(self._pipeline_path, self._pipeline_name)
    self._valid_create_and_check(self._pipeline_path_v2, self._pipeline_name_v2)
    self.addCleanup(self._cleanup_kfp_server, self._pipeline_name_v2)

    # List pipelines.
    result = test_utils.run_cli([
        'pipeline', 'list', '--engine', 'kubeflow', '--endpoint', self._endpoint
    ])
    self.assertIn('Listing all pipelines', result)
    self.assertIn(self._pipeline_name, result)
    self.assertIn(self._pipeline_name_v2, result)

  def testPipelineCreateAutoDetectFail(self):
    if labels.AIRFLOW_PACKAGE_NAME not in self._pip_list:
      self.skipTest(
          'Airflow doesn\'t exist. Airflow needed to make auto detect fail.')
    with self.assertRaises(subprocess.CalledProcessError) as cm:
      test_utils.run_cli([
          'pipeline', 'create', '--engine', 'auto', '--pipeline_path',
          self._pipeline_path, '--endpoint', self._endpoint
      ])
    self.assertIn(
        'Multiple orchestrators found. Choose one using --engine flag.',
        cm.exception.output)

  def testPipelineCreateAutoDetectSuccess(self):
    if labels.AIRFLOW_PACKAGE_NAME in self._pip_list:
      self.skipTest(
          'Airflow exists. Airflow should not exist to make auto detect fail.')
    result = test_utils.run_cli([
        'pipeline', 'create', '--engine', 'auto', '--pipeline_path',
        self._pipeline_path, '--endpoint', self._endpoint
    ])
    self.assertIn(
        'Pipeline "{}" created successfully.'.format(self._pipeline_name),
        result)

  def testRunCreate(self):
    with self.assertRaises(subprocess.CalledProcessError) as cm:
      test_utils.run_cli([
          'run', 'create', '--engine', 'kubeflow', '--pipeline_name',
          self._pipeline_name, '--endpoint', self._endpoint
      ])
    self.assertIn('Cannot find pipeline "{}".'.format(self._pipeline_name),
                  cm.exception.output)

    # Now create a pipeline.
    self._valid_create_and_check(self._pipeline_path, self._pipeline_name)

    # Run pipeline.
    result = test_utils.run_cli([
        'run', 'create', '--engine', 'kubeflow', '--pipeline_name',
        self._pipeline_name, '--endpoint', self._endpoint
    ])

    self.assertIn('Creating a run for pipeline: {}'.format(self._pipeline_name),
                  result)
    self.assertNotIn(
        'Pipeline "{}" does not exist.'.format(self._pipeline_name), result)
    self.assertIn('Run created for pipeline: {}'.format(self._pipeline_name),
                  result)

  def testRunDelete(self):
    # Now create a pipeline.
    self._valid_create_and_check(self._pipeline_path, self._pipeline_name)

    # Run pipeline using kfp client to get run_id.
    run = self._run_pipeline_using_kfp_client(self._pipeline_name)

    absl.logging.info('Deleting run: %s', run.id)
    result = test_utils.run_cli([
        'run', 'delete', '--engine', 'kubeflow', '--endpoint', self._endpoint,
        '--run_id', run.id
    ])
    self.assertIn('Deleting run.', result)
    self.assertIn('Run deleted.', result)

  def testRunTerminate(self):
    # Now create a pipeline.
    self._valid_create_and_check(self._pipeline_path, self._pipeline_name)

    # Run pipeline using kfp client to get run_id.
    run = self._run_pipeline_using_kfp_client(self._pipeline_name)

    absl.logging.info('Terminating run: %s', run.id)
    result = test_utils.run_cli([
        'run', 'terminate', '--engine', 'kubeflow', '--endpoint',
        self._endpoint, '--run_id', run.id
    ])
    self.assertIn('Terminating run.', result)
    self.assertIn('Run terminated.', result)

  def testRunStatus(self):
    # Now create a pipeline.
    self._valid_create_and_check(self._pipeline_path, self._pipeline_name)

    # Run pipeline using kfp client to get run_id.
    run = self._run_pipeline_using_kfp_client(self._pipeline_name)

    absl.logging.info('Retrieving run status: %s(%s)', run.id,
                      self._pipeline_name)
    result = test_utils.run_cli([
        'run', 'status', '--engine', 'kubeflow', '--pipeline_name',
        self._pipeline_name, '--endpoint', self._endpoint, '--run_id', run.id
    ])
    self.assertIn('Retrieving run status.', result)
    self.assertIn(str(run.id), result)
    self.assertIn(self._pipeline_name, result)

  def testRunList(self):
    # Now create a pipeline.
    self._valid_create_and_check(self._pipeline_path, self._pipeline_name)

    # Run pipeline using kfp client to get run_id.
    run_1 = self._run_pipeline_using_kfp_client(self._pipeline_name)
    run_2 = self._run_pipeline_using_kfp_client(self._pipeline_name)

    # List runs.
    result = test_utils.run_cli([
        'run', 'list', '--engine', 'kubeflow', '--pipeline_name',
        self._pipeline_name, '--endpoint', self._endpoint
    ])
    self.assertIn(
        'Listing all runs of pipeline: {}'.format(self._pipeline_name), result)
    self.assertIn(str(run_1.id), result)
    self.assertIn(str(run_2.id), result)
    self.assertIn(self._pipeline_name, result)


if __name__ == '__main__':
  absl.logging.set_verbosity(absl.logging.INFO)
  tf.test.main()
