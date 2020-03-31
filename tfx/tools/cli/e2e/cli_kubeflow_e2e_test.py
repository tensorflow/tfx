# Lint as: python2, python3
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import datetime
import json
import locale
import logging
import os
import random
import shutil
import string
import subprocess
import sys
import tempfile
from typing import Text, Tuple

import absl
from click import testing as click_testing
import kfp
import kfp_server_api
import tensorflow as tf

from google.cloud import storage
from tensorflow.python.lib.io import file_io  # pylint: disable=g-direct-tensorflow-import
from tfx.tools.cli import labels
from tfx.tools.cli.cli_main import cli_group


class CliKubeflowEndToEndTest(tf.test.TestCase):

  def _get_endpoint(self, config: Text) -> Text:
    lines = config.decode('utf-8').split('\n')
    for line in lines:
      if line.endswith('googleusercontent.com'):
        return line

  def setUp(self):
    super(CliKubeflowEndToEndTest, self).setUp()
    random.seed(datetime.datetime.now())

    # List of packages installed.
    self._pip_list = str(subprocess.check_output(['pip', 'freeze', '--local']))

    # Check if Kubeflow is installed before running E2E tests.
    if labels.KUBEFLOW_PACKAGE_NAME not in self._pip_list:
      sys.exit('Kubeflow not installed.')

    # Change the encoding for Click since Python 3 is configured to use ASCII as
    # encoding for the environment.
    if codecs.lookup(locale.getpreferredencoding()).name == 'ascii':
      os.environ['LANG'] = 'en_US.utf-8'

    # Initialize CLI runner.
    self.runner = click_testing.CliRunner()

    # Testdata path.
    self._testdata_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'testdata')
    self._testdata_dir_updated = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    tf.io.gfile.makedirs(self._testdata_dir_updated)

    self._pipeline_name = 'cli-kubeflow-e2e-test-%s-%s' % (
        datetime.datetime.now().strftime('%s'), ''.join([
            random.choice(string.ascii_lowercase + string.digits)
            for _ in range(10)
        ]))
    absl.logging.info('Pipeline name is %s' % self._pipeline_name)
    self._pipeline_name_v2 = self._pipeline_name + '_v2'

    orig_pipeline_path = os.path.join(self._testdata_dir,
                                      'test_pipeline_kubeflow_1.py')
    self._pipeline_path = os.path.join(self._testdata_dir_updated,
                                       'test_pipeline_kubeflow_1.py')
    self._pipeline_path_v2 = os.path.join(self._testdata_dir_updated,
                                          'test_pipeline_kubeflow_2.py')

    self._change_pipeline_name(orig_pipeline_path, self._pipeline_path,
                               'chicago_taxi_pipeline_kubeflow',
                               self._pipeline_name)
    self.assertTrue(tf.io.gfile.exists(self._pipeline_path))
    self._change_pipeline_name(orig_pipeline_path, self._pipeline_path_v2,
                               'chicago_taxi_pipeline_kubeflow',
                               self._pipeline_name_v2)
    self.assertTrue(tf.io.gfile.exists(self._pipeline_path_v2))

    # Endpoint URL
    self._endpoint = self._get_endpoint(
        subprocess.check_output(
            'kubectl describe configmap inverse-proxy-config -n kubeflow'.split(
                )))
    absl.logging.info('ENDPOINT: ' + self._endpoint)

    # Change home directories
    self._olddir = os.getcwd()
    self._old_kubeflow_home = os.environ.get('KUBEFLOW_HOME')
    os.environ['KUBEFLOW_HOME'] = os.path.join(tempfile.mkdtemp(),
                                               'CLI_Kubeflow_Pipelines')
    self._kubeflow_home = os.environ['KUBEFLOW_HOME']
    tf.io.gfile.makedirs(self._kubeflow_home)
    os.chdir(self._kubeflow_home)

    self._handler_pipeline_path = os.path.join(self._kubeflow_home,
                                               self._pipeline_name)
    self._handler_pipeline_args_path = os.path.join(self._handler_pipeline_path,
                                                    'pipeline_args.json')
    self._pipeline_package_path = '{}.tar.gz'.format(self._pipeline_name)

    try:
      # Create a kfp client for cleanup after running commands.
      self._client = kfp.Client(host=self._endpoint)
    except kfp_server_api.rest.ApiException as err:
      absl.logging.info(err)

  def tearDown(self):
    super(CliKubeflowEndToEndTest, self).tearDown()
    self._cleanup_kfp_server()
    if self._old_kubeflow_home:
      os.environ['KUBEFLOW_HOME'] = self._old_kubeflow_home
    os.chdir(self._olddir)
    shutil.rmtree(self._kubeflow_home)
    absl.logging.info('Deleted all runs.')

  def _change_pipeline_name(self, orig_path: Text, new_path: Text,
                            origin_pipeline_name: Text,
                            new_pipeline_name: Text) -> None:
    """Copy pipeline file to new path with pipeline name changed."""
    contents = file_io.read_file_to_string(orig_path)
    assert contents.count(origin_pipeline_name
                         ) == 1, 'DSL file can only contain one pipeline name'
    contents = contents.replace(origin_pipeline_name, new_pipeline_name)
    file_io.write_string_to_file(new_path, contents)

  def _cleanup_kfp_server(self):
    pipelines = tf.io.gfile.listdir(self._kubeflow_home)
    for pipeline_name in pipelines:
      if tf.io.gfile.isdir(pipeline_name):
        self._delete_experiment(pipeline_name)
        self._delete_pipeline(pipeline_name)
        self._delete_pipeline_output(pipeline_name)
        self._delete_pipeline_metadata(pipeline_name)

  def _delete_pipeline(self, pipeline_name: Text):
    pipeline_id, _ = self._get_pipeline_id_and_experiment_id(pipeline_name)
    if self._client._pipelines_api.get_pipeline(pipeline_id):
      self._client._pipelines_api.delete_pipeline(id=pipeline_id)
      absl.logging.info('Deleted pipeline : {}'.format(pipeline_name))

  def _delete_experiment(self, pipeline_name: Text):
    if self._client.get_experiment(experiment_name=pipeline_name):
      experiment_id = self._client.get_experiment(
          experiment_name=pipeline_name).id
      self._delete_all_runs(experiment_id)
      self._client._experiment_api.delete_experiment(experiment_id)
      absl.logging.info('Deleted experiment : {}'.format(pipeline_name))

  def _get_pipeline_id_and_experiment_id(
      self, pipeline_name: Text) -> Tuple[Text, Text]:
    # Path to pipeline_args.json .
    pipeline_args_path = os.path.join(self._kubeflow_home, pipeline_name,
                                      'pipeline_args.json')
    # Get pipeline_id and experiment_id from pipeline_args.json
    with open(pipeline_args_path, 'r') as f:
      pipeline_args = json.load(f)
    pipeline_id = pipeline_args[labels.PIPELINE_ID]
    experiment_id = pipeline_args[labels.EXPERIMENT_ID]
    return pipeline_id, experiment_id

  def _delete_pipeline_output(self, pipeline_name: Text) -> None:
    """Deletes output produced by the named pipeline.

    Args:
      pipeline_name: The name of the pipeline.
    """
    gcp_project_id = 'tfx-oss-testing'
    bucket_name = 'tfx-oss-testing-bucket'
    client = storage.Client(project=gcp_project_id)
    bucket = client.get_bucket(bucket_name)
    prefix = 'test_output/{}'.format(pipeline_name)
    absl.logging.info(
        'Deleting output under GCS bucket prefix: {}'.format(prefix))
    blobs = bucket.list_blobs(prefix=prefix)
    bucket.delete_blobs(blobs)

  def _get_mysql_pod_name(self) -> Text:
    """Returns MySQL pod name in the cluster."""
    pod_name = subprocess.check_output([
        'kubectl',
        '-n',
        'kubeflow',
        'get',
        'pods',
        '-l',
        'app=mysql',
        '--no-headers',
        '-o',
        'custom-columns=:metadata.name',
    ]).decode('utf-8').strip('\n')
    absl.logging.info('MySQL pod name is: {}'.format(pod_name))
    return pod_name

  def _delete_pipeline_metadata(self, pipeline_name: Text) -> None:
    """Drops the database containing metadata produced by the pipeline.

    Args:
      pipeline_name: The name of the pipeline owning the database.
    """
    pod_name = self._get_mysql_pod_name()
    valid_mysql_name = pipeline_name.replace('-', '_')
    # MySQL database name cannot exceed 64 characters.
    db_name = 'mlmd_{}'.format(valid_mysql_name[-59:])

    command = [
        'kubectl',
        '-n',
        'kubeflow',
        'exec',
        '-it',
        pod_name,
        '--',
        'mysql',
        '--user',
        'root',
        '--execute',
        'drop database if exists {};'.format(db_name),
    ]
    absl.logging.info('Dropping MLMD DB with name: {}'.format(db_name))
    subprocess.run(command, check=True)

  def _delete_all_runs(self, experiment_id: Text):
    try:
      # Get all runs related to the experiment_id.
      response = self._client.list_runs(experiment_id=experiment_id)
      if response and response.runs:
        for run in response.runs:
          self._client._run_api.delete_run(id=run.id)
    except kfp_server_api.rest.ApiException as err:
      absl.logging.info(err)

  def _valid_create_and_check(self, pipeline_path: Text,
                              pipeline_name: Text) -> None:
    result = self.runner.invoke(cli_group, [
        'pipeline', 'create', '--engine', 'kubeflow', '--pipeline_path',
        pipeline_path, '--endpoint', self._endpoint
    ])
    self.assertIn('Creating pipeline', result.output)
    self.assertTrue(tf.io.gfile.exists(self._pipeline_package_path))
    self.assertTrue(tf.io.gfile.exists(self._handler_pipeline_args_path))
    self.assertIn('Pipeline "{}" created successfully.'.format(pipeline_name),
                  result.output)

  def _run_pipeline_using_kfp_client(self, pipeline_name: Text):
    try:
      pipeline_id, experiment_id = self._get_pipeline_id_and_experiment_id(
          pipeline_name)
      run = self._client.run_pipeline(
          experiment_id=experiment_id,
          job_name=pipeline_name,
          pipeline_id=pipeline_id)

      return run

    except kfp_server_api.rest.ApiException as err:
      absl.logging.info(err)

  def testPipelineCreate(self):
    # Create a pipeline.
    self._valid_create_and_check(self._pipeline_path, self._pipeline_name)

    # Test pipeline create when pipeline already exists.
    result = self.runner.invoke(cli_group, [
        'pipeline', 'create', '--engine', 'kubeflow', '--pipeline_path',
        self._pipeline_path, '--endpoint', self._endpoint
    ])
    self.assertIn('Creating pipeline', result.output)
    self.assertTrue('Pipeline "{}" already exists.'.format(self._pipeline_name),
                    result.output)

  def testPipelineUpdate(self):
    # Try pipeline update when pipeline does not exist.
    result = self.runner.invoke(cli_group, [
        'pipeline', 'update', '--engine', 'kubeflow', '--pipeline_path',
        self._pipeline_path, '--endpoint', self._endpoint
    ])
    self.assertIn('Updating pipeline', result.output)
    self.assertIn('Pipeline "{}" does not exist.'.format(self._pipeline_name),
                  result.output)
    self.assertFalse(tf.io.gfile.exists(self._handler_pipeline_path))

    # Now update an existing pipeline.
    self._valid_create_and_check(self._pipeline_path, self._pipeline_name)

    result = self.runner.invoke(cli_group, [
        'pipeline', 'update', '--engine', 'kubeflow', '--pipeline_path',
        self._pipeline_path, '--endpoint', self._endpoint
    ])
    self.assertIn('Updating pipeline', result.output)
    self.assertIn(
        'Pipeline "{}" updated successfully.'.format(self._pipeline_name),
        result.output)
    self.assertTrue(tf.io.gfile.exists(self._handler_pipeline_args_path))

  def testPipelineCompile(self):
    # Invalid DSL path
    pipeline_path = os.path.join(self._testdata_dir, 'test_pipeline_flink.py')
    result = self.runner.invoke(cli_group, [
        'pipeline', 'compile', '--engine', 'kubeflow', '--pipeline_path',
        pipeline_path
    ])
    self.assertIn('Compiling pipeline', result.output)
    self.assertIn('Invalid pipeline path: {}'.format(pipeline_path),
                  result.output)

    # Wrong Runner.
    pipeline_path = os.path.join(self._testdata_dir,
                                 'test_pipeline_airflow_1.py')
    result = self.runner.invoke(cli_group, [
        'pipeline', 'compile', '--engine', 'kubeflow', '--pipeline_path',
        pipeline_path
    ])
    self.assertIn('Compiling pipeline', result.output)
    self.assertIn('kubeflow runner not found in dsl.', result.output)

    # Successful compilation.
    result = self.runner.invoke(cli_group, [
        'pipeline', 'compile', '--engine', 'kubeflow', '--pipeline_path',
        self._pipeline_path
    ])
    self.assertIn('Compiling pipeline', result.output)
    self.assertIn('Pipeline compiled successfully', result.output)

  def testPipelineDelete(self):
    # Try deleting a non existent pipeline.
    result = self.runner.invoke(cli_group, [
        'pipeline', 'delete', '--engine', 'kubeflow', '--pipeline_name',
        self._pipeline_name, '--endpoint', self._endpoint
    ])
    self.assertIn('Deleting pipeline', result.output)
    self.assertIn('Pipeline "{}" does not exist.'.format(self._pipeline_name),
                  result.output)
    self.assertFalse(tf.io.gfile.exists(self._handler_pipeline_path))

    # Create a pipeline.
    self._valid_create_and_check(self._pipeline_path, self._pipeline_name)

    # Now delete the pipeline.
    result = self.runner.invoke(cli_group, [
        'pipeline', 'delete', '--engine', 'kubeflow', '--pipeline_name',
        self._pipeline_name, '--endpoint', self._endpoint
    ])
    self.assertIn('Deleting pipeline', result.output)
    self.assertFalse(tf.io.gfile.exists(self._handler_pipeline_path))
    self.assertIn(
        'Pipeline {} deleted successfully.'.format(self._pipeline_name),
        result.output)

  def testPipelineList(self):
    # Create pipelines.
    self._valid_create_and_check(self._pipeline_path, self._pipeline_name)
    self._valid_create_and_check(self._pipeline_path_v2, self._pipeline_name_v2)

    # List pipelines.
    result = self.runner.invoke(cli_group, [
        'pipeline', 'list', '--engine', 'kubeflow', '--endpoint', self._endpoint
    ])
    self.assertIn('Listing all pipelines', result.output)
    self.assertIn(self._pipeline_name, result.output)
    self.assertIn(self._pipeline_name_v2, result.output)

  def testPipelineCreateAutoDetect(self):
    result = self.runner.invoke(cli_group, [
        'pipeline', 'create', '--engine', 'auto', '--pipeline_path',
        self._pipeline_path, '--endpoint', self._endpoint
    ])
    self.assertIn('Creating pipeline', result.output)
    if labels.AIRFLOW_PACKAGE_NAME in self._pip_list and labels.KUBEFLOW_PACKAGE_NAME in self._pip_list:
      self.assertIn(
          'Multiple orchestrators found. Choose one using --engine flag.',
          result.output)
    else:
      self.assertTrue(tf.io.gfile.exists(self._pipeline_package_path))
      self.assertTrue(tf.io.gfile.exists(self._handler_pipeline_args_path))
      self.assertIn(
          'Pipeline "{}" created successfully.'.format(self._pipeline_name),
          result.output)

  def testRunCreate(self):
    # Try running a non-existent pipeline.
    result = self.runner.invoke(cli_group, [
        'run', 'create', '--engine', 'kubeflow', '--pipeline_name',
        self._pipeline_name, '--endpoint', self._endpoint
    ])
    self.assertIn('Creating a run for pipeline: {}'.format(self._pipeline_name),
                  result.output)
    self.assertIn('Pipeline "{}" does not exist.'.format(self._pipeline_name),
                  result.output)

    # Now create a pipeline.
    self._valid_create_and_check(self._pipeline_path, self._pipeline_name)

    # Run pipeline.
    result = self.runner.invoke(cli_group, [
        'run', 'create', '--engine', 'kubeflow', '--pipeline_name',
        self._pipeline_name, '--endpoint', self._endpoint
    ])

    self.assertIn('Creating a run for pipeline: {}'.format(self._pipeline_name),
                  result.output)
    self.assertNotIn(
        'Pipeline "{}" does not exist.'.format(self._pipeline_name),
        result.output)
    self.assertIn('Run created for pipeline: {}'.format(self._pipeline_name),
                  result.output)

  def testRunDelete(self):
    # Now create a pipeline.
    self._valid_create_and_check(self._pipeline_path, self._pipeline_name)

    # Run pipeline using kfp client to get run_id.
    run = self._run_pipeline_using_kfp_client(self._pipeline_name)

    # Delete run.
    result = self.runner.invoke(cli_group, [
        'run', 'delete', '--engine', 'kubeflow', '--endpoint', self._endpoint,
        '--run_id', run.id
    ])
    self.assertIn('Deleting run.', result.output)
    self.assertIn('Run deleted.', result.output)

  def testRunTerminate(self):
    # Now create a pipeline.
    self._valid_create_and_check(self._pipeline_path, self._pipeline_name)

    # Run pipeline using kfp client to get run_id.
    run = self._run_pipeline_using_kfp_client(self._pipeline_name)

    # Delete run.
    result = self.runner.invoke(cli_group, [
        'run', 'terminate', '--engine', 'kubeflow', '--endpoint',
        self._endpoint, '--run_id', run.id
    ])
    self.assertIn('Terminating run.', result.output)
    self.assertIn('Run terminated.', result.output)

  def testRunStatus(self):
    # Now create a pipeline.
    self._valid_create_and_check(self._pipeline_path, self._pipeline_name)

    # Run pipeline using kfp client to get run_id.
    run = self._run_pipeline_using_kfp_client(self._pipeline_name)

    # Delete run.
    result = self.runner.invoke(cli_group, [
        'run', 'status', '--engine', 'kubeflow', '--pipeline_name',
        self._pipeline_name, '--endpoint', self._endpoint, '--run_id', run.id
    ])
    self.assertIn('Retrieving run status.', result.output)
    self.assertIn(str(run.id), result.output)
    self.assertIn(self._pipeline_name, result.output)

  def testRunList(self):
    # Now create a pipeline.
    self._valid_create_and_check(self._pipeline_path, self._pipeline_name)

    # Run pipeline using kfp client to get run_id.
    run_1 = self._run_pipeline_using_kfp_client(self._pipeline_name)
    run_2 = self._run_pipeline_using_kfp_client(self._pipeline_name)

    # List runs.
    result = self.runner.invoke(cli_group, [
        'run', 'list', '--engine', 'kubeflow', '--pipeline_name',
        self._pipeline_name, '--endpoint', self._endpoint
    ])
    self.assertIn(
        'Listing all runs of pipeline: {}'.format(self._pipeline_name),
        result.output)
    self.assertIn(str(run_1.id), result.output)
    self.assertIn(str(run_2.id), result.output)
    self.assertIn(self._pipeline_name, result.output)


if __name__ == '__main__':
  logging.basicConfig(stream=sys.stdout, level=logging.INFO)
  tf.test.main()
