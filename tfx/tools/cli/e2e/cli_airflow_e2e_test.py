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
"""E2E Airflow tests for CLI."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import locale
import os
import subprocess
import sys
import tempfile
import time

import absl
from click import testing as click_testing
import tensorflow as tf
from tfx.tools.cli import labels
from tfx.tools.cli.cli_main import cli_group
from tfx.utils import io_utils


class CliAirflowEndToEndTest(tf.test.TestCase):

  def setUp(self):
    super(CliAirflowEndToEndTest, self).setUp()

    # List of packages installed.
    self._pip_list = str(subprocess.check_output(['pip', 'freeze', '--local']))

    # Check if Apache Airflow is installed before running E2E tests.
    if labels.AIRFLOW_PACKAGE_NAME not in self._pip_list:
      sys.exit('Apache Airflow not installed.')

    # Change the encoding for Click since Python 3 is configured to use ASCII as
    # encoding for the environment.
    if codecs.lookup(locale.getpreferredencoding()).name == 'ascii':
      os.environ['LANG'] = 'en_US.utf-8'

    # Setup airflow_home in a temp directory
    self._airflow_home = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', tempfile.mkdtemp()),
        self._testMethodName, 'airflow')
    self._old_airflow_home = os.environ.get('AIRFLOW_HOME')
    os.environ['AIRFLOW_HOME'] = self._airflow_home
    self._old_home = os.environ.get('HOME')
    os.environ['HOME'] = self._airflow_home
    absl.logging.info('Using %s as AIRFLOW_HOME and HOME in this e2e test',
                      self._airflow_home)

    # Testdata path.
    self._testdata_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')

    # Do not load examples to make this a bit faster.
    os.environ['AIRFLOW__CORE__LOAD_EXAMPLES'] = 'False'

    # Copy data.
    chicago_taxi_pipeline_dir = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        'examples', 'chicago_taxi_pipeline', '')
    data_dir = os.path.join(chicago_taxi_pipeline_dir, 'data', 'simple')
    content = tf.io.gfile.listdir(data_dir)
    assert content, 'content in {} is empty'.format(data_dir)
    target_data_dir = os.path.join(self._airflow_home, 'taxi', 'data', 'simple')
    io_utils.copy_dir(data_dir, target_data_dir)
    assert tf.io.gfile.isdir(target_data_dir)
    content = tf.io.gfile.listdir(target_data_dir)
    assert content, 'content in {} is {}'.format(target_data_dir, content)
    io_utils.copy_file(
        os.path.join(chicago_taxi_pipeline_dir, 'taxi_utils.py'),
        os.path.join(self._airflow_home, 'taxi', 'taxi_utils.py'))

    self._airflow_initdb()

    # Initialize CLI runner.
    self.runner = click_testing.CliRunner()

  def tearDown(self):
    super(CliAirflowEndToEndTest, self).tearDown()
    if self._old_airflow_home:
      os.environ['AIRFLOW_HOME'] = self._old_airflow_home
    if self._old_home:
      os.environ['HOME'] = self._old_home

  def _airflow_initdb(self):
    _ = subprocess.check_output(['airflow', 'initdb'])

  def _reload_airflow_dags(self):
    # Created pipelines can be registered to the DB by airflow scheduler
    # asynchronously. But we will rely on `initdb` which does same job
    # synchronously for deterministic and fast test execution.
    # (And it doesn't initialize db.)
    self._airflow_initdb()

  def _valid_create_and_check(self, pipeline_path, pipeline_name):
    handler_pipeline_path = os.path.join(self._airflow_home, 'dags',
                                         pipeline_name)

    # Create a pipeline.
    result = self.runner.invoke(cli_group, [
        'pipeline', 'create', '--engine', 'airflow', '--pipeline_path',
        pipeline_path
    ])

    self.assertIn('CLI', result.output)
    self.assertIn('Creating pipeline', result.output)
    self.assertTrue(
        tf.io.gfile.exists(
            os.path.join(handler_pipeline_path, 'pipeline_args.json')))
    self.assertIn('Pipeline "{}" created successfully.'.format(pipeline_name),
                  result.output)

  def testPipelineCreateAutoDetect(self):
    pipeline_path = os.path.join(self._testdata_dir,
                                 'test_pipeline_airflow_1.py')
    pipeline_name = 'chicago_taxi_simple'
    handler_pipeline_path = os.path.join(self._airflow_home, 'dags',
                                         pipeline_name)
    result = self.runner.invoke(cli_group, [
        'pipeline', 'create', '--engine', 'auto', '--pipeline_path',
        pipeline_path
    ])
    self.assertIn('CLI', result.output)
    self.assertIn('Creating pipeline', result.output)
    if labels.AIRFLOW_PACKAGE_NAME in self._pip_list and labels.KUBEFLOW_PACKAGE_NAME in self._pip_list:
      self.assertIn(
          'Multiple orchestrators found. Choose one using --engine flag.',
          result.output)
    else:
      self.assertIn('Detected Airflow', result.output)
      self.assertTrue(
          tf.io.gfile.exists(
              os.path.join(handler_pipeline_path, 'pipeline_args.json')))
      self.assertIn('Pipeline "{}" created successfully.'.format(pipeline_name),
                    result.output)

  def testPipelineCreate(self):
    # Create a pipeline.
    pipeline_path = os.path.join(self._testdata_dir,
                                 'test_pipeline_airflow_1.py')
    pipeline_name = 'chicago_taxi_simple'
    self._valid_create_and_check(pipeline_path, pipeline_name)

    # Test pipeline create when pipeline already exists.
    result = self.runner.invoke(cli_group, [
        'pipeline', 'create', '--engine', 'airflow', '--pipeline_path',
        pipeline_path
    ])
    self.assertIn('CLI', result.output)
    self.assertIn('Creating pipeline', result.output)
    self.assertTrue('Pipeline "{}" already exists.'.format(pipeline_name),
                    result.output)

  def testPipelineUpdate(self):
    pipeline_name = 'chicago_taxi_simple'
    handler_pipeline_path = os.path.join(self._airflow_home, 'dags',
                                         pipeline_name)

    # Try pipeline update when pipeline does not exist.
    pipeline_path_1 = os.path.join(self._testdata_dir,
                                   'test_pipeline_airflow_1.py')
    result = self.runner.invoke(cli_group, [
        'pipeline', 'update', '--engine', 'airflow', '--pipeline_path',
        pipeline_path_1
    ])
    self.assertIn('CLI', result.output)
    self.assertIn('Updating pipeline', result.output)
    self.assertIn('Pipeline "{}" does not exist.'.format(pipeline_name),
                  result.output)
    self.assertFalse(tf.io.gfile.exists(handler_pipeline_path))

    # Now update an existing pipeline.
    self._valid_create_and_check(pipeline_path_1, pipeline_name)

    pipeline_path_2 = os.path.join(self._testdata_dir,
                                   'test_pipeline_airflow_2.py')
    result = self.runner.invoke(cli_group, [
        'pipeline', 'update', '--engine', 'airflow', '--pipeline_path',
        pipeline_path_2
    ])
    self.assertIn('CLI', result.output)
    self.assertIn('Updating pipeline', result.output)
    self.assertIn('Pipeline "{}" updated successfully.'.format(pipeline_name),
                  result.output)
    self.assertTrue(
        tf.io.gfile.exists(
            os.path.join(handler_pipeline_path, 'pipeline_args.json')))

  def testPipelineCompile(self):

    # Invalid DSL path
    pipeline_path = os.path.join(self._testdata_dir, 'test_pipeline_flink.py')
    result = self.runner.invoke(cli_group, [
        'pipeline', 'compile', '--engine', 'airflow', '--pipeline_path',
        pipeline_path
    ])
    self.assertIn('CLI', result.output)
    self.assertIn('Compiling pipeline', result.output)
    self.assertIn('Invalid pipeline path: {}'.format(pipeline_path),
                  result.output)

    # Wrong Runner.
    pipeline_path = os.path.join(self._testdata_dir,
                                 'test_pipeline_kubeflow_1.py')
    result = self.runner.invoke(cli_group, [
        'pipeline', 'compile', '--engine', 'airflow', '--pipeline_path',
        pipeline_path
    ])
    self.assertIn('CLI', result.output)
    self.assertIn('Compiling pipeline', result.output)
    self.assertIn('airflow runner not found in dsl.', result.output)

    # Successful compilation.
    pipeline_path = os.path.join(self._testdata_dir,
                                 'test_pipeline_airflow_2.py')
    result = self.runner.invoke(cli_group, [
        'pipeline', 'compile', '--engine', 'airflow', '--pipeline_path',
        pipeline_path
    ])
    self.assertIn('CLI', result.output)
    self.assertIn('Compiling pipeline', result.output)
    self.assertIn('Pipeline compiled successfully', result.output)

  def testPipelineDelete(self):
    pipeline_path = os.path.join(self._testdata_dir,
                                 'test_pipeline_airflow_1.py')
    pipeline_name = 'chicago_taxi_simple'
    handler_pipeline_path = os.path.join(self._airflow_home, 'dags',
                                         pipeline_name)

    # Try deleting a non existent pipeline.
    result = self.runner.invoke(cli_group, [
        'pipeline', 'delete', '--engine', 'airflow', '--pipeline_name',
        pipeline_name
    ])
    self.assertIn('CLI', result.output)
    self.assertIn('Deleting pipeline', result.output)
    self.assertIn('Pipeline "{}" does not exist.'.format(pipeline_name),
                  result.output)
    self.assertFalse(tf.io.gfile.exists(handler_pipeline_path))

    # Create a pipeline.
    self._valid_create_and_check(pipeline_path, pipeline_name)

    # Now delete the pipeline.
    result = self.runner.invoke(cli_group, [
        'pipeline', 'delete', '--engine', 'airflow', '--pipeline_name',
        pipeline_name
    ])
    self.assertIn('CLI', result.output)
    self.assertIn('Deleting pipeline', result.output)
    self.assertFalse(tf.io.gfile.exists(handler_pipeline_path))
    self.assertIn('Pipeline "{}" deleted successfully.'.format(pipeline_name),
                  result.output)

  def testPipelineList(self):

    # Try listing pipelines when there are none.
    result = self.runner.invoke(cli_group,
                                ['pipeline', 'list', '--engine', 'airflow'])
    self.assertIn('CLI', result.output)
    self.assertIn('Listing all pipelines', result.output)

    # Create pipelines.
    pipeline_name_1 = 'chicago_taxi_simple'
    pipeline_path_1 = os.path.join(self._testdata_dir,
                                   'test_pipeline_airflow_1.py')
    self._valid_create_and_check(pipeline_path_1, pipeline_name_1)

    pipeline_name_2 = 'chicago_taxi_simple_v2'
    pipeline_path_2 = os.path.join(self._testdata_dir,
                                   'test_pipeline_airflow_3.py')
    self._valid_create_and_check(pipeline_path_2, pipeline_name_2)

    # List pipelines.
    result = self.runner.invoke(cli_group,
                                ['pipeline', 'list', '--engine', 'airflow'])
    self.assertIn('CLI', result.output)
    self.assertIn('Listing all pipelines', result.output)
    self.assertIn(pipeline_name_1, result.output)
    self.assertIn(pipeline_name_2, result.output)

  def testPipelineSchemaError(self):
    pipeline_path = os.path.join(self._testdata_dir,
                                 'test_pipeline_airflow_2.py')
    pipeline_name = 'chicago_taxi_simple'

    # Try getting schema without creating pipeline.
    result = self.runner.invoke(cli_group, [
        'pipeline', 'schema', '--engine', 'airflow', '--pipeline_name',
        pipeline_name
    ])
    self.assertIn('CLI', result.output)
    self.assertIn('Getting latest schema.', result.output)
    self.assertIn('Pipeline "{}" does not exist.'.format(pipeline_name),
                  result.output)

    # Create a pipeline.
    self._valid_create_and_check(pipeline_path, pipeline_name)

    # Try getting schema without creating a pipeline run.
    result = self.runner.invoke(cli_group, [
        'pipeline', 'schema', '--engine', 'airflow', '--pipeline_name',
        pipeline_name
    ])
    self.assertIn('CLI', result.output)
    self.assertIn('Getting latest schema.', result.output)
    self.assertIn(
        'Create a run before inferring schema. If pipeline is already running, then wait for it to successfully finish.',
        result.output)

    # Run pipeline.
    self._valid_run_and_check(pipeline_name)

    # Try inferring schema without SchemaGen output.
    result = self.runner.invoke(cli_group, [
        'pipeline', 'schema', '--engine', 'airflow', '--pipeline_name',
        pipeline_name
    ])
    self.assertIn('CLI', result.output)
    self.assertIn('Getting latest schema.', result.output)
    self.assertIn(
        'Create a run before inferring schema. If pipeline is already running, then wait for it to successfully finish.',
        result.output)

  def _valid_run_and_check(self, pipeline_name):
    # Wait to fill up the DagBag.
    response = ''
    while pipeline_name not in response:
      response = str(
          subprocess.check_output(['airflow', 'list_dags', '--report']))

    self._reload_airflow_dags()

    result = self.runner.invoke(cli_group, [
        'run', 'create', '--engine', 'airflow', '--pipeline_name', pipeline_name
    ])

    self.assertIn('CLI', result.output)
    self.assertIn('Creating a run for pipeline: {}'.format(pipeline_name),
                  result.output)
    self.assertNotIn('Pipeline "{}" does not exist.'.format(pipeline_name),
                     result.output)
    self.assertIn('Run created for pipeline: {}'.format(pipeline_name),
                  result.output)

    # NOTE: Because airflow scheduler was not launched, this run will not "run",
    #       actually. This e2e test covers CLI side of the execution only.

  def testRunCreate(self):
    pipeline_name = 'chicago_taxi_simple'
    pipeline_path = os.path.join(self._testdata_dir,
                                 'test_pipeline_airflow_1.py')

    # Try running a non-existent pipeline.
    result = self.runner.invoke(cli_group, [
        'run', 'create', '--engine', 'airflow', '--pipeline_name', pipeline_name
    ])
    self.assertIn('CLI', result.output)
    self.assertIn('Creating a run for pipeline: {}'.format(pipeline_name),
                  result.output)
    self.assertIn('Pipeline "{}" does not exist.'.format(pipeline_name),
                  result.output)

    # Now create a pipeline.
    self._valid_create_and_check(pipeline_path, pipeline_name)

    # Run pipeline.
    self._valid_run_and_check(pipeline_name)

  def testRunList(self):
    pipeline_name = 'chicago_taxi_simple'
    pipeline_path = os.path.join(self._testdata_dir,
                                 'test_pipeline_airflow_1.py')

    # Now create a pipeline.
    self._valid_create_and_check(pipeline_path, pipeline_name)

    # Check if pipeline runs exist.
    result = self.runner.invoke(cli_group, [
        'run', 'list', '--engine', 'airflow', '--pipeline_name', pipeline_name
    ])
    self.assertIn('CLI', result.output)
    self.assertIn('Listing all runs of pipeline: {}'.format(pipeline_name),
                  result.output)
    self.assertIn('No pipeline runs for {}'.format(pipeline_name),
                  result.output)

    # Run pipeline.
    self._valid_run_and_check(pipeline_name)

    time.sleep(1)   # Sleep to ensure two pipelines have different timestamps.
    # Run pipeline again.
    self._valid_run_and_check(pipeline_name)

    # List pipeline runs.
    result = self.runner.invoke(cli_group, [
        'run', 'list', '--engine', 'airflow', '--pipeline_name', pipeline_name
    ])
    self.assertIn('CLI', result.output)
    self.assertIn('Listing all runs of pipeline: {}'.format(pipeline_name),
                  result.output)

  def testUninstalledOrchestratorKubeflow(self):
    result = self.runner.invoke(cli_group,
                                ['pipeline', 'list', '--engine', 'kubeflow'])
    self.assertIn('CLI', result.output)
    self.assertIn('Listing all pipelines', result.output)
    # When only Airflow is installed.
    if labels.KUBEFLOW_PACKAGE_NAME not in self._pip_list:
      self.assertIn('Kubeflow not found', result.output)


if __name__ == '__main__':
  tf.test.main()
