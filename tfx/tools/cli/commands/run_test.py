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
"""Tests for tfx.tools.cli.commands.run."""

import codecs
import locale
import os
from unittest import mock

from click import testing as click_testing
import tensorflow as tf

from tfx.tools.cli.commands.run import run_group
from tfx.tools.cli.handler import handler_factory
from tfx.utils import test_case_utils


class RunTest(test_case_utils.TfxTest):

  def setUp(self):
    # Change the encoding for Click since Python 3 is configured to use ASCII as
    # encoding for the environment.
    super().setUp()
    if codecs.lookup(locale.getpreferredencoding()).name == 'ascii':
      os.environ['LANG'] = 'en_US.utf-8'
    self.runner = click_testing.CliRunner()
    self.mock_create_handler = self.enter_context(
        mock.patch.object(handler_factory, 'create_handler', autospec=True))

  def assertSucceeded(self, result) -> None:
    # Test is expected to succeed
    if self.assertEqual(0, result.exit_code, f'test_result={result}'):
      pass

  def assertFailed(self, result) -> None:
    # Test is expected to fail
    if self.assertNotEqual(0, result.exit_code, f'test_result={result}'):
      pass

  def testRunCreateAirflow(self):
    result = self.runner.invoke(
        run_group,
        ['create', '--pipeline_name', 'chicago', '--engine', 'airflow'])
    self.assertIn('Creating a run for pipeline', result.output)
    self.assertSucceeded(result)
    result = self.runner.invoke(
        run_group,
        ['create', '--pipeline-name', 'chicago', '--engine', 'airflow'])
    self.assertIn('Creating a run for pipeline', result.output)
    self.assertSucceeded(result)
    result = self.runner.invoke(run_group, [
        'create', '--pipeline_name', 'chicago', '--engine', 'airflow',
        '--runtime_parameter', 'a=1'
    ])
    self.assertIn('Creating a run for pipeline', result.output)
    self.assertSucceeded(result)

  def testRunCreateKubeflow(self):
    result = self.runner.invoke(run_group, [
        'create', '--pipeline_name', 'chicago', '--engine', 'kubeflow',
        '--iap_client_id', 'fake_id', '--namespace', 'kubeflow', '--endpoint',
        'endpoint_url'
    ])
    self.assertIn('Creating a run for pipeline', result.output)
    self.assertSucceeded(result)
    result = self.runner.invoke(run_group, [
        'create', '--pipeline-name', 'chicago', '--engine', 'kubeflow',
        '--iap-client-id', 'fake_id', '--namespace', 'kubeflow', '--endpoint',
        'endpoint_url'
    ])
    self.assertIn('Creating a run for pipeline', result.output)
    self.assertEqual(0, result.exit_code, f'test_result={result}')
    result = self.runner.invoke(run_group, [
        'create', '--pipeline_name', 'chicago', '--engine', 'kubeflow',
        '--iap_client_id', 'fake_id', '--namespace', 'kubeflow', '--endpoint',
        'endpoint_url', '--runtime_parameter', 'a=1', '--runtime_parameter',
        'b=2'
    ])
    self.assertIn('Creating a run for pipeline', result.output)
    self.assertSucceeded(result)
    result = self.runner.invoke(run_group, [
        'create', '--pipeline_name', 'chicago', '--engine', 'kubeflow',
        '--iap_client_id', 'fake_id', '--namespace', 'kubeflow', '--endpoint',
        'endpoint_url', '--runtime_parameter', 'a=1', '--runtime_parameter',
        'b=2'
    ])
    self.assertIn('Creating a run for pipeline', result.output)
    self.assertSucceeded(result)

  def testRunList(self):
    result = self.runner.invoke(
        run_group,
        ['list', '--pipeline_name', 'chicago', '--engine', 'airflow'])
    self.assertIn('Listing all runs of pipeline', result.output)
    self.assertSucceeded(result)
    result = self.runner.invoke(
        run_group,
        ['list', '--pipeline-name', 'chicago', '--engine', 'airflow'])
    self.assertIn('Listing all runs of pipeline', result.output)
    self.assertSucceeded(result)

  def testRunStatusAirflow(self):
    result = self.runner.invoke(run_group, [
        'status', '--pipeline_name', 'chicago_taxi_pipeline', '--run_id',
        'airflow_run_id', '--engine', 'airflow'
    ])
    self.assertIn('Retrieving run status', result.output)
    self.assertSucceeded(result)
    result = self.runner.invoke(run_group, [
        'status', '--pipeline-name', 'chicago_taxi_pipeline', '--run-id',
        'airflow_run_id', '--engine', 'airflow'
    ])
    self.assertIn('Retrieving run status', result.output)
    self.assertSucceeded(result)

  def testRunStatusKubeflow(self):
    result = self.runner.invoke(run_group, [
        'status', '--pipeline_name', 'chicago_taxi_pipeline', '--run_id',
        'kubeflow_run_id', '--engine', 'kubeflow', '--iap_client_id', 'fake_id',
        '--namespace', 'kubeflow', '--endpoint', 'endpoint_url'
    ])
    self.assertIn('Retrieving run status', result.output)
    self.assertSucceeded(result)
    result = self.runner.invoke(run_group, [
        'status', '--pipeline-name', 'chicago_taxi_pipeline', '--run-id',
        'kubeflow_run_id', '--engine', 'kubeflow', '--iap-client-id', 'fake_id',
        '--namespace', 'kubeflow', '--endpoint', 'endpoint_url'
    ])
    self.assertIn('Retrieving run status', result.output)
    self.assertSucceeded(result)

  def testRunTerminate(self):
    result = self.runner.invoke(
        run_group,
        ['terminate', '--run_id', 'airflow_run_id', '--engine', 'airflow'])
    self.assertIn('Terminating run.', result.output)
    self.assertSucceeded(result)
    result = self.runner.invoke(
        run_group,
        ['terminate', '--run-id', 'airflow_run_id', '--engine', 'airflow'])
    self.assertIn('Terminating run.', result.output)
    self.assertSucceeded(result)

  def testRunDelete(self):
    result = self.runner.invoke(run_group, [
        'delete', '--run_id', 'kubeflow_run_id', '--engine', 'kubeflow',
        '--iap_client_id', 'fake_id', '--namespace', 'kubeflow', '--endpoint',
        'endpoint_url'
    ])
    self.assertIn('Deleting run', result.output)
    self.assertSucceeded(result)
    result = self.runner.invoke(run_group, [
        'delete', '--run-id', 'kubeflow_run_id', '--engine', 'kubeflow',
        '--iap-client-id', 'fake_id', '--namespace', 'kubeflow', '--endpoint',
        'endpoint_url'
    ])
    self.assertIn('Deleting run', result.output)
    self.assertSucceeded(result)


if __name__ == '__main__':
  tf.test.main()
