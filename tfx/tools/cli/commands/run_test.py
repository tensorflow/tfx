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
"""Tests for tfx.tools.cli.commands.run."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import locale
import os
import sys

from click import testing as click_testing
import mock
import tensorflow as tf

from tfx.tools.cli.commands.run import run_group


class RunTest(tf.test.TestCase):

  def setUp(self):
    # Change the encoding for Click since Python 3 is configured to use ASCII as
    # encoding for the environment.
    super(RunTest, self).setUp()
    if codecs.lookup(locale.getpreferredencoding()).name == 'ascii':
      os.environ['LANG'] = 'en_US.utf-8'
    self.runner = click_testing.CliRunner()
    sys.modules['handler_factory'] = mock.Mock()

  def testRunCreateAirflow(self):
    result = self.runner.invoke(
        run_group,
        ['create', '--pipeline_name', 'chicago', '--engine', 'airflow'])
    self.assertIn('Creating a run for pipeline', result.output)
    result = self.runner.invoke(
        run_group,
        ['create', '--pipeline-name', 'chicago', '--engine', 'airflow'])
    self.assertIn('Creating a run for pipeline', result.output)

  def testRunCreateKubeflow(self):
    result = self.runner.invoke(run_group, [
        'create', '--pipeline_name', 'chicago', '--engine', 'kubeflow',
        '--iap_client_id', 'fake_id', '--namespace', 'kubeflow', '--endpoint',
        'endpoint_url'
    ])
    self.assertIn('Creating a run for pipeline', result.output)
    result = self.runner.invoke(run_group, [
        'create', '--pipeline-name', 'chicago', '--engine', 'kubeflow',
        '--iap-client-id', 'fake_id', '--namespace', 'kubeflow', '--endpoint',
        'endpoint_url'
    ])
    self.assertIn('Creating a run for pipeline', result.output)

  def testRunList(self):
    result = self.runner.invoke(
        run_group,
        ['list', '--pipeline_name', 'chicago', '--engine', 'airflow'])
    self.assertIn('Listing all runs of pipeline', result.output)
    result = self.runner.invoke(
        run_group,
        ['list', '--pipeline-name', 'chicago', '--engine', 'airflow'])
    self.assertIn('Listing all runs of pipeline', result.output)

  def testRunStatusAirflow(self):
    result = self.runner.invoke(run_group, [
        'status', '--pipeline_name', 'chicago_taxi_pipeline', '--run_id',
        'airflow_run_id', '--engine', 'airflow'
    ])
    self.assertIn('Retrieving run status', result.output)
    result = self.runner.invoke(run_group, [
        'status', '--pipeline-name', 'chicago_taxi_pipeline', '--run-id',
        'airflow_run_id', '--engine', 'airflow'
    ])
    self.assertIn('Retrieving run status', result.output)

  def testRunStatusKubeflow(self):
    result = self.runner.invoke(run_group, [
        'status', '--pipeline_name', 'chicago_taxi_pipeline', '--run_id',
        'kubeflow_run_id', '--engine', 'kubeflow', '--iap_client_id', 'fake_id',
        '--namespace', 'kubeflow', '--endpoint', 'endpoint_url'
    ])
    self.assertIn('Retrieving run status', result.output)
    result = self.runner.invoke(run_group, [
        'status', '--pipeline-name', 'chicago_taxi_pipeline', '--run-id',
        'kubeflow_run_id', '--engine', 'kubeflow', '--iap-client-id', 'fake_id',
        '--namespace', 'kubeflow', '--endpoint', 'endpoint_url'
    ])
    self.assertIn('Retrieving run status', result.output)

  def testRunTerminate(self):
    result = self.runner.invoke(
        run_group,
        ['terminate', '--run_id', 'airflow_run_id', '--engine', 'airflow'])
    self.assertIn('Terminating run.', result.output)
    result = self.runner.invoke(
        run_group,
        ['terminate', '--run-id', 'airflow_run_id', '--engine', 'airflow'])
    self.assertIn('Terminating run.', result.output)

  def testRunDelete(self):
    result = self.runner.invoke(run_group, [
        'delete', '--run_id', 'kubeflow_run_id', '--engine', 'kubeflow',
        '--iap_client_id', 'fake_id', '--namespace', 'kubeflow', '--endpoint',
        'endpoint_url'
    ])
    self.assertIn('Deleting run', result.output)
    result = self.runner.invoke(run_group, [
        'delete', '--run-id', 'kubeflow_run_id', '--engine', 'kubeflow',
        '--iap-client-id', 'fake_id', '--namespace', 'kubeflow', '--endpoint',
        'endpoint_url'
    ])
    self.assertIn('Deleting run', result.output)


if __name__ == '__main__':
  tf.test.main()
