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
"""Tests for tfx.tools.cli.cmd.pipeline_commands."""

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

from tfx.tools.cli.commands.pipeline import pipeline_group


class PipelineTest(tf.test.TestCase):

  def setUp(self):
    # Change the encoding for Click since Python 3 is configured to use ASCII as
    # encoding for the environment.
    super(PipelineTest, self).setUp()
    if codecs.lookup(locale.getpreferredencoding()).name == 'ascii':
      os.environ['LANG'] = 'en_US.utf-8'
    self.runner = click_testing.CliRunner()
    sys.modules['handler_factory'] = mock.Mock()

  def testPipelineCreateAuto(self):
    result = self.runner.invoke(pipeline_group,
                                ['create', '--pipeline_path', 'chicago.py'])
    self.assertIn('Creating pipeline', result.output)
    result = self.runner.invoke(pipeline_group,
                                ['create', '--pipeline-path', 'chicago.py'])
    self.assertIn('Creating pipeline', result.output)

  def testPipelineUpdate(self):
    result = self.runner.invoke(pipeline_group, [
        'update', '--pipeline_path', 'chicago.py', '--engine', 'kubeflow',
        '--package_path', 'chicago.tar.gz', '--iap_client_id', 'fake_id',
        '--namespace', 'kubeflow', '--endpoint', 'endpoint_url'
    ])
    self.assertIn('Updating pipeline', result.output)
    result = self.runner.invoke(pipeline_group, [
        'update', '--pipeline-path', 'chicago.py', '--engine', 'kubeflow',
        '--package-path', 'chicago.tar.gz', '--iap-client-id', 'fake_id',
        '--namespace', 'kubeflow', '--endpoint', 'endpoint_url'
    ])
    self.assertIn('Updating pipeline', result.output)

  def testPipelineDelete(self):
    result = self.runner.invoke(
        pipeline_group,
        ['delete', '--pipeline_name', 'chicago', '--engine', 'airflow'])
    self.assertIn('Deleting pipeline', result.output)
    result = self.runner.invoke(
        pipeline_group,
        ['delete', '--pipeline-name', 'chicago', '--engine', 'airflow'])
    self.assertIn('Deleting pipeline', result.output)

  def testPipelineList(self):
    result = self.runner.invoke(pipeline_group, ['list', '--engine', 'airflow'])
    self.assertIn('Listing all pipelines', result.output)

  def testPipelineCompile(self):
    result = self.runner.invoke(pipeline_group, [
        'compile', '--pipeline_path', 'chicago.py', '--engine', 'kubeflow',
        '--package_path', 'chicago.tar.gz'
    ])
    self.assertIn('Compiling pipeline', result.output)
    result = self.runner.invoke(pipeline_group, [
        'compile', '--pipeline-path', 'chicago.py', '--engine', 'kubeflow',
        '--package-path', 'chicago.tar.gz'
    ])
    self.assertIn('Compiling pipeline', result.output)

  def testPipelineInvalidFlag(self):
    result = self.runner.invoke(pipeline_group,
                                ['create', '--pipeline_name', 'chicago.py'])
    self.assertIn('no such option', result.output)
    self.assertNotEqual(0, result.exit_code)

  def testPipelineInvalidFlagType(self):
    result = self.runner.invoke(pipeline_group,
                                ['update', '--pipeline_name', 1])
    self.assertNotEqual(0, result.exit_code)

  def testPipelineMissingFlag(self):
    result = self.runner.invoke(pipeline_group, ['update'])
    self.assertIn('Missing option', result.output)
    self.assertNotEqual(0, result.exit_code)

  def testPipelineInvalidCommand(self):
    result = self.runner.invoke(pipeline_group, ['rerun'])
    self.assertIn('No such command', result.output)
    self.assertNotEqual(0, result.exit_code)

  def testPipelineEmptyFlagValue(self):
    result = self.runner.invoke(pipeline_group, ['create', '--pipeline_path'])
    self.assertIn('option requires an argument', result.output)
    self.assertNotEqual(0, result.exit_code)


if __name__ == '__main__':
  tf.test.main()
