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

import codecs
import locale
import os
from unittest import mock

from click import testing as click_testing

from tfx.tools.cli.commands.pipeline import pipeline_group
from tfx.tools.cli.handler import handler_factory
from tfx.utils import test_case_utils


class PipelineTest(test_case_utils.TfxTest):

  def setUp(self):
    # Change the encoding for Click since Python 3 is configured to use ASCII as
    # encoding for the environment.
    super().setUp()
    if codecs.lookup(locale.getpreferredencoding()).name == 'ascii':
      os.environ['LANG'] = 'en_US.utf-8'
    self.runner = click_testing.CliRunner()
    self.mock_create_handler = self.enter_context(
        mock.patch.object(handler_factory, 'create_handler', autospec=True))

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
        '--iap_client_id', 'fake_id', '--namespace', 'kubeflow', '--endpoint',
        'endpoint_url'
    ])
    self.assertIn('Updating pipeline', result.output)
    result = self.runner.invoke(pipeline_group, [
        'update', '--pipeline-path', 'chicago.py', '--engine', 'kubeflow',
        '--iap-client-id', 'fake_id', '--namespace', 'kubeflow', '--endpoint',
        'endpoint_url'
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
    result = self.runner.invoke(
        pipeline_group,
        ['compile', '--pipeline_path', 'chicago.py', '--engine', 'kubeflow'])
    self.assertIn('Compiling pipeline', result.output)
    result = self.runner.invoke(
        pipeline_group,
        ['compile', '--pipeline-path', 'chicago.py', '--engine', 'kubeflow'])
    self.assertIn('Compiling pipeline', result.output)

  def testPipelineInvalidFlag(self):
    result = self.runner.invoke(pipeline_group,
                                ['create', '--pipeline_name', 'chicago.py'])
    self.assertNotEqual(0, result.exit_code)

  def testPipelineInvalidFlagType(self):
    result = self.runner.invoke(pipeline_group,
                                ['update', '--pipeline_name', 1])
    self.assertNotEqual(0, result.exit_code)

  def testPipelineMissingFlag(self):
    result = self.runner.invoke(pipeline_group, ['update'])
    self.assertNotEqual(0, result.exit_code)

  def testPipelineInvalidCommand(self):
    result = self.runner.invoke(pipeline_group, ['rerun'])
    self.assertNotEqual(0, result.exit_code)

  def testPipelineEmptyFlagValue(self):
    result = self.runner.invoke(pipeline_group, ['create', '--pipeline_path'])
    self.assertNotEqual(0, result.exit_code)

  def testPipelineDeprecatedFlags(self):
    result = self.runner.invoke(pipeline_group, [
        'create', '--pipeline_path', 'chicago.py', '--skaffold-cmd', 'skaffold'
    ])
    self.assertIn('skaffold-cmd', result.output)
    self.assertNotEqual(0, result.exit_code)

    result = self.runner.invoke(pipeline_group, [
        'create', '--pipeline_path', 'chicago.py', '--build-target-image',
        'tf/tfx'
    ])
    self.assertIn('build-target-image', result.output)
    self.assertNotEqual(0, result.exit_code)

    result = self.runner.invoke(pipeline_group, [
        'create', '--pipeline_path', 'chicago.py', '--pipeline-package-path',
        'a.tar.gz'
    ])
    self.assertIn('pipeline-package-path', result.output)
    self.assertNotEqual(0, result.exit_code)

    result = self.runner.invoke(pipeline_group, [
        'update',
        '--pipeline-path',
        'chicago.py',
        '--engine',
        'kubeflow',
        '--endpoint',
        'endpoint_url',
        '--pipeline-package-path',
        'a.tar.gz',
    ])
    self.assertIn('pipeline-package-path', result.output)
    self.assertNotEqual(0, result.exit_code)

    result = self.runner.invoke(pipeline_group, [
        'compile',
        '--pipeline-path',
        'chicago.py',
        '--engine',
        'kubeflow',
        '--pipeline-package-path',
        'a.tar.gz',
    ])
    self.assertIn('pipeline-package-path', result.output)
    self.assertNotEqual(0, result.exit_code)
