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
"""Tests for Kubeflow V2 pipeline commands."""
# TODO(b/154611466): Add kokoro test coverage for this test.

import codecs
import locale
import os
import sys
from unittest import mock

from click import testing as click_testing
import tensorflow as tf
from tfx.tools.cli.kubeflow_v2.commands.pipeline import pipeline_group


class PipelineTest(tf.test.TestCase):

  def setUp(self):
    # Change the encoding for Click since Python 3 is configured to use ASCII as
    # encoding for the environment.
    super(PipelineTest, self).setUp()
    if codecs.lookup(locale.getpreferredencoding()).name == 'ascii':
      os.environ['LANG'] = 'en_US.utf-8'
    self.runner = click_testing.CliRunner()
    sys.modules['handler_factory'] = mock.Mock()

  def testPipelineCreate(self):
    result = self.runner.invoke(pipeline_group, [
        'create', '--pipeline_path', 'chicago.py', '--build_image',
        '--build_base_image', 'gcr.io/my-base-image'
    ])
    self.assertIn('Creating pipeline', result.output)
    result = self.runner.invoke(pipeline_group, [
        'create', '--pipeline-path', 'chicago.py', '--build-image',
        '--build-base-image', 'gcr.io/my-base-image'
    ])
    self.assertIn('Creating pipeline', result.output)

  def testPipelineUpdate(self):
    result = self.runner.invoke(pipeline_group, [
        'update',
        '--pipeline_path',
        'chicago.py',
    ])
    self.assertIn('Updating pipeline', result.output)
    result = self.runner.invoke(pipeline_group, [
        'update',
        '--pipeline-path',
        'chicago.py',
    ])
    self.assertIn('Updating pipeline', result.output)

  def testPipelineCompile(self):
    result = self.runner.invoke(pipeline_group, [
        'compile',
        '--pipeline_path',
        'chicago.py',
    ])
    self.assertIn('Compiling pipeline', result.output)
    result = self.runner.invoke(pipeline_group, [
        'compile',
        '--pipeline-path',
        'chicago.py',
    ])
    self.assertIn('Compiling pipeline', result.output)

  def testPipelineDelete(self):
    result = self.runner.invoke(pipeline_group,
                                ['delete', '--pipeline_name', 'chicago'])
    self.assertIn('Deleting pipeline', result.output)
    result = self.runner.invoke(pipeline_group,
                                ['delete', '--pipeline-name', 'chicago'])
    self.assertIn('Deleting pipeline', result.output)

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
