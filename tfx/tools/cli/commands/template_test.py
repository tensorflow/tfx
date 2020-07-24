# Lint as: python2, python3
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
"""Tests for tfx.tools.cli.commands.copy_template."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import locale
import os

from absl import logging
from click import testing as click_testing
import mock
import tensorflow as tf

from tfx.tools.cli.commands.template import template_group
from tfx.tools.cli.handler import template_handler


class TemplateTest(tf.test.TestCase):

  def setUp(self):
    super(TemplateTest, self).setUp()
    # Change the encoding for Click since Python 3 is configured to use ASCII as
    # encoding for the environment.
    if codecs.lookup(locale.getpreferredencoding()).name == 'ascii':
      os.environ['LANG'] = 'en_US.utf-8'
      logging.info('Changing locale to %s to ensure UTF-8 environment.',
                   os.environ['LANG'])
    self.runner = click_testing.CliRunner()
    self.addCleanup(mock.patch.stopall)
    mock.patch.object(template_handler, 'list_template').start()
    mock.patch.object(template_handler, 'copy_template').start()

  def testListSuccess(self):
    result = self.runner.invoke(template_group, ['list'])
    self.assertEqual(0, result.exit_code)
    self.assertIn('Available templates', result.output)

  def testMissingPipelineName(self):
    result = self.runner.invoke(
        template_group, ['copy', '--model', 'm', '--destination_path', '/path'])
    self.assertNotEqual(0, result.exit_code)
    self.assertIn('pipeline_name', result.output)

  def testMissingDestinationPath(self):
    result = self.runner.invoke(
        template_group, ['copy', '--pipeline_name', 'p', '--model', 'm'])
    self.assertNotEqual(0, result.exit_code)
    self.assertIn('destination_path', result.output)

  def testMissingModel(self):
    result = self.runner.invoke(
        template_group,
        ['copy', '--pipeline_name', 'p', '--destination_path', '/path'])
    self.assertNotEqual(0, result.exit_code)
    self.assertIn('model', result.output)

  def testCopySuccess(self):
    result = self.runner.invoke(template_group, [
        'copy', '--pipeline_name', 'p', '--destination_path', '/path',
        '--model', 'm'
    ])
    self.assertEqual(0, result.exit_code)
    self.assertIn('Copying', result.output)
    result = self.runner.invoke(template_group, [
        'copy', '--pipeline-name', 'p', '--destination-path', '/path',
        '--model', 'm'
    ])
    self.assertEqual(0, result.exit_code)
    self.assertIn('Copying', result.output)


if __name__ == '__main__':
  tf.test.main()
