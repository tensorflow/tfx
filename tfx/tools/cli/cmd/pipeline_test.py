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

from click.testing import CliRunner
import tensorflow as tf
from tfx.tools.cli.cmd.pipeline import pipeline_group


class PipelineTest(tf.test.TestCase):

  def setUp(self):
    # Change the encoding for Click since Python 3 is configured to use ASCII as
    # encoding for the environment.
    if codecs.lookup(locale.getpreferredencoding()).name == 'ascii':
      os.environ['LANG'] = 'en_US.utf-8'
    self.runner = CliRunner()

  # TODO(b/132286477):Change tests after writing default_handler()
  def test_pipeline_create_auto(self):
    result = self.runner.invoke(pipeline_group,
                                ['create', '--path', 'chicago.py'])
    self.assertNotEqual(0, result.exit_code)
    self.assertIn('Creating pipeline', result.output)

  def test_pipeline_update(self):
    result = self.runner.invoke(
        pipeline_group,
        ['update', '--path', 'chicago.py', '--engine', 'kubeflow'])
    self.assertEqual(0, result.exit_code)
    self.assertIn('Updating pipeline', result.output)

  def test_pipeline_run(self):
    result = self.runner.invoke(
        pipeline_group, ['run', '--name', 'chicago', '--engine', 'airflow'])
    self.assertIn('Triggering pipeline', result.output)

  def test_pipeline_delete(self):
    result = self.runner.invoke(
        pipeline_group, ['delete', '--name', 'chicago', '--engine', 'airflow'])
    self.assertIn('Deleting pipeline', result.output)

  def test_pipeline_list(self):
    result = self.runner.invoke(pipeline_group, ['list', '--engine', 'airflow'])
    self.assertIn('Listing all pipelines', result.output)

  def test_pipeline_invalid_flag(self):
    result = self.runner.invoke(pipeline_group,
                                ['create', '--name', 'chicago.py'])
    self.assertIn('no such option', result.output)
    self.assertNotEqual(0, result.exit_code)

  def test_pipeline_invalid_flag_type(self):
    result = self.runner.invoke(pipeline_group, ['update', '--name', 1])
    self.assertNotEqual(0, result.exit_code)

  def test_pipeline_missing_flag(self):
    result = self.runner.invoke(pipeline_group, ['run'])
    self.assertIn('Missing option', result.output)
    self.assertNotEqual(0, result.exit_code)

  def test_pipeline_invalid_command(self):
    result = self.runner.invoke(pipeline_group, ['rerun'])
    self.assertIn('No such command', result.output)
    self.assertNotEqual(0, result.exit_code)

  def test_pipeline_empty_flag_value(self):
    result = self.runner.invoke(pipeline_group, ['create', '--path'])
    self.assertIn('option requires an argument', result.output)
    self.assertNotEqual(0, result.exit_code)


if __name__ == '__main__':
  tf.test.main()
