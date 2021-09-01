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
"""E2E Tests for tfx.tools.cli."""

import codecs
import locale
import os

from click import testing as click_testing
import tensorflow as tf

from tfx.tools.cli.cli_main import cli_group


class CliCommonEndToEndTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()

    # Change the encoding for Click since Python 3 is configured to use ASCII as
    # encoding for the environment.
    if codecs.lookup(locale.getpreferredencoding()).name == 'ascii':
      os.environ['LANG'] = 'en_US.utf-8'

    self.chicago_taxi_pipeline_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')
    self.runner = click_testing.CliRunner()

  def testPipelineCreateUnsupportedEngine(self):
    pipeline_path = os.path.join(self.chicago_taxi_pipeline_dir,
                                 'test_pipeline_beam_1.py')
    result = self.runner.invoke(cli_group, [
        'pipeline', 'create', '--engine', 'flink', '--pipeline_path',
        pipeline_path
    ])
    self.assertIn('CLI', result.output)
    self.assertIn('Creating pipeline', result.output)
    self.assertIn('Engine flink is not supported.', str(result.exception))

  def testMissingRequiredFlag(self):
    pipeline_name_1 = 'chicago_taxi_simple'

    # Missing flag for pipeline create.
    result = self.runner.invoke(cli_group,
                                ['pipeline', 'create', '--engine', 'beam'])
    self.assertIn('CLI', result.output)
    self.assertIn('Missing option', result.output)
    self.assertIn('--pipeline_path', result.output)

    # Missing flag for run create.
    result = self.runner.invoke(cli_group,
                                ['run', 'create', '--engine', 'airflow'])
    self.assertIn('CLI', result.output)
    self.assertIn('Missing option', result.output)
    self.assertIn('--pipeline_name', result.output)

    # Missing flag for run status.
    result = self.runner.invoke(
        cli_group, ['run', 'status', '--pipeline_name', pipeline_name_1])
    self.assertIn('CLI', result.output)
    self.assertIn('Missing option', result.output)
    self.assertIn('--run_id', result.output)


if __name__ == '__main__':
  tf.test.main()
