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
"""E2E Beam tests for CLI."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import locale
import os

from click import testing as click_testing
import tensorflow as tf

from tfx.tools.cli.cli_main import cli_group


class CliBeamEndToEndTest(tf.test.TestCase):

  def setUp(self):
    super(CliBeamEndToEndTest, self).setUp()

    # Change the encoding for Click since Python 3 is configured to use ASCII as
    # encoding for the environment.
    if codecs.lookup(locale.getpreferredencoding()).name == 'ascii':
      os.environ['LANG'] = 'en_US.utf-8'

    # Set home folders for engines.
    self._home = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    self._original_home_value = os.environ.get('HOME', '')
    os.environ['HOME'] = self._home
    self._original_beam_home_value = os.environ.get('BEAM_HOME', '')
    os.environ['BEAM_HOME'] = os.path.join(os.environ['HOME'], 'beam')

    self.chicago_taxi_pipeline_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')
    self.runner = click_testing.CliRunner()

  def tearDown(self):
    super(CliBeamEndToEndTest, self).tearDown()
    os.environ['HOME'] = self._original_home_value
    os.environ['BEAM_HOME'] = self._original_beam_home_value

  def _valid_create_and_check(self, pipeline_path, pipeline_name):
    handler_pipeline_path = os.path.join(os.environ['BEAM_HOME'], pipeline_name)

    # Create a pipeline.
    result = self.runner.invoke(cli_group, [
        'pipeline', 'create', '--engine', 'beam', '--pipeline_path',
        pipeline_path
    ])
    self.assertIn('CLI', result.output)
    self.assertIn('Creating pipeline', result.output)
    self.assertTrue(
        tf.io.gfile.exists(
            os.path.join(handler_pipeline_path, 'pipeline_args.json')))
    self.assertIn('Pipeline {} created successfully.'.format(pipeline_name),
                  result.output)

  def test_pipeline_create(self):
    # Create a pipeline.
    pipeline_path = os.path.join(self.chicago_taxi_pipeline_dir,
                                 'test_pipeline_beam_1.py')
    pipeline_name = 'chicago_taxi_beam'
    self._valid_create_and_check(pipeline_path, pipeline_name)

    # Test pipeline create when pipeline already exists.
    result = self.runner.invoke(cli_group, [
        'pipeline', 'create', '--engine', 'beam', '--pipeline_path',
        pipeline_path
    ])
    self.assertIn('CLI', result.output)
    self.assertIn('Creating pipeline', result.output)
    self.assertTrue('Pipeline {} already exists.'.format(pipeline_name),
                    result.output)

  def test_pipeline_update(self):
    pipeline_name = 'chicago_taxi_beam'
    handler_pipeline_path = os.path.join(os.environ['BEAM_HOME'], pipeline_name)
    pipeline_path_1 = os.path.join(self.chicago_taxi_pipeline_dir,
                                   'test_pipeline_beam_1.py')
    # Try pipeline update when pipeline does not exist.
    result = self.runner.invoke(cli_group, [
        'pipeline', 'update', '--engine', 'beam', '--pipeline_path',
        pipeline_path_1
    ])
    self.assertIn('CLI', result.output)
    self.assertIn('Updating pipeline', result.output)
    self.assertIn('Pipeline {} does not exist.'.format(pipeline_name),
                  result.output)
    self.assertFalse(tf.io.gfile.exists(handler_pipeline_path))

    # Now update an existing pipeline.
    self._valid_create_and_check(pipeline_path_1, pipeline_name)

    pipeline_path_2 = os.path.join(self.chicago_taxi_pipeline_dir,
                                   'test_pipeline_beam_2.py')
    result = self.runner.invoke(cli_group, [
        'pipeline', 'update', '--engine', 'beam', '--pipeline_path',
        pipeline_path_2
    ])
    self.assertIn('CLI', result.output)
    self.assertIn('Updating pipeline', result.output)
    self.assertIn('Pipeline {} updated successfully.'.format(pipeline_name),
                  result.output)
    self.assertTrue(
        tf.io.gfile.exists(
            os.path.join(handler_pipeline_path, 'pipeline_args.json')))

  def test_pipeline_compile(self):
    pipeline_path = os.path.join(self.chicago_taxi_pipeline_dir,
                                 'test_pipeline_beam_2.py')
    result = self.runner.invoke(cli_group, [
        'pipeline', 'compile', '--engine', 'beam', '--pipeline_path',
        pipeline_path
    ])
    self.assertIn('CLI', result.output)
    self.assertIn('Compiling pipeline', result.output)
    self.assertIn('Pipeline compiled successfully', result.output)

  def test_pipeline_delete(self):
    pipeline_path = os.path.join(self.chicago_taxi_pipeline_dir,
                                 'test_pipeline_beam_1.py')
    pipeline_name = 'chicago_taxi_beam'
    handler_pipeline_path = os.path.join(os.environ['BEAM_HOME'], pipeline_name)

    # Try deleting a non existent pipeline.
    result = self.runner.invoke(cli_group, [
        'pipeline', 'delete', '--engine', 'beam', '--pipeline_name',
        pipeline_name
    ])
    self.assertIn('CLI', result.output)
    self.assertIn('Deleting pipeline', result.output)
    self.assertIn('Pipeline {} does not exist.'.format(pipeline_name),
                  result.output)
    self.assertFalse(tf.io.gfile.exists(handler_pipeline_path))

    # Create a pipeline.
    self._valid_create_and_check(pipeline_path, pipeline_name)

    # Now delete the pipeline.
    result = self.runner.invoke(cli_group, [
        'pipeline', 'delete', '--engine', 'beam', '--pipeline_name',
        pipeline_name
    ])
    self.assertIn('CLI', result.output)
    self.assertIn('Deleting pipeline', result.output)
    self.assertFalse(tf.io.gfile.exists(handler_pipeline_path))
    self.assertIn('Pipeline {} deleted successfully.'.format(pipeline_name),
                  result.output)

  def test_pipeline_list(self):

    # Try listing pipelines when there are none.
    result = self.runner.invoke(cli_group,
                                ['pipeline', 'list', '--engine', 'beam'])
    self.assertIn('CLI', result.output)
    self.assertIn('Listing all pipelines', result.output)
    self.assertIn('No pipelines to display.', result.output)

    # Create pipelines.
    pipeline_name_1 = 'chicago_taxi_beam'
    pipeline_path_1 = os.path.join(self.chicago_taxi_pipeline_dir,
                                   'test_pipeline_beam_1.py')
    self._valid_create_and_check(pipeline_path_1, pipeline_name_1)

    pipeline_name_2 = 'chicago_taxi_beam_v2'
    pipeline_path_2 = os.path.join(self.chicago_taxi_pipeline_dir,
                                   'test_pipeline_beam_3.py')
    self._valid_create_and_check(pipeline_path_2, pipeline_name_2)

    # List pipelines.
    result = self.runner.invoke(cli_group,
                                ['pipeline', 'list', '--engine', 'beam'])
    self.assertIn('CLI', result.output)
    self.assertIn('Listing all pipelines', result.output)
    self.assertIn(pipeline_name_1, result.output)
    self.assertIn(pipeline_name_2, result.output)

  def test_run_create(self):
    # Create a pipeline first.
    pipeline_name_1 = 'chicago_taxi_beam'
    pipeline_path_1 = os.path.join(self.chicago_taxi_pipeline_dir,
                                   'test_pipeline_beam_1.py')
    self._valid_create_and_check(pipeline_path_1, pipeline_name_1)

    # Now run a different pipeline
    pipeline_name_2 = 'chicago_taxi_beam_v2'
    result = self.runner.invoke(cli_group, [
        'run', 'create', '--engine', 'beam', '--pipeline_name', pipeline_name_2
    ])
    self.assertIn('CLI', result.output)
    self.assertIn('Creating a run for pipeline: {}'.format(pipeline_name_2),
                  result.output)
    self.assertIn('Pipeline {} does not exist.'.format(pipeline_name_2),
                  result.output)

    # Now run the pipeline
    result = self.runner.invoke(cli_group, [
        'run', 'create', '--engine', 'beam', '--pipeline_name', pipeline_name_2
    ])
    self.assertIn('CLI', result.output)
    self.assertIn('Creating a run for pipeline: {}'.format(pipeline_name_2),
                  result.output)


if __name__ == '__main__':
  tf.test.main()
