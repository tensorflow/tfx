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
"""E2E test using local orchestrator for penguin template."""

import os
import subprocess
import sys
import unittest

from absl import logging
import tensorflow as tf

from tfx.experimental.templates import test_utils


@unittest.skipIf(tf.__version__ < '2',
                 'Uses keras Model only compatible with TF 2.x')
class PenguinTemplateLocalEndToEndTest(test_utils.BaseEndToEndTest):
  """This test runs all components in the template."""

  def setUp(self):
    super().setUp()
    self._pipeline_name = 'PENGUIN_TEMPLATE_E2E_TEST'

  def _getAllUnitTests(self):
    for root, _, files in os.walk(self._project_dir):
      base_dir = os.path.relpath(root, self._project_dir)
      if base_dir == '.':  # project_dir == root
        base_module = ''
      else:
        base_module = base_dir.replace(os.path.sep, '.') + '.'

      for filename in files:
        if filename.endswith('_test.py'):
          yield base_module + filename[:-3]

  def testGeneratedUnitTests(self):
    self._copyTemplate('penguin')
    for m in self._getAllUnitTests():
      logging.info('Running unit test "%s"', m)
      # A failed googletest will raise a CalledProcessError.
      _ = subprocess.check_output([sys.executable, '-m', m])

  def testLocalPipeline(self):
    self._copyTemplate('penguin')
    os.environ['LOCAL_HOME'] = os.path.join(self._temp_dir, 'local')

    # Create a pipeline with only initial components.
    result = self._runCli([
        'pipeline',
        'create',
        '--engine',
        'local',
        '--pipeline_path',
        'local_runner.py',
    ])
    self.assertIn(
        'Pipeline "{}" created successfully.'.format(self._pipeline_name),
        result)

    # Run the pipeline.
    self._runCli([
        'run',
        'create',
        '--engine',
        'local',
        '--pipeline_name',
        self._pipeline_name,
    ])

    # Update the pipeline to include all components.
    updated_pipeline_file = self._addAllComponents()
    logging.info('Updated %s to add all components to the pipeline.',
                 updated_pipeline_file)
    # Lowers required threshold to make tests stable.
    self._replaceFileContent(
        os.path.join('pipeline', 'configs.py'), [
            ('EVAL_ACCURACY_THRESHOLD = 0.6', 'EVAL_ACCURACY_THRESHOLD = 0.1'),
        ])
    result = self._runCli([
        'pipeline',
        'update',
        '--engine',
        'local',
        '--pipeline_path',
        'local_runner.py',
    ])
    self.assertIn(
        'Pipeline "{}" updated successfully.'.format(self._pipeline_name),
        result)

    # Run the updated pipeline.
    self._runCli([
        'run',
        'create',
        '--engine',
        'local',
        '--pipeline_name',
        self._pipeline_name,
    ])


if __name__ == '__main__':
  tf.test.main()
