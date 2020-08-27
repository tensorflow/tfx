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
"""E2E test using Beam orchestrator for taxi template."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess
import sys

from absl import logging
import tensorflow as tf

from tfx.experimental.templates.taxi.e2e_tests import test_utils


class TaxiTemplateBeamEndToEndTest(test_utils.BaseEndToEndTest):
  """This test covers step 1~6 of the accompanying document[1] for taxi template.

  [1]https://github.com/tensorflow/tfx/blob/master/docs/tutorials/tfx/template.ipynb
  """

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
    self._copyTemplate()
    for m in self._getAllUnitTests():
      logging.info('Running unit test "%s"', m)
      # A failed googletest will raise a CalledProcessError.
      _ = subprocess.check_output([sys.executable, '-m', m])

  def testBeamPipeline(self):
    self._copyTemplate()
    os.environ['BEAM_HOME'] = os.path.join(self._temp_dir, 'beam')

    # Create a pipeline with only one component.
    result = self._runCli([
        'pipeline',
        'create',
        '--engine',
        'beam',
        '--pipeline_path',
        'beam_dag_runner.py',
    ])
    self.assertEqual(0, result.exit_code)
    self.assertIn(
        'Pipeline "{}" created successfully.'.format(self._pipeline_name),
        result.output)

    # Run the pipeline.
    result = self._runCli([
        'run',
        'create',
        '--engine',
        'beam',
        '--pipeline_name',
        self._pipeline_name,
    ])
    self.assertEqual(0, result.exit_code)

    # Update the pipeline to include all components.
    updated_pipeline_file = self._addAllComponents()
    logging.info('Updated %s to add all components to the pipeline.',
                 updated_pipeline_file)
    result = self._runCli([
        'pipeline',
        'update',
        '--engine',
        'beam',
        '--pipeline_path',
        'beam_dag_runner.py',
    ])
    self.assertEqual(0, result.exit_code)
    self.assertIn(
        'Pipeline "{}" updated successfully.'.format(self._pipeline_name),
        result.output)

    # Run the updated pipeline.
    result = self._runCli([
        'run',
        'create',
        '--engine',
        'beam',
        '--pipeline_name',
        self._pipeline_name,
    ])
    self.assertEqual(0, result.exit_code)


if __name__ == '__main__':
  tf.test.main()
