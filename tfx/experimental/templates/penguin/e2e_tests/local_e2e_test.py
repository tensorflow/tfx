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
class PenguinTemplateLocalEndToEndTest(test_utils.BaseLocalEndToEndTest):
  """This test runs all components in the template."""

  def setUp(self):
    super().setUp()
    self._pipeline_name = 'PENGUIN_TEMPLATE_E2E_TEST'

  def testGeneratedUnitTests(self):
    self._copyTemplate('penguin')
    for m in self._getAllUnitTests():
      logging.info('Running unit test "%s"', m)
      # A failed googletest will raise a CalledProcessError.
      _ = subprocess.check_output([sys.executable, '-m', m])

  def testLocalPipeline(self):
    self._copyTemplate('penguin')
    os.environ['LOCAL_HOME'] = os.path.join(self._temp_dir, 'local')

    # Create a pipeline with only one component.
    self._create_pipeline()
    self._run_pipeline()

    self._copy_schema()

    # Update the pipeline to include all components.
    updated_pipeline_file = self._addAllComponents()
    logging.info('Updated %s to add all components to the pipeline.',
                 updated_pipeline_file)

    # Update the pipeline to use ImportSchemaGen
    self._uncomment('local_runner.py', ['schema_path=generated_schema_path'])
    self._replaceFileContent(
        'local_runner.py',
        [('schema_path=generated_schema_path', 'schema_path=\'schema.pbtxt\'')])

    # Lowers required threshold to make tests stable.
    self._replaceFileContent(
        os.path.join('pipeline', 'configs.py'), [
            ('EVAL_ACCURACY_THRESHOLD = 0.6', 'EVAL_ACCURACY_THRESHOLD = 0.1'),
        ])

    logging.info(
        'Updated pipeline to add all components and use user provided schema.')
    self._update_pipeline()
    self._run_pipeline()


if __name__ == '__main__':
  tf.test.main()
