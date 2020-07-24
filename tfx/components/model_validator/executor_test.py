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
"""Tests for tfx.components.model_validator.executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from tfx.components.model_validator import constants
from tfx.components.model_validator import executor
from tfx.types import artifact_utils
from tfx.types import standard_artifacts


class ExecutorTest(tf.test.TestCase):

  def setUp(self):
    super(ExecutorTest, self).setUp()
    self._source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')
    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    self.component_id = 'test_component'

    # Create input dict.
    eval_examples = standard_artifacts.Examples()
    eval_examples.split_names = artifact_utils.encode_split_names(['eval'])
    eval_examples.uri = os.path.join(self._source_data_dir, 'csv_example_gen')
    model = standard_artifacts.Model()
    model.uri = os.path.join(self._source_data_dir, 'trainer/current')
    self._input_dict = {
        constants.EXAMPLES_KEY: [eval_examples],
        constants.MODEL_KEY: [model],
    }

    # Create output dict.
    self._blessing = standard_artifacts.ModelBlessing()
    self._blessing.uri = os.path.join(output_data_dir, 'blessing')
    self._output_dict = {constants.BLESSING_KEY: [self._blessing]}

    # Create context
    self._tmp_dir = os.path.join(output_data_dir, '.temp')
    self._context = executor.Executor.Context(tmp_dir=self._tmp_dir,
                                              unique_id='2')

  def testDoWithBlessedModel(self):
    # Create exe properties.
    exec_properties = {
        'blessed_model': os.path.join(self._source_data_dir, 'trainer/blessed'),
        'blessed_model_id': 123,
        'current_component_id': self.component_id,
    }

    # Run executor.
    model_validator = executor.Executor(self._context)
    model_validator.Do(self._input_dict, self._output_dict, exec_properties)

    # Check model validator outputs.
    self.assertTrue(tf.io.gfile.exists(os.path.join(self._tmp_dir)))
    self.assertTrue(
        tf.io.gfile.exists(
            os.path.join(self._blessing.uri, constants.BLESSED_FILE_NAME)))

  def testDoWithoutBlessedModel(self):
    # Create exe properties.
    exec_properties = {
        'blessed_model': None,
        'blessed_model_id': None,
        'current_component_id': self.component_id,
    }

    # Run executor.
    model_validator = executor.Executor(self._context)
    model_validator.Do(self._input_dict, self._output_dict, exec_properties)

    # Check model validator outputs.
    self.assertTrue(tf.io.gfile.exists(os.path.join(self._tmp_dir)))
    self.assertTrue(
        tf.io.gfile.exists(
            os.path.join(self._blessing.uri, constants.BLESSED_FILE_NAME)))


if __name__ == '__main__':
  tf.test.main()
