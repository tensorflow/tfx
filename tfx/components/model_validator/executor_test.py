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
# TODO(jyzhao): BucketizeWithInputBoundaries error without this.
from tensorflow.contrib.boosted_trees.python.ops import quantile_ops  # pylint: disable=unused-import
from tfx import types
from tfx.components.model_validator import executor


class ExecutorTest(tf.test.TestCase):

  def setUp(self):
    self._source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')
    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    self.component_id = 'test_component'

    # Create input dict.
    eval_examples = types.Artifact(type_name='ExamplesPath', split='eval')
    eval_examples.uri = os.path.join(self._source_data_dir,
                                     'csv_example_gen/eval/')
    model = types.Artifact(type_name='ModelExportPath')
    model.uri = os.path.join(self._source_data_dir, 'trainer/current/')
    self._input_dict = {
        'examples': [eval_examples],
        'model': [model],
    }

    # Create output dict.
    self._blessing = types.Artifact('ModelBlessingPath')
    self._blessing.uri = os.path.join(output_data_dir, 'blessing')
    self._output_dict = {
        'blessing': [self._blessing]
    }

    # Create context
    self._tmp_dir = os.path.join(output_data_dir, '.temp')
    self._context = executor.Executor.Context(tmp_dir=self._tmp_dir,
                                              unique_id='2')

  def test_do_with_blessed_model(self):
    # Create exe properties.
    exec_properties = {
        'blessed_model':
            os.path.join(self._source_data_dir, 'trainer/blessed/'),
        'blessed_model_id':
            123,
        'component_id':
            self.component_id,
    }

    # Run executor.
    model_validator = executor.Executor(self._context)
    model_validator.Do(self._input_dict, self._output_dict, exec_properties)

    # Check model validator outputs.
    self.assertTrue(
        tf.gfile.Exists(os.path.join(self._tmp_dir)))
    self.assertTrue(
        tf.gfile.Exists(os.path.join(self._blessing.uri, 'BLESSED')))

  def test_do_without_blessed_model(self):
    # Create exe properties.
    exec_properties = {
        'blessed_model': None,
        'blessed_model_id': None,
        'component_id': self.component_id,
    }

    # Run executor.
    model_validator = executor.Executor(self._context)
    model_validator.Do(self._input_dict, self._output_dict, exec_properties)

    # Check model validator outputs.
    self.assertTrue(
        tf.gfile.Exists(os.path.join(self._tmp_dir)))
    self.assertTrue(
        tf.gfile.Exists(os.path.join(self._blessing.uri, 'BLESSED')))


if __name__ == '__main__':
  tf.test.main()
