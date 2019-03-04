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
"""Tests for tfx.components.example_gen.csv_example_gen.executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tfx.components.example_gen.csv_example_gen import executor
from tfx.utils import types


class ExecutorTest(tf.test.TestCase):

  def test_do(self):
    input_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'testdata')
    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    # Create input dict.
    input_base = types.TfxType(type_name='ExternalPath')
    input_base.uri = os.path.join(input_data_dir, 'external/csv/')
    input_dict = {'input-base': [input_base]}

    # Create output dict.
    train_examples = types.TfxType(type_name='ExamplesPath', split='train')
    train_examples.uri = os.path.join(output_data_dir, 'train')
    eval_examples = types.TfxType(type_name='ExamplesPath', split='eval')
    eval_examples.uri = os.path.join(output_data_dir, 'eval')
    output_dict = {'examples': [train_examples, eval_examples]}

    # Run executor.
    csv_example_gen = executor.Executor()
    csv_example_gen.Do(input_dict, output_dict, {})

    # Check CSV example gen outputs.
    train_output_file = os.path.join(train_examples.uri,
                                     'data_tfrecord-00000-of-00001.gz')
    eval_output_file = os.path.join(eval_examples.uri,
                                    'data_tfrecord-00000-of-00001.gz')
    self.assertTrue(tf.gfile.Exists(train_output_file))
    self.assertTrue(tf.gfile.Exists(eval_output_file))
    self.assertGreater(
        tf.gfile.GFile(train_output_file).size(),
        tf.gfile.GFile(eval_output_file).size())


if __name__ == '__main__':
  tf.test.main()
