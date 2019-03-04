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
"""Tests for tfx.components.statistics_gen.executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import tensorflow_data_validation as tfdv
from tfx.components.statistics_gen import executor
from tfx.utils import types


class ExecutorTest(tf.test.TestCase):

  def _validate_stats_output(self, stats_path):
    self.assertTrue(tf.gfile.Exists(stats_path))
    stats = tfdv.load_statistics(stats_path)
    self.assertEqual(1, len(stats.datasets))
    data_set = stats.datasets[0]
    self.assertGreater(data_set.num_examples, 0)
    self.assertNotEqual(0, len(data_set.features))
    # TODO(b/126245422): verify content of generated stats after we have stable
    # test data set.

  def test_do(self):
    source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')
    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    tf.gfile.MakeDirs(output_data_dir)

    # Create input dict.
    train_examples = types.TfxType(type_name='ExamplesPath', split='train')
    train_examples.uri = os.path.join(source_data_dir, 'csv_example_gen/train/')
    eval_examples = types.TfxType(type_name='ExamplesPath', split='eval')
    eval_examples.uri = os.path.join(source_data_dir, 'csv_example_gen/eval/')

    train_stats = types.TfxType(
        type_name='ExampleStatisticsPath', split='train')
    train_stats.uri = os.path.join(output_data_dir, 'train', '')
    eval_stats = types.TfxType(type_name='ExampleStatisticsPath', split='eval')
    eval_stats.uri = os.path.join(output_data_dir, 'eval', '')
    input_dict = {
        'input_data': [train_examples, eval_examples],
    }

    output_dict = {
        'output': [train_stats, eval_stats],
    }

    # Run executor.
    evaluator = executor.Executor()
    evaluator.Do(input_dict, output_dict, exec_properties={})

    # Check statistics_gen outputs.
    self._validate_stats_output(os.path.join(train_stats.uri, 'stats_tfrecord'))
    self._validate_stats_output(os.path.join(eval_stats.uri, 'stats_tfrecord'))


if __name__ == '__main__':
  tf.test.main()
