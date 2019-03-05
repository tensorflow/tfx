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
"""Tests for tfx.components.evaluator.executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
# TODO(jyzhao): BucketizeWithInputBoundaries error without this.
from tensorflow.contrib.boosted_trees.python.ops import quantile_ops  # pylint: disable=unused-import
from tfx.components.evaluator import executor
from tfx.proto import evaluator_pb2
from tfx.utils import types
from google.protobuf import json_format


class ExecutorTest(tf.test.TestCase):

  def test_do(self):
    source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')
    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    # Create input dict.
    train_examples = types.TfxType(type_name='ExamplesPath', split='train')
    eval_examples = types.TfxType(type_name='ExamplesPath', split='eval')
    eval_examples.uri = os.path.join(source_data_dir, 'csv_example_gen/eval/')
    model_exports = types.TfxType(type_name='ModelExportPath')
    model_exports.uri = os.path.join(source_data_dir, 'trainer/current/')
    input_dict = {
        'examples': [train_examples, eval_examples],
        'model_exports': [model_exports],
    }

    # Create output dict.
    eval_output = types.TfxType('ModelEvalPath')
    eval_output.uri = os.path.join(output_data_dir, 'eval_output')
    output_dict = {'output': [eval_output]}

    # Create exec proterties.
    exec_properties = {
        'feature_slicing_spec':
            json_format.MessageToJson(
                evaluator_pb2.FeatureSlicingSpec(specs=[
                    evaluator_pb2.SingleSlicingSpec(
                        column_for_slicing=['trip_start_hour']),
                    evaluator_pb2.SingleSlicingSpec(
                        column_for_slicing=['trip_start_day', 'trip_miles']),
                ]))
    }

    # Run executor.
    evaluator = executor.Executor()
    evaluator.Do(input_dict, output_dict, exec_properties)

    # Check evaluator outputs.
    self.assertTrue(
        tf.gfile.Exists(os.path.join(eval_output.uri, 'eval_config')))
    self.assertTrue(tf.gfile.Exists(os.path.join(eval_output.uri, 'metrics')))
    self.assertTrue(tf.gfile.Exists(os.path.join(eval_output.uri, 'plots')))


if __name__ == '__main__':
  tf.test.main()
