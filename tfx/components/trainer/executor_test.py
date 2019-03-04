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
"""Tests for tfx.components.trainer.executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import apache_beam as beam
import tensorflow as tf

from tfx.components.trainer import executor
from tfx.proto import trainer_pb2
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
    train_examples.uri = os.path.join(source_data_dir,
                                      'transform/transformed_examples/train/')
    eval_examples = types.TfxType(type_name='ExamplesPath', split='eval')
    eval_examples.uri = os.path.join(source_data_dir,
                                     'transform/transformed_examples/eval/')
    transform_output = types.TfxType(type_name='TransformPath')
    transform_output.uri = os.path.join(source_data_dir,
                                        'transform/transform_output/')
    schema = types.TfxType(type_name='ExamplesPath')
    schema.uri = os.path.join(source_data_dir, 'schema_gen/')

    input_dict = {
        'transformed_examples': [train_examples, eval_examples],
        'transform_output': [transform_output],
        'schema': [schema],
    }

    # Create output dict.
    model_exports = types.TfxType(type_name='ModelExportPath')
    model_exports.uri = os.path.join(output_data_dir, 'model_export_path')
    output_dict = {'output': [model_exports]}

    # Create exec properties.
    module_file_path = os.path.join(source_data_dir, 'module_file',
                                    'trainer_module.py')

    exec_properties = {
        'train_args':
            json_format.MessageToJson(trainer_pb2.TrainArgs(num_steps=1000)),
        'eval_args':
            json_format.MessageToJson(trainer_pb2.EvalArgs(num_steps=500)),
        'module_file':
            module_file_path,
        'warm_starting':
            False,
    }

    # Run executor.
    pipeline = beam.Pipeline()
    evaluator = executor.Executor(pipeline)
    evaluator.Do(
        input_dict=input_dict,
        output_dict=output_dict,
        exec_properties=exec_properties)

    # Check outputs.
    self.assertTrue(
        tf.gfile.Exists(os.path.join(model_exports.uri, 'eval_model_dir')))
    self.assertTrue(
        tf.gfile.Exists(os.path.join(model_exports.uri, 'serving_model_dir')))


if __name__ == '__main__':
  tf.test.main()
