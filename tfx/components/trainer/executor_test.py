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
import tensorflow as tf

from tfx.components.testdata.module_file import trainer_module
from tfx.components.trainer import executor
from tfx.proto import trainer_pb2
from tfx.types import standard_artifacts
from google.protobuf import json_format


class ExecutorTest(tf.test.TestCase):

  def setUp(self):
    super(ExecutorTest, self).setUp()
    self._source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')
    self._output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    # Create input dict.
    train_examples = standard_artifacts.Examples(split='train')
    train_examples.uri = os.path.join(self._source_data_dir,
                                      'transform/transformed_examples/train/')
    eval_examples = standard_artifacts.Examples(split='eval')
    eval_examples.uri = os.path.join(self._source_data_dir,
                                     'transform/transformed_examples/eval/')
    transform_output = standard_artifacts.TransformGraph()
    transform_output.uri = os.path.join(self._source_data_dir,
                                        'transform/transform_output/')
    schema = standard_artifacts.Examples()
    schema.uri = os.path.join(self._source_data_dir, 'schema_gen/')

    self._input_dict = {
        'examples': [train_examples, eval_examples],
        'transform_output': [transform_output],
        'schema': [schema],
    }

    # Create output dict.
    self._model_exports = standard_artifacts.Model()
    self._model_exports.uri = os.path.join(self._output_data_dir,
                                           'model_export_path')
    self._output_dict = {'output': [self._model_exports]}

    # Create exec properties skeleton.
    self._exec_properties = {
        'train_args':
            json_format.MessageToJson(trainer_pb2.TrainArgs(num_steps=1000)),
        'eval_args':
            json_format.MessageToJson(trainer_pb2.EvalArgs(num_steps=500)),
        'warm_starting':
            False,
    }

    self._module_file = os.path.join(self._source_data_dir, 'module_file',
                                     'trainer_module.py')
    self._trainer_fn = '%s.%s' % (trainer_module.trainer_fn.__module__,
                                  trainer_module.trainer_fn.__name__)

    # Executor for test.
    self._trainer_executor = executor.Executor()

  def _verify_model_exports(self):
    self.assertTrue(
        tf.io.gfile.exists(
            os.path.join(self._model_exports.uri, 'eval_model_dir')))
    self.assertTrue(
        tf.io.gfile.exists(
            os.path.join(self._model_exports.uri, 'serving_model_dir')))

  def testDoWithModuleFile(self):
    self._exec_properties['module_file'] = self._module_file
    self._trainer_executor.Do(
        input_dict=self._input_dict,
        output_dict=self._output_dict,
        exec_properties=self._exec_properties)
    self._verify_model_exports()

  def testDoWithTrainerFn(self):
    self._exec_properties['trainer_fn'] = self._trainer_fn
    self._trainer_executor.Do(
        input_dict=self._input_dict,
        output_dict=self._output_dict,
        exec_properties=self._exec_properties)
    self._verify_model_exports()

  def testDoWithNoTrainerFn(self):
    with self.assertRaises(ValueError):
      self._trainer_executor.Do(
          input_dict=self._input_dict,
          output_dict=self._output_dict,
          exec_properties=self._exec_properties)

  def testDoWithDuplicateTrainerFn(self):
    self._exec_properties['module_file'] = self._module_file
    self._exec_properties['trainer_fn'] = self._trainer_fn
    with self.assertRaises(ValueError):
      self._trainer_executor.Do(
          input_dict=self._input_dict,
          output_dict=self._output_dict,
          exec_properties=self._exec_properties)


if __name__ == '__main__':
  tf.test.main()
