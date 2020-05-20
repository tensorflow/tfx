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
"""Tests for tfx.components.trainer.executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

# Standard Imports
import mock
import tensorflow as tf

from google.protobuf import json_format
from tfx.components.testdata.module_file import trainer_module
from tfx.components.trainer import constants
from tfx.components.trainer import executor
from tfx.proto import trainer_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.utils import io_utils
from tfx.utils import path_utils


class ExecutorTest(tf.test.TestCase):

  def setUp(self):
    super(ExecutorTest, self).setUp()
    self._source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')
    self._output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    # Create input dict.
    examples = standard_artifacts.Examples()
    examples.uri = os.path.join(self._source_data_dir,
                                'transform/transformed_examples')
    examples.split_names = artifact_utils.encode_split_names(['train', 'eval'])
    transform_output = standard_artifacts.TransformGraph()
    transform_output.uri = os.path.join(self._source_data_dir,
                                        'transform/transform_graph')
    schema = standard_artifacts.Schema()
    schema.uri = os.path.join(self._source_data_dir, 'schema_gen')
    previous_model = standard_artifacts.Model()
    previous_model.uri = os.path.join(self._source_data_dir, 'trainer/previous')

    self._input_dict = {
        constants.EXAMPLES_KEY: [examples],
        constants.TRANSFORM_GRAPH_KEY: [transform_output],
        constants.SCHEMA_KEY: [schema],
        constants.BASE_MODEL_KEY: [previous_model]
    }

    # Create output dict.
    self._model_exports = standard_artifacts.Model()
    self._model_exports.uri = os.path.join(self._output_data_dir,
                                           'model_export_path')
    self._output_dict = {constants.OUTPUT_MODEL_KEY: [self._model_exports]}

    # Create exec properties skeleton.
    self._exec_properties = {
        'train_args':
            json_format.MessageToJson(
                trainer_pb2.TrainArgs(num_steps=1000),
                preserving_proto_field_name=True),
        'eval_args':
            json_format.MessageToJson(
                trainer_pb2.EvalArgs(num_steps=500),
                preserving_proto_field_name=True),
        'warm_starting':
            False,
    }

    self._module_file = os.path.join(self._source_data_dir, 'module_file',
                                     'trainer_module.py')
    self._trainer_fn = '%s.%s' % (trainer_module.trainer_fn.__module__,
                                  trainer_module.trainer_fn.__name__)

    # Executors for test.
    self._trainer_executor = executor.Executor()
    self._generic_trainer_executor = executor.GenericExecutor()

  def _verify_model_exports(self):
    self.assertTrue(
        tf.io.gfile.exists(path_utils.eval_model_dir(self._model_exports.uri)))
    self.assertTrue(
        tf.io.gfile.exists(
            path_utils.serving_model_dir(self._model_exports.uri)))

  def _verify_no_eval_model_exports(self):
    self.assertFalse(
        tf.io.gfile.exists(path_utils.eval_model_dir(self._model_exports.uri)))

  def _do(self, test_executor):
    test_executor.Do(
        input_dict=self._input_dict,
        output_dict=self._output_dict,
        exec_properties=self._exec_properties)

  def testGenericExecutor(self):
    self._exec_properties['module_file'] = self._module_file
    self._do(self._generic_trainer_executor)
    self._verify_model_exports()

  @mock.patch('tfx.components.trainer.executor._is_chief')
  def testDoChief(self, mock_is_chief):
    mock_is_chief.return_value = True
    self._exec_properties['module_file'] = self._module_file
    self._do(self._trainer_executor)
    self._verify_model_exports()

  @mock.patch('tfx.components.trainer.executor._is_chief')
  def testDoNonChief(self, mock_is_chief):
    mock_is_chief.return_value = False
    self._exec_properties['module_file'] = self._module_file
    self._do(self._trainer_executor)
    self._verify_no_eval_model_exports()

  def testDoWithModuleFile(self):
    self._exec_properties['module_file'] = self._module_file
    self._do(self._trainer_executor)
    self._verify_model_exports()

  def testDoWithTrainerFn(self):
    self._exec_properties['trainer_fn'] = self._trainer_fn
    self._do(self._trainer_executor)
    self._verify_model_exports()

  def testDoWithNoTrainerFn(self):
    with self.assertRaises(ValueError):
      self._do(self._trainer_executor)

  def testDoWithDuplicateTrainerFn(self):
    self._exec_properties['module_file'] = self._module_file
    self._exec_properties['trainer_fn'] = self._trainer_fn
    with self.assertRaises(ValueError):
      self._do(self._trainer_executor)

  def testDoWithHyperParameters(self):
    hp_artifact = standard_artifacts.HyperParameters()
    hp_artifact.uri = os.path.join(self._output_data_dir, 'hyperparameters/')

    # TODO(jyzhao): use real kerastuner.HyperParameters instead of dict.
    hyperparameters = {}
    hyperparameters['first_dnn_layer_size'] = 100
    hyperparameters['num_dnn_layers'] = 4
    hyperparameters['dnn_decay_factor'] = 0.7
    io_utils.write_string_file(
        os.path.join(hp_artifact.uri, 'hyperparameters.txt'),
        json.dumps(hyperparameters))

    self._input_dict[constants.HYPERPARAMETERS_KEY] = [hp_artifact]

    self._exec_properties['module_file'] = self._module_file
    self._do(self._trainer_executor)
    self._verify_model_exports()


if __name__ == '__main__':
  tf.test.main()
