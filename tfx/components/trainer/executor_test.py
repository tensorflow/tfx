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

import copy
import json
import os

from unittest import mock
import tensorflow as tf
from tfx.components.testdata.module_file import trainer_module
from tfx.components.trainer import executor
from tfx.dsl.io import fileio
from tfx.proto import trainer_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs
from tfx.utils import io_utils
from tfx.utils import path_utils
from tfx.utils import proto_utils


class ExecutorTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')
    self._output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    # Create input dict.
    e1 = standard_artifacts.Examples()
    e1.uri = os.path.join(self._source_data_dir,
                          'transform/transformed_examples')
    e1.split_names = artifact_utils.encode_split_names(['train', 'eval'])

    e2 = copy.deepcopy(e1)

    self._single_artifact = [e1]
    self._multiple_artifacts = [e1, e2]

    transform_graph = standard_artifacts.TransformGraph()
    transform_graph.uri = os.path.join(self._source_data_dir,
                                       'transform/transform_graph')

    schema = standard_artifacts.Schema()
    schema.uri = os.path.join(self._source_data_dir, 'schema_gen')
    previous_model = standard_artifacts.Model()
    previous_model.uri = os.path.join(self._source_data_dir, 'trainer/previous')

    self._input_dict = {
        standard_component_specs.EXAMPLES_KEY: self._single_artifact,
        standard_component_specs.TRANSFORM_GRAPH_KEY: [transform_graph],
        standard_component_specs.SCHEMA_KEY: [schema],
        standard_component_specs.BASE_MODEL_KEY: [previous_model]
    }

    # Create output dict.
    self._model_exports = standard_artifacts.Model()
    self._model_exports.uri = os.path.join(self._output_data_dir,
                                           'model_export_path')
    self._model_run_exports = standard_artifacts.ModelRun()
    self._model_run_exports.uri = os.path.join(self._output_data_dir,
                                               'model_run_path')
    self._output_dict = {
        standard_component_specs.MODEL_KEY: [self._model_exports],
        standard_component_specs.MODEL_RUN_KEY: [self._model_run_exports]
    }

    # Create exec properties skeleton.
    self._exec_properties = {
        standard_component_specs.TRAIN_ARGS_KEY:
            proto_utils.proto_to_json(trainer_pb2.TrainArgs(num_steps=1000)),
        standard_component_specs.EVAL_ARGS_KEY:
            proto_utils.proto_to_json(trainer_pb2.EvalArgs(num_steps=500)),
        'warm_starting':
            False,
    }

    self._module_file = os.path.join(self._source_data_dir,
                                     standard_component_specs.MODULE_FILE_KEY,
                                     'trainer_module.py')
    self._trainer_fn = '%s.%s' % (trainer_module.trainer_fn.__module__,
                                  trainer_module.trainer_fn.__name__)

    # Executors for test.
    self._trainer_executor = executor.Executor()
    self._generic_trainer_executor = executor.GenericExecutor()

  def _verify_model_exports(self):
    self.assertTrue(
        fileio.exists(path_utils.eval_model_dir(self._model_exports.uri)))
    self.assertTrue(
        fileio.exists(path_utils.serving_model_dir(self._model_exports.uri)))

  def _verify_no_eval_model_exports(self):
    self.assertFalse(
        fileio.exists(path_utils.eval_model_dir(self._model_exports.uri)))

  def _verify_model_run_exports(self):
    self.assertTrue(fileio.exists(os.path.dirname(self._model_run_exports.uri)))

  def _do(self, test_executor):
    test_executor.Do(
        input_dict=self._input_dict,
        output_dict=self._output_dict,
        exec_properties=self._exec_properties)

  def testGenericExecutor(self):
    self._exec_properties[
        standard_component_specs.MODULE_FILE_KEY] = self._module_file
    self._do(self._generic_trainer_executor)
    self._verify_model_exports()
    self._verify_model_run_exports()

  @mock.patch('tfx.components.trainer.executor._is_chief')
  def testDoChief(self, mock_is_chief):
    mock_is_chief.return_value = True
    self._exec_properties[
        standard_component_specs.MODULE_FILE_KEY] = self._module_file
    self._do(self._trainer_executor)
    self._verify_model_exports()
    self._verify_model_run_exports()

  @mock.patch('tfx.components.trainer.executor._is_chief')
  def testDoNonChief(self, mock_is_chief):
    mock_is_chief.return_value = False
    self._exec_properties[
        standard_component_specs.MODULE_FILE_KEY] = self._module_file
    self._do(self._trainer_executor)
    self._verify_no_eval_model_exports()
    self._verify_model_run_exports()

  def testDoWithModuleFile(self):
    self._exec_properties[
        standard_component_specs.MODULE_FILE_KEY] = self._module_file
    self._do(self._trainer_executor)
    self._verify_model_exports()
    self._verify_model_run_exports()

  def testDoWithTrainerFn(self):
    self._exec_properties[
        standard_component_specs.TRAINER_FN_KEY] = self._trainer_fn
    self._do(self._trainer_executor)
    self._verify_model_exports()
    self._verify_model_run_exports()

  def testDoWithNoTrainerFn(self):
    with self.assertRaises(ValueError):
      self._do(self._trainer_executor)

  def testDoWithHyperParameters(self):
    hp_artifact = standard_artifacts.HyperParameters()
    hp_artifact.uri = os.path.join(self._output_data_dir, 'hyperparameters/')

    # TODO(jyzhao): use real keras_tuner.HyperParameters instead of dict.
    hyperparameters = {}
    hyperparameters['first_dnn_layer_size'] = 100
    hyperparameters['num_dnn_layers'] = 4
    hyperparameters['dnn_decay_factor'] = 0.7
    io_utils.write_string_file(
        os.path.join(hp_artifact.uri, 'hyperparameters.txt'),
        json.dumps(hyperparameters))

    self._input_dict[standard_component_specs.HYPERPARAMETERS_KEY] = [
        hp_artifact
    ]

    self._exec_properties[
        standard_component_specs.MODULE_FILE_KEY] = self._module_file
    self._do(self._trainer_executor)
    self._verify_model_exports()
    self._verify_model_run_exports()

  def testMultipleArtifacts(self):
    self._input_dict[
        standard_component_specs.EXAMPLES_KEY] = self._multiple_artifacts
    self._exec_properties[
        standard_component_specs.MODULE_FILE_KEY] = self._module_file
    self._do(self._generic_trainer_executor)
    self._verify_model_exports()
    self._verify_model_run_exports()

  def testDoWithCustomSplits(self):
    # Update input dict.
    io_utils.copy_dir(
        os.path.join(self._source_data_dir,
                     'transform/transformed_examples/Split-train'),
        os.path.join(self._output_data_dir, 'data/Split-training'))
    io_utils.copy_dir(
        os.path.join(self._source_data_dir,
                     'transform/transformed_examples/Split-eval'),
        os.path.join(self._output_data_dir, 'data/Split-evaluating'))
    examples = standard_artifacts.Examples()
    examples.uri = os.path.join(self._output_data_dir, 'data')
    examples.split_names = artifact_utils.encode_split_names(
        ['training', 'evaluating'])
    self._input_dict[standard_component_specs.EXAMPLES_KEY] = [examples]

    # Update exec properties skeleton with custom splits.
    self._exec_properties[
        standard_component_specs.TRAIN_ARGS_KEY] = proto_utils.proto_to_json(
            trainer_pb2.TrainArgs(splits=['training'], num_steps=1000))
    self._exec_properties[
        standard_component_specs.EVAL_ARGS_KEY] = proto_utils.proto_to_json(
            trainer_pb2.EvalArgs(splits=['evaluating'], num_steps=500))

    self._exec_properties[
        standard_component_specs.MODULE_FILE_KEY] = self._module_file
    self._do(self._trainer_executor)
    self._verify_model_exports()
    self._verify_model_run_exports()


if __name__ == '__main__':
  tf.test.main()
