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
"""Tests for tfx.components.tuner.executor."""

import copy
import json
import os
import unittest

from keras_tuner import HyperParameters
import tensorflow as tf
from tfx.components.testdata.module_file import tuner_module
from tfx.components.tuner import executor
from tfx.dsl.io import fileio
from tfx.proto import trainer_pb2
from tfx.proto import tuner_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs
from tfx.utils import io_utils
from tfx.utils import name_utils
from tfx.utils import proto_utils

from tensorflow.python.lib.io import file_io  # pylint: disable=g-direct-tensorflow-import


@unittest.skipIf(tf.__version__ < '2',
                 'This test uses testdata only compatible with TF 2.x')
class ExecutorTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._testdata_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')
    self._output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    self._context = executor.Executor.Context(
        tmp_dir=self._output_data_dir, unique_id='1')

    # Create input dict.
    e1 = standard_artifacts.Examples()
    e1.uri = os.path.join(self._testdata_dir, 'penguin', 'data')
    e1.split_names = artifact_utils.encode_split_names(['train', 'eval'])

    e2 = copy.deepcopy(e1)

    self._single_artifact = [e1]
    self._multiple_artifacts = [e1, e2]

    schema = standard_artifacts.Schema()
    schema.uri = os.path.join(self._testdata_dir, 'penguin', 'schema')

    base_model = standard_artifacts.Model()
    base_model.uri = os.path.join(self._testdata_dir, 'trainer/previous')

    self._input_dict = {
        standard_component_specs.EXAMPLES_KEY: self._single_artifact,
        standard_component_specs.SCHEMA_KEY: [schema],
        standard_component_specs.BASE_MODEL_KEY: [base_model]
    }

    # Create output dict.
    self._best_hparams = standard_artifacts.HyperParameters()
    self._best_hparams.uri = os.path.join(self._output_data_dir, 'best_hparams')
    self._tuner_results = standard_artifacts.TunerResults()
    self._tuner_results.uri = os.path.join(self._output_data_dir, 'results')

    self._output_dict = {
        standard_component_specs.BEST_HYPERPARAMETERS_KEY: [self._best_hparams],
        standard_component_specs.TUNER_RESULTS_KEY: [self._tuner_results]
    }

    # Create exec properties.
    self._exec_properties = {
        standard_component_specs.TRAIN_ARGS_KEY:
            proto_utils.proto_to_json(trainer_pb2.TrainArgs(num_steps=100)),
        standard_component_specs.EVAL_ARGS_KEY:
            proto_utils.proto_to_json(trainer_pb2.EvalArgs(num_steps=50)),
    }

  def _verify_output(self):
    """Verifies that best hparams and tuning results are saved."""
    best_hparams_path = os.path.join(self._best_hparams.uri,
                                     executor._DEFAULT_BEST_HP_FILE_NAME)
    self.assertTrue(fileio.exists(best_hparams_path))
    best_hparams_config = json.loads(
        file_io.read_file_to_string(best_hparams_path))
    best_hparams = HyperParameters.from_config(best_hparams_config)
    self.assertIn(best_hparams.get('learning_rate'), (1e-1, 1e-3))
    self.assertBetween(best_hparams.get('num_layers'), 1, 5)

    tuner_results_path = os.path.join(self._tuner_results.uri,
                                      executor._DEFAULT_TUNER_RESULTS_FILE_NAME)
    self.assertTrue(fileio.exists(tuner_results_path))
    tuner_results = json.loads(file_io.read_file_to_string(tuner_results_path))
    self.assertLen(tuner_results, 3)
    self.assertEqual({'trial_id', 'score', 'learning_rate', 'num_layers'},
                     tuner_results[0].keys())

  def testDoWithModuleFile(self):
    self._exec_properties[
        standard_component_specs.MODULE_FILE_KEY] = os.path.join(
            self._testdata_dir, 'module_file', 'tuner_module.py')

    tuner = executor.Executor(self._context)
    tuner.Do(
        input_dict=self._input_dict,
        output_dict=self._output_dict,
        exec_properties=self._exec_properties)

    self._verify_output()

  def testDoWithTunerFn(self):
    self._exec_properties[standard_component_specs.TUNER_FN_KEY] = (
        name_utils.get_full_name(tuner_module.tuner_fn))

    tuner = executor.Executor(self._context)
    tuner.Do(
        input_dict=self._input_dict,
        output_dict=self._output_dict,
        exec_properties=self._exec_properties)

    self._verify_output()

  def testTuneArgs(self):
    with self.assertRaises(ValueError):
      self._exec_properties[
          standard_component_specs.TUNE_ARGS_KEY] = proto_utils.proto_to_json(
              tuner_pb2.TuneArgs(num_parallel_trials=3))

      tuner = executor.Executor(self._context)
      tuner.Do(
          input_dict=self._input_dict,
          output_dict=self._output_dict,
          exec_properties=self._exec_properties)

  def testDoWithCustomSplits(self):
    # Update input dict.
    io_utils.copy_dir(
        os.path.join(self._testdata_dir, 'penguin/data/Split-train'),
        os.path.join(self._output_data_dir, 'data/Split-training'))
    io_utils.copy_dir(
        os.path.join(self._testdata_dir, 'penguin/data/Split-eval'),
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
        standard_component_specs.MODULE_FILE_KEY] = os.path.join(
            self._testdata_dir, 'module_file', 'tuner_module.py')

    tuner = executor.Executor(self._context)
    tuner.Do(
        input_dict=self._input_dict,
        output_dict=self._output_dict,
        exec_properties=self._exec_properties)

    self._verify_output()

  def testMultipleArtifacts(self):
    self._input_dict[
        standard_component_specs.EXAMPLES_KEY] = self._multiple_artifacts
    self._exec_properties[
        standard_component_specs.MODULE_FILE_KEY] = os.path.join(
            self._testdata_dir, 'module_file', 'tuner_module.py')

    tuner = executor.Executor(self._context)
    tuner.Do(
        input_dict=self._input_dict,
        output_dict=self._output_dict,
        exec_properties=self._exec_properties)

    self._verify_output()


if __name__ == '__main__':
  tf.test.main()
