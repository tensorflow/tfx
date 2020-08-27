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
"""Tests for tfx.components.tuner.executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import os
from kerastuner import HyperParameters
import tensorflow as tf

from tfx.components.testdata.module_file import tuner_module
from tfx.components.tuner import executor
from tfx.proto import trainer_pb2
from tfx.proto import tuner_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.utils import io_utils
from google.protobuf import json_format
from tensorflow.python.lib.io import file_io  # pylint: disable=g-direct-tensorflow-import


class ExecutorTest(tf.test.TestCase):

  def setUp(self):
    super(ExecutorTest, self).setUp()
    self._testdata_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')
    self._output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    self._context = executor.Executor.Context(
        tmp_dir=self._output_data_dir, unique_id='1')

    # Create input dict.
    e1 = standard_artifacts.Examples()
    e1.uri = os.path.join(self._testdata_dir, 'iris', 'data')
    e1.split_names = artifact_utils.encode_split_names(['train', 'eval'])

    e2 = copy.deepcopy(e1)

    self._single_artifact = [e1]
    self._multiple_artifacts = [e1, e2]

    schema = standard_artifacts.Schema()
    schema.uri = os.path.join(self._testdata_dir, 'iris', 'schema')

    self._input_dict = {
        'examples': self._single_artifact,
        'schema': [schema],
    }

    # Create output dict.
    self._best_hparams = standard_artifacts.Model()
    self._best_hparams.uri = os.path.join(self._output_data_dir, 'best_hparams')

    self._output_dict = {
        'best_hyperparameters': [self._best_hparams],
    }

    # Create exec properties.
    self._exec_properties = {
        'train_args':
            json_format.MessageToJson(
                trainer_pb2.TrainArgs(num_steps=100),
                preserving_proto_field_name=True),
        'eval_args':
            json_format.MessageToJson(
                trainer_pb2.EvalArgs(num_steps=50),
                preserving_proto_field_name=True),
    }

  def _verify_output(self):
    # Test best hparams.
    best_hparams_path = os.path.join(self._best_hparams.uri,
                                     'best_hyperparameters.txt')
    self.assertTrue(tf.io.gfile.exists(best_hparams_path))
    best_hparams_config = json.loads(
        file_io.read_file_to_string(best_hparams_path))
    best_hparams = HyperParameters.from_config(best_hparams_config)
    self.assertIn(best_hparams.get('learning_rate'), (1e-1, 1e-3))
    self.assertBetween(best_hparams.get('num_layers'), 1, 5)

  def testDoWithModuleFile(self):
    self._exec_properties['module_file'] = os.path.join(self._testdata_dir,
                                                        'module_file',
                                                        'tuner_module.py')

    tuner = executor.Executor(self._context)
    tuner.Do(
        input_dict=self._input_dict,
        output_dict=self._output_dict,
        exec_properties=self._exec_properties)

    self._verify_output()

  def testDoWithTunerFn(self):
    self._exec_properties['tuner_fn'] = '%s.%s' % (
        tuner_module.tuner_fn.__module__, tuner_module.tuner_fn.__name__)

    tuner = executor.Executor(self._context)
    tuner.Do(
        input_dict=self._input_dict,
        output_dict=self._output_dict,
        exec_properties=self._exec_properties)

    self._verify_output()

  def testTuneArgs(self):
    with self.assertRaises(ValueError):
      self._exec_properties['tune_args'] = json_format.MessageToJson(
          tuner_pb2.TuneArgs(num_parallel_trials=3),
          preserving_proto_field_name=True)

      tuner = executor.Executor(self._context)
      tuner.Do(
          input_dict=self._input_dict,
          output_dict=self._output_dict,
          exec_properties=self._exec_properties)

  def testDoWithCustomSplits(self):
    # Update input dict.
    io_utils.copy_dir(
        os.path.join(self._testdata_dir, 'iris/data/train'),
        os.path.join(self._output_data_dir, 'data/training'))
    io_utils.copy_dir(
        os.path.join(self._testdata_dir, 'iris/data/eval'),
        os.path.join(self._output_data_dir, 'data/evaluating'))
    examples = standard_artifacts.Examples()
    examples.uri = os.path.join(self._output_data_dir, 'data')
    examples.split_names = artifact_utils.encode_split_names(
        ['training', 'evaluating'])
    self._input_dict['examples'] = [examples]

    # Update exec properties skeleton with custom splits.
    self._exec_properties['train_args'] = json_format.MessageToJson(
        trainer_pb2.TrainArgs(splits=['training'], num_steps=1000),
        preserving_proto_field_name=True)
    self._exec_properties['eval_args'] = json_format.MessageToJson(
        trainer_pb2.EvalArgs(splits=['evaluating'], num_steps=500),
        preserving_proto_field_name=True)
    self._exec_properties['module_file'] = os.path.join(self._testdata_dir,
                                                        'module_file',
                                                        'tuner_module.py')

    tuner = executor.Executor(self._context)
    tuner.Do(
        input_dict=self._input_dict,
        output_dict=self._output_dict,
        exec_properties=self._exec_properties)

    self._verify_output()

  def testMultipleArtifacts(self):
    self._input_dict['examples'] = self._multiple_artifacts
    self._exec_properties['module_file'] = os.path.join(self._testdata_dir,
                                                        'module_file',
                                                        'tuner_module.py')

    tuner = executor.Executor(self._context)
    tuner.Do(
        input_dict=self._input_dict,
        output_dict=self._output_dict,
        exec_properties=self._exec_properties)

    self._verify_output()


if __name__ == '__main__':
  tf.test.main()
