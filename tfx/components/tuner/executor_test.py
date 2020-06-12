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

import json
import os
from kerastuner import HyperParameters
import tensorflow as tf

from google.protobuf import json_format
from tensorflow.python.lib.io import file_io  # pylint: disable=g-direct-tensorflow-import
from tfx.components.testdata.module_file import tuner_module
from tfx.components.tuner import executor
from tfx.proto import trainer_pb2
from tfx.proto import tuner_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts


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
    examples = standard_artifacts.Examples()
    examples.uri = os.path.join(self._testdata_dir, 'iris', 'data')
    examples.split_names = artifact_utils.encode_split_names(['train', 'eval'])
    schema = standard_artifacts.Schema()
    schema.uri = os.path.join(self._testdata_dir, 'iris', 'schema')

    self._input_dict = {
        'examples': [examples],
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
      self._exec_properties['tune_args'] = tuner_pb2.TuneArgs(
          num_parallel_trials=3)

      tuner = executor.Executor(self._context)
      tuner.Do(
          input_dict=self._input_dict,
          output_dict=self._output_dict,
          exec_properties=self._exec_properties)


if __name__ == '__main__':
  tf.test.main()
