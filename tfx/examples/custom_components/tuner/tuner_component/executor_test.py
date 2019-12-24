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
from kerastuner import HyperParameters
import tensorflow as tf

from tensorflow.python.lib.io import file_io  # pylint: disable=g-direct-tensorflow-import
from tensorflow_metadata.proto.v0 import schema_pb2
from tfx.examples.custom_components.tuner.testdata import module
from tfx.examples.custom_components.tuner.tuner_component import executor
from tfx.types import standard_artifacts
from tfx.utils import io_utils

# TODO(jyzhao): v1 doesn't work for dataset and tuner.
tf.enable_v2_behavior()


class ExecutorTest(tf.test.TestCase):

  def setUp(self):
    super(ExecutorTest, self).setUp()
    self._source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')
    self._output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    io_utils.write_pbtxt_file(
        os.path.join(self._output_data_dir, 'schema', 'schema.pbtxt'),
        schema_pb2.Schema())

    self._context = executor.Executor.Context(
        tmp_dir=self._output_data_dir, unique_id='1')

    # Create input dict.
    train_examples = standard_artifacts.Examples(split='train')
    train_examples.uri = 'path/train/'
    eval_examples = standard_artifacts.Examples(split='eval')
    eval_examples.uri = 'path/eval/'
    schema = standard_artifacts.Schema()
    schema.uri = os.path.join(self._output_data_dir, 'schema')

    self._input_dict = {
        'examples': [train_examples, eval_examples],
        'schema': [schema],
    }

    # Create output dict.
    model = standard_artifacts.Model()
    model.uri = os.path.join(self._output_data_dir, 'model')
    self._best_hparams = standard_artifacts.Model()
    self._best_hparams.uri = os.path.join(self._output_data_dir, 'best_hparams')

    self._output_dict = {
        'model': [model],
        'study_best_hparams_path': [self._best_hparams],
    }

  def _verify_output(self):
    # Test best hparams.
    best_hparams_path = os.path.join(self._best_hparams.uri, 'best_hparams.txt')
    self.assertTrue(tf.gfile.Exists(best_hparams_path))
    best_hparams_config = json.loads(
        file_io.read_file_to_string(best_hparams_path))
    best_hparams = HyperParameters.from_config(best_hparams_config)
    self.assertIn(best_hparams.get('learning_rate'), (1e-1, 1e-3))
    self.assertBetween(best_hparams.get('num_layers'), 2, 10)

  def testDoWithModuleFile(self):
    # Create exec properties.
    exec_properties = {
        'module_file': os.path.join(self._source_data_dir, 'module.py')
    }

    # Run tuner.
    tuner = executor.Executor(self._context)
    tuner.Do(
        input_dict=self._input_dict,
        output_dict=self._output_dict,
        exec_properties=exec_properties)

    self._verify_output()

  def testDoWithTunerFn(self):
    # Create exec properties.
    exec_properties = {
        'tuner_fn':
            '%s.%s' % (module.tuner_fn.__module__, module.tuner_fn.__name__)
    }

    # Run tuner.
    tuner = executor.Executor(self._context)
    tuner.Do(
        input_dict=self._input_dict,
        output_dict=self._output_dict,
        exec_properties=exec_properties)

    self._verify_output()


if __name__ == '__main__':
  tf.test.main()
