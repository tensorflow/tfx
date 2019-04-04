# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for chicago_taxi."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from tensorflow.python.lib.io import file_io  # pylint: disable=g-direct-tensorflow-import
from tfx.examples.chicago_taxi import preprocess
from tfx.examples.chicago_taxi import process_tfma
from tfx.examples.chicago_taxi import tfdv_analyze_and_validate
from tfx.examples.chicago_taxi.trainer import task as trainer_task

_DATA_DIR_PATH = os.path.join(os.path.dirname(__file__), 'data')
_TRAINER_OUTPUT_DIR = 'train_output'
_TRAIN_DATA_FILE_PREFIX = 'transformed_train_data'
_EVAL_DATA_FILE_PREFIX = 'transformed_eval_data'


class TaxiTest(tf.test.TestCase):
  """Unit test for taxi util."""

  def setUp(self):
    self._working_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    file_io.recursive_create_dir(self._working_dir)

  def test_pipeline(self):
    # TODO(b/113256925): Split this test to test preprocess and training
    # separately. Possibly using tft_unit to test the result of transform_data.
    stats_file = os.path.join(self._working_dir, 'train_stats')
    tfdv_analyze_and_validate.compute_stats(
        input_handle=os.path.join(_DATA_DIR_PATH, 'train/data.csv'),
        stats_path=stats_file)

    schema_file = os.path.join(self._working_dir, 'schema.pbtxt')
    tfdv_analyze_and_validate.infer_schema(stats_file, schema_file)

    transform_output_path = os.path.join(self._working_dir, 'transform_output')

    preprocess.transform_data(
        os.path.join(_DATA_DIR_PATH, 'train/data.csv'),
        outfile_prefix=_TRAIN_DATA_FILE_PREFIX,
        working_dir=transform_output_path,
        schema_file=schema_file)
    preprocess.transform_data(
        input_handle=os.path.join(_DATA_DIR_PATH, 'eval/data.csv'),
        outfile_prefix=_EVAL_DATA_FILE_PREFIX,
        working_dir=transform_output_path,
        transform_dir=transform_output_path,
        schema_file=schema_file)

    hparams = tf.contrib.training.HParams(
        train_steps=100,
        eval_steps=50,
        job_dir=self._working_dir,
        output_dir=os.path.join(self._working_dir, _TRAINER_OUTPUT_DIR),
        tf_transform_dir=transform_output_path,
        train_files=os.path.join(transform_output_path,
                                 '{}-*'.format(_TRAIN_DATA_FILE_PREFIX)),
        eval_files=os.path.join(transform_output_path,
                                '{}-*'.format(_EVAL_DATA_FILE_PREFIX)),
        schema_file=schema_file)
    # TODO(b/113256925): Test the result of run_experiment.
    trainer_task.run_experiment(hparams)

    # Find where Trainer wrote the eval model
    eval_model_dir = os.path.join(self._working_dir, _TRAINER_OUTPUT_DIR,
                                  trainer_task.EVAL_MODEL_DIR)

    # Appends the directory name where the model was exported to (some number).
    eval_model_dir = os.path.join(eval_model_dir, os.listdir(eval_model_dir)[0])

    # The data under eval_model was produced by test_train.
    # TODO(b/113256925): Test the result of process_tfma.
    process_tfma.process_tfma(
        eval_result_dir=self._working_dir,
        input_csv=os.path.join(_DATA_DIR_PATH, 'eval/data.csv'),
        eval_model_dir=eval_model_dir,
        schema_file=schema_file)


if __name__ == '__main__':
  tf.test.main()
