# Copyright 2018 Google LLC
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
import types
import tensorflow as tf
from tensorflow_metadata.proto.v0 import schema_pb2
from tfx.examples.chicago_taxi_pipeline import taxi_utils
from tfx.utils import io_utils


class TaxiUtilsTest(tf.test.TestCase):

  def test_utils(self):
    key = 'fare'
    xfm_key = taxi_utils._transformed_name(key)
    self.assertEqual(xfm_key, 'fare_xf')

  def test_trainer_fn(self):
    testdata_path = os.path.join(os.path.dirname(__file__), 'testdata')
    schema_file = os.path.join(testdata_path, 'schema.pbtxt')
    hparams = tf.contrib.training.HParams(
        train_files=os.path.join(testdata_path, 'transformed_examples/train/*'),
        transform_output=os.path.join(testdata_path, 'transform_output/'),
        output_dir=os.path.join(testdata_path, 'trainer_output/'),
        serving_model_dir=os.path.join(testdata_path, 'serving_model_dir/'),
        eval_files=os.path.join(testdata_path, 'transformed_examples/eval/*'),
        schema_file=schema_file,
        train_steps=10001,
        eval_steps=5000,
        verbosity='INFO',
        warm_start_from=os.path.join(testdata_path, '/serving_model_dir'))
    schema = io_utils.parse_pbtxt_file(schema_file, schema_pb2.Schema())
    training_spec = taxi_utils.trainer_fn(hparams, schema)

    self.assertIsInstance(training_spec['estimator'],
                          tf.estimator.DNNLinearCombinedClassifier)
    self.assertIsInstance(training_spec['train_spec'], tf.estimator.TrainSpec)
    self.assertIsInstance(training_spec['eval_spec'], tf.estimator.EvalSpec)
    self.assertIsInstance(training_spec['eval_input_receiver_fn'],
                          types.FunctionType)


if __name__ == '__main__':
  tf.test.main()
