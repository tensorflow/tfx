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
"""Test module file for tuner's executor_test.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Text
import kerastuner
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import keras
from tensorflow_metadata.proto.v0 import schema_pb2
from tfx.examples.custom_components.tuner.tuner_component import component

# TODO(b/142654751): unify with trainer's module file.


def _make_input_dataset(
    file_pattern: Text,
    schema: schema_pb2.Schema,  # pylint: disable=unused-argument
    batch_size: int = 200) -> tf.data.Dataset:
  """Generates features and labels for tuning/training.

  Args:
    file_pattern: input tfrecord file pattern.
    schema: Schema of the input data.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

  Returns:
    A tf.data.Dataset that contains features-labels tuples.
  """
  # TODO(b/142654751): make it real, converted from tfrecord.
  if 'train' in file_pattern:
    data = np.random.random((1000, 32))
    labels = np.random.random((1000, 10))
  else:
    data = np.random.random((100, 32))
    labels = np.random.random((100, 10))

  dataset = tf.data.Dataset.from_tensor_slices((data, labels))
  dataset = dataset.batch(batch_size)
  return dataset


def _build_keras_model(hparams: kerastuner.HyperParameters) -> tf.keras.Model:
  """Creates Keras model for testing.

  Args:
    hparams: Holds HyperParameters for tuning.

  Returns:
    A Keras Model.
  """
  model = keras.Sequential()
  model.add(keras.layers.Dense(64, activation='relu', input_shape=(32,)))
  for _ in range(hparams.get('num_layers')):  # pytype: disable=wrong-arg-types
    model.add(keras.layers.Dense(64, activation='relu'))
  model.add(keras.layers.Dense(10, activation='softmax'))
  model.compile(
      optimizer=keras.optimizers.Adam(hparams.get('learning_rate')),
      loss='categorical_crossentropy',
      metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')])
  return model


def tuner_fn(working_dir: Text, train_data_pattern: Text,
             eval_data_pattern: Text,
             schema: schema_pb2.Schema) -> component.TunerFnResult:
  """Build the tuner using the Keras Tuner API.

  Args:
    working_dir: working dir for KerasTuner.
    train_data_pattern: file pattern of training tfrecord data.
    eval_data_pattern: file pattern of eval tfrecord data.
    schema: Schema of the input data.

  Returns:
    A namedtuple contains the following:
      - tuner: A KerasTuner that will be used for tuning.
      - train_dataset: A tf.data.Dataset of training data.
      - eval_dataset: A tf.data.Dataset of eval data.
  """
  hparams = kerastuner.HyperParameters()
  hparams.Choice('learning_rate', [1e-1, 1e-3])
  hparams.Int('num_layers', 2, 10)

  tuner = kerastuner.RandomSearch(
      _build_keras_model,
      max_trials=3,
      hyperparameters=hparams,
      allow_new_entries=False,
      objective='val_accuracy',
      directory=working_dir,
      project_name='test_project')

  return component.TunerFnResult(
      tuner=tuner,
      train_dataset=_make_input_dataset(train_data_pattern, schema, 32),
      eval_dataset=_make_input_dataset(eval_data_pattern, schema, 32))
