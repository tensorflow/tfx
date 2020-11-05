# Lint as: python2, python3
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
"""Python source file include Iris pipeline functions and necessary utils.

The utilities in this file are used to build a model with native Keras.
This module file will be used in Transform and generic Trainer.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Text
import absl
import kerastuner
import tensorflow as tf
from tensorflow import keras
from tensorflow_transform.tf_metadata import schema_utils

from tfx.components.trainer.executor import TrainerFnArgs
from tfx.components.trainer.fn_args_utils import DataAccessor
from tfx_bsl.tfxio import dataset_options
from tensorflow_metadata.proto.v0 import schema_pb2

_FEATURE_KEYS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
_LABEL_KEY = 'variety'

# Iris dataset has 150 records, and is divided to train and eval splits in 2:1
# ratio.
_TRAIN_DATA_SIZE = 100
_EVAL_DATA_SIZE = 50
_TRAIN_BATCH_SIZE = 20
_EVAL_BATCH_SIZE = 10

_FEATURE_SPEC = {
    'petal_length': tf.io.FixedLenFeature(shape=[1], dtype=tf.float32),
    'petal_width': tf.io.FixedLenFeature(shape=[1], dtype=tf.float32),
    'sepal_length': tf.io.FixedLenFeature(shape=[1], dtype=tf.float32),
    'sepal_width': tf.io.FixedLenFeature(shape=[1], dtype=tf.float32),
    'variety': tf.io.FixedLenFeature(shape=[1], dtype=tf.int64)
}


def _get_serve_tf_examples_fn(model, feature_spec):
  """Returns a function that parses a serialized tf.Example."""

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    """Returns the output to be used in the serving signature."""
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

    return model(parsed_features)

  return serve_tf_examples_fn


def _input_fn(file_pattern: List[Text],
              data_accessor: DataAccessor,
              schema: schema_pb2.Schema,
              batch_size: int = 200) -> tf.data.Dataset:
  """Generates features and label for tuning/training.

  Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    data_accessor: DataAccessor for converting input to RecordBatch.
    schema: schema of the input data.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
  return data_accessor.tf_dataset_factory(
      file_pattern,
      dataset_options.TensorFlowDatasetOptions(
          batch_size=batch_size, label_key=_LABEL_KEY),
      schema=schema)


def _get_hyperparameters() -> kerastuner.HyperParameters:
  """Returns hyperparameters for building Keras model."""
  hp = kerastuner.HyperParameters()
  # Defines search space.
  hp.Choice('learning_rate', [1e-2, 1e-3], default=1e-2)
  hp.Int('num_layers', 1, 3, default=2)
  return hp


def _build_keras_model(hparams: kerastuner.HyperParameters) -> tf.keras.Model:
  """Creates a DNN Keras model for classifying iris data.

  Args:
    hparams: Holds HyperParameters for tuning.

  Returns:
    A Keras Model.
  """
  # The model below is built with Functional API, please refer to
  # https://www.tensorflow.org/guide/keras/overview for all API options.
  inputs = [keras.layers.Input(shape=(1,), name=f) for f in _FEATURE_KEYS]
  d = keras.layers.concatenate(inputs)
  for _ in range(int(hparams.get('num_layers'))):
    d = keras.layers.Dense(8, activation='relu')(d)
  outputs = keras.layers.Dense(3, activation='softmax')(d)

  model = keras.Model(inputs=inputs, outputs=outputs)
  model.compile(
      optimizer=keras.optimizers.Adam(hparams.get('learning_rate')),
      loss='sparse_categorical_crossentropy',
      metrics=[keras.metrics.SparseCategoricalAccuracy()])

  model.summary(print_fn=absl.logging.info)
  return model


# TFX Trainer will call this function.
def run_fn(fn_args: TrainerFnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """
  schema = schema_utils.schema_from_feature_spec(_FEATURE_SPEC)

  train_dataset = _input_fn(
      fn_args.train_files,
      fn_args.data_accessor,
      schema,
      batch_size=_TRAIN_BATCH_SIZE)
  eval_dataset = _input_fn(
      fn_args.eval_files,
      fn_args.data_accessor,
      schema,
      batch_size=_EVAL_BATCH_SIZE)

  if fn_args.hyperparameters:
    hparams = kerastuner.HyperParameters.from_config(fn_args.hyperparameters)
  else:
    # This is a shown case when hyperparameters is decided and Tuner is removed
    # from the pipeline. User can also inline the hyperparameters directly in
    # _build_keras_model.
    hparams = _get_hyperparameters()
  absl.logging.info('HyperParameters for training: %s' % hparams.get_config())

  mirrored_strategy = tf.distribute.MirroredStrategy()
  with mirrored_strategy.scope():
    model = _build_keras_model(hparams)

  steps_per_epoch = _TRAIN_DATA_SIZE / _TRAIN_BATCH_SIZE

  # Write logs to path
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=fn_args.model_run_dir, update_freq='batch')

  model.fit(
      train_dataset,
      epochs=int(fn_args.train_steps / steps_per_epoch),
      steps_per_epoch=steps_per_epoch,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps,
      callbacks=[tensorboard_callback])

  signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(model, _FEATURE_SPEC).get_concrete_function(
              tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')),
  }
  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
