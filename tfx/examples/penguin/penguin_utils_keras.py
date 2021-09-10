# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Supplement for penguin_utils_base.py with specifics for Keras models.

This module file will be used in the Transform, Tuner and generic Trainer
components.
"""

import absl
import keras_tuner
import tensorflow as tf
from tensorflow import keras
import tensorflow_transform as tft

from tfx import v1 as tfx
from tfx.examples.penguin import penguin_utils_base as base


# TFX Transform will call this function.
preprocessing_fn = base.preprocessing_fn


def _get_hyperparameters() -> keras_tuner.HyperParameters:
  """Returns hyperparameters for building Keras model."""
  hp = keras_tuner.HyperParameters()
  # Defines search space.
  hp.Choice('learning_rate', [1e-2, 1e-3], default=1e-2)
  hp.Int('num_layers', 1, 3, default=2)
  return hp


def _make_keras_model(hparams: keras_tuner.HyperParameters) -> tf.keras.Model:
  """Creates a DNN Keras model for classifying penguin data.

  Args:
    hparams: Holds HyperParameters for tuning.

  Returns:
    A Keras Model.
  """
  # The model below is built with Functional API, please refer to
  # https://www.tensorflow.org/guide/keras/overview for all API options.
  inputs = [
      keras.layers.Input(shape=(1,), name=base.transformed_name(f))
      for f in base.FEATURE_KEYS
  ]
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


# TFX Tuner will call this function.
def tuner_fn(fn_args: tfx.components.FnArgs) -> tfx.components.TunerFnResult:
  """Build the tuner using the KerasTuner API.

  Args:
    fn_args: Holds args as name/value pairs.
      - working_dir: working dir for tuning.
      - train_files: List of file paths containing training tf.Example data.
      - eval_files: List of file paths containing eval tf.Example data.
      - train_steps: number of train steps.
      - eval_steps: number of eval steps.
      - schema_path: optional schema of the input data.
      - transform_graph_path: optional transform graph produced by TFT.

  Returns:
    A namedtuple contains the following:
      - tuner: A BaseTuner that will be used for tuning.
      - fit_kwargs: Args to pass to tuner's run_trial function for fitting the
                    model , e.g., the training and validation dataset. Required
                    args depend on the above tuner's implementation.
  """
  # RandomSearch is a subclass of keras_tuner.Tuner which inherits from
  # BaseTuner.
  tuner = keras_tuner.RandomSearch(
      _make_keras_model,
      max_trials=6,
      hyperparameters=_get_hyperparameters(),
      allow_new_entries=False,
      objective=keras_tuner.Objective('val_sparse_categorical_accuracy', 'max'),
      directory=fn_args.working_dir,
      project_name='penguin_tuning')

  transform_graph = tft.TFTransformOutput(fn_args.transform_graph_path)

  train_dataset = base.input_fn(
      fn_args.train_files,
      fn_args.data_accessor,
      transform_graph,
      base.TRAIN_BATCH_SIZE)

  eval_dataset = base.input_fn(
      fn_args.eval_files,
      fn_args.data_accessor,
      transform_graph,
      base.EVAL_BATCH_SIZE)

  return tfx.components.TunerFnResult(
      tuner=tuner,
      fit_kwargs={
          'x': train_dataset,
          'validation_data': eval_dataset,
          'steps_per_epoch': fn_args.train_steps,
          'validation_steps': fn_args.eval_steps
      })


# TFX Trainer will call this function.
def run_fn(fn_args: tfx.components.FnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  train_dataset = base.input_fn(
      fn_args.train_files,
      fn_args.data_accessor,
      tf_transform_output,
      base.TRAIN_BATCH_SIZE)

  eval_dataset = base.input_fn(
      fn_args.eval_files,
      fn_args.data_accessor,
      tf_transform_output,
      base.EVAL_BATCH_SIZE)

  if fn_args.hyperparameters:
    hparams = keras_tuner.HyperParameters.from_config(fn_args.hyperparameters)
  else:
    # This is a shown case when hyperparameters is decided and Tuner is removed
    # from the pipeline. User can also inline the hyperparameters directly in
    # _build_keras_model.
    hparams = _get_hyperparameters()
  absl.logging.info('HyperParameters for training: %s' % hparams.get_config())

  mirrored_strategy = tf.distribute.MirroredStrategy()
  with mirrored_strategy.scope():
    model = _make_keras_model(hparams)

  # Write logs to path
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=fn_args.model_run_dir, update_freq='batch')

  model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps,
      callbacks=[tensorboard_callback])

  signatures = base.make_serving_signatures(model, tf_transform_output)
  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
