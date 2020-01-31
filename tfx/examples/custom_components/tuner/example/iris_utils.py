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
"""Python source file include Iris pipeline functions and necesasry utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Text

import absl
import kerastuner
import tensorflow as tf
from tensorflow import keras
import tensorflow_model_analysis as tfma
from tensorflow_transform.tf_metadata import schema_utils

from tensorflow_metadata.proto.v0 import schema_pb2
from tfx.examples.custom_components.tuner.tuner_component import component

_FEATURE_KEYS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
_LABEL_KEY = 'variety'


# Tf.Transform considers these features as "raw"
def _get_raw_feature_spec(schema):
  return schema_utils.schema_as_feature_spec(schema).feature_spec


def _gzip_reader_fn(filenames):
  """Small utility returning a record reader that can read gzip'ed files."""
  return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def _serving_input_receiver_fn(schema):
  """Build the serving inputs.

  Args:
    schema: the schema of the input data.

  Returns:
    serving_input_receiver_fn for serving this model, since no transformation is
    required in this case it does not include a tf-transform graph.
  """
  raw_feature_spec = _get_raw_feature_spec(schema)
  raw_feature_spec.pop(_LABEL_KEY)

  raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
      raw_feature_spec, default_batch_size=None)
  return raw_input_fn()


def _eval_input_receiver_fn(schema):
  """Build the evalution inputs for the tf-model-analysis to run the model.

  Args:
    schema: the schema of the input data.

  Returns:
    EvalInputReceiver function, which contains:
      - Features (dict of Tensors) to be passed to the model.
      - Raw features as serialized tf.Examples.
      - Labels
  """
  # Notice that the inputs are raw features, not transformed features here.
  raw_feature_spec = _get_raw_feature_spec(schema)

  raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
      raw_feature_spec, default_batch_size=None)
  serving_input_receiver = raw_input_fn()

  labels = serving_input_receiver.features.pop(_LABEL_KEY)
  return tfma.export.EvalInputReceiver(
      features=serving_input_receiver.features,
      labels=labels,
      receiver_tensors=serving_input_receiver.receiver_tensors)


def _input_fn(file_pattern: Text,
              schema: schema_pb2.Schema,
              batch_size: int = 200) -> tf.data.Dataset:
  """Generates features and label for tuning/training.

  Args:
    file_pattern: input tfrecord file pattern.
    schema: Schema of the input data.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
  feature_spec = _get_raw_feature_spec(schema)

  dataset = tf.data.experimental.make_batched_features_dataset(
      file_pattern=file_pattern,
      batch_size=batch_size,
      features=feature_spec,
      reader=_gzip_reader_fn,
      label_key=_LABEL_KEY)

  return dataset


def _build_keras_model(hparams: kerastuner.HyperParameters) -> tf.keras.Model:
  """Creates a DNN Keras model for classifying iris data.

  Args:
    hparams: Holds HyperParameters for tuning.

  Returns:
    A Keras Model.
  """
  absl.logging.info('HyperParameters config: %s' % hparams.get_config())
  inputs = [keras.layers.Input(shape=(1,), name=f) for f in _FEATURE_KEYS]
  d = keras.layers.concatenate(inputs)
  for _ in range(hparams.get('num_layers')):  # pytype: disable=wrong-arg-types
    d = keras.layers.Dense(8, activation='relu')(d)
  output = keras.layers.Dense(3, activation='softmax')(d)
  model = keras.Model(inputs=inputs, outputs=output)
  model.compile(
      optimizer=keras.optimizers.Adam(hparams.get('learning_rate')),
      loss='sparse_categorical_crossentropy',
      metrics=[keras.metrics.SparseCategoricalAccuracy(name='accuracy')])
  absl.logging.info(model.summary())
  return model


# TFX will call this function
def trainer_fn(trainer_fn_args, schema):
  """Build the estimator using the high level API.

  Args:
    trainer_fn_args: Holds args used to train the model as name/value pairs.
    schema: Holds the schema of the training examples.

  Returns:
    A dict of the following:
      - estimator: The estimator that will be used for training and eval.
      - train_spec: Spec for training.
      - eval_spec: Spec for eval.
      - eval_input_receiver_fn: Input function for eval.
  """
  train_batch_size = 40
  eval_batch_size = 40

  train_input_fn = lambda: _input_fn(  # pylint: disable=g-long-lambda
      trainer_fn_args.train_files,
      schema,
      batch_size=train_batch_size)

  eval_input_fn = lambda: _input_fn(  # pylint: disable=g-long-lambda
      trainer_fn_args.eval_files,
      schema,
      batch_size=eval_batch_size)

  train_spec = tf.estimator.TrainSpec(
      train_input_fn, max_steps=trainer_fn_args.train_steps)

  serving_receiver_fn = lambda: _serving_input_receiver_fn(schema)

  exporter = tf.estimator.FinalExporter('iris', serving_receiver_fn)
  eval_spec = tf.estimator.EvalSpec(
      eval_input_fn,
      steps=trainer_fn_args.eval_steps,
      exporters=[exporter],
      name='iris-eval')

  run_config = tf.estimator.RunConfig(
      save_checkpoints_steps=999, keep_checkpoint_max=1)

  run_config = run_config.replace(model_dir=trainer_fn_args.serving_model_dir)

  # TODO(jyzhao): change to native keras when supported.
  estimator = tf.keras.estimator.model_to_estimator(
      keras_model=_build_keras_model(
          kerastuner.HyperParameters.from_config(
              trainer_fn_args.hyperparameters)),
      config=run_config)

  # Create an input receiver for TFMA processing
  eval_receiver_fn = lambda: _eval_input_receiver_fn(schema)

  return {
      'estimator': estimator,
      'train_spec': train_spec,
      'eval_spec': eval_spec,
      'eval_input_receiver_fn': eval_receiver_fn
  }


# TFX will call this function
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
  hparams.Int('num_layers', 1, 5)

  # TODO(jyzhao): support params, e.g., max_trials in user input config.
  tuner = kerastuner.RandomSearch(
      _build_keras_model,
      max_trials=5,
      hyperparameters=hparams,
      allow_new_entries=False,
      objective='val_accuracy',
      directory=working_dir,
      project_name='iris')

  return component.TunerFnResult(
      tuner=tuner,
      train_dataset=_input_fn(train_data_pattern, schema, 10),
      eval_dataset=_input_fn(eval_data_pattern, schema, 10))
