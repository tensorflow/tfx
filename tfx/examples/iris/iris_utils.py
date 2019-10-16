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
"""Python source file include Iris pipeline functions and necesasry utils.

For a TFX pipeline to successfully run, a preprocessing_fn and a
_build_estimator function needs to be provided.  This file contains both.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import absl
import tensorflow as tf
import tensorflow_model_analysis as tfma
from tensorflow_transform.tf_metadata import schema_utils

_FEATURE_KEYS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
_LABEL_KEY = 'variety'


# Tf.Transform considers these features as "raw"
def _get_raw_feature_spec(schema):
  return schema_utils.schema_as_feature_spec(schema).feature_spec


def _gzip_reader_fn(filenames):
  """Small utility returning a record reader that can read gzip'ed files."""
  return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def _example_serving_receiver_fn(schema):
  """Build the serving in inputs.

  Args:
    schema: the schema of the input data.

  Returns:
    Tensorflow graph which parses examples, applying tf-transform to them.
  """
  raw_feature_spec = _get_raw_feature_spec(schema)
  raw_feature_spec.pop(_LABEL_KEY)

  raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
      raw_feature_spec, default_batch_size=None)
  serving_input_receiver = raw_input_fn()

  return tf.estimator.export.ServingInputReceiver(
      serving_input_receiver.features, serving_input_receiver.receiver_tensors)


def _eval_input_receiver_fn(schema):
  """Build everything needed for the tf-model-analysis to run the model.

  Args:
    schema: the schema of the input data.

  Returns:
    EvalInputReceiver function, which contains:
      - Tensorflow graph which parses raw untransformed features, applies the
        tf-transform preprocessing operators.
      - Set of raw, untransformed features.
      - Label against which predictions will be compared.
  """
  # Notice that the inputs are raw features, not transformed features here.
  raw_feature_spec = _get_raw_feature_spec(schema)

  serialized_tf_example = tf.compat.v1.placeholder(
      dtype=tf.string, shape=[None], name='input_example_tensor')

  # Add a parse_example operator to the tensorflow graph, which will parse
  # raw, untransformed, tf examples.
  features = tf.io.parse_example(
      serialized=serialized_tf_example, features=raw_feature_spec)

  # The key name MUST be 'examples'.
  receiver_tensors = {'examples': serialized_tf_example}

  return tfma.export.EvalInputReceiver(
      features=features,
      receiver_tensors=receiver_tensors,
      labels=features.pop(_LABEL_KEY))


def _input_fn(filenames, schema, batch_size=200):
  """Generates features and labels for training or evaluation.

  Args:
    filenames: [str] list of CSV files to read data from.
    schema: Schema of the input data.
    batch_size: int First dimension size of the Tensors returned by input_fn

  Returns:
    A (features, indices) tuple where features is a dictionary of
      Tensors, and indices is a single Tensor of label indices.
  """

  feature_spec = _get_raw_feature_spec(schema)

  dataset = tf.data.experimental.make_batched_features_dataset(
      filenames, batch_size, feature_spec, reader=_gzip_reader_fn)

  features = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()

  # We pop the label because we do not want to use it as a feature while we're
  # training.
  return features, features.pop(_LABEL_KEY)


def _keras_model_builder():
  """Creates a DNN Keras model  for classifying iris data.

  Returns:
    A keras Model.
  """

  l = tf.keras.layers
  opt = tf.keras.optimizers
  inputs = [l.Input(shape=(1,), name=f) for f in _FEATURE_KEYS]
  input_layer = l.concatenate(inputs)
  d1 = l.Dense(8, activation='relu')(input_layer)
  output = l.Dense(3, activation='softmax')(d1)
  model = tf.keras.Model(inputs=inputs, outputs=output)
  model.compile(
      loss='sparse_categorical_crossentropy',
      optimizer=opt.Adam(lr=0.001),
      metrics=['accuracy'])
  absl.logging.info(model.summary())
  return model


# TFX will call this function
def trainer_fn(hparams, schema):
  """Build the estimator using the high level API.

  Args:
    hparams: Holds hyperparameters used to train the model as name/value pairs.
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
      hparams.train_files,
      schema,
      batch_size=train_batch_size)

  eval_input_fn = lambda: _input_fn(  # pylint: disable=g-long-lambda
      hparams.eval_files,
      schema,
      batch_size=eval_batch_size)

  train_spec = tf.estimator.TrainSpec(  # pylint: disable=g-long-lambda
      train_input_fn,
      max_steps=hparams.train_steps)

  serving_receiver_fn = lambda: _example_serving_receiver_fn(schema)  # pylint: disable=g-long-lambda

  exporter = tf.estimator.FinalExporter('iris', serving_receiver_fn)
  eval_spec = tf.estimator.EvalSpec(
      eval_input_fn,
      steps=hparams.eval_steps,
      exporters=[exporter],
      name='iris-eval')

  run_config = tf.estimator.RunConfig(
      save_checkpoints_steps=999, keep_checkpoint_max=1)

  run_config = run_config.replace(model_dir=hparams.serving_model_dir)

  estimator = tf.keras.estimator.model_to_estimator(
      keras_model=_keras_model_builder(), config=run_config)

  # Create an input receiver for TFMA processing
  receiver_fn = lambda: _eval_input_receiver_fn(schema)  # pylint: disable=g-long-lambda

  return {
      'estimator': estimator,
      'train_spec': train_spec,
      'eval_spec': eval_spec,
      'eval_input_receiver_fn': receiver_fn
  }
