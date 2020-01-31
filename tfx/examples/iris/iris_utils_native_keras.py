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
This module file will be used in generic Trainer.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Text

import absl
import tensorflow as tf
from tensorflow import keras
from tensorflow_transform.tf_metadata import schema_utils

from tensorflow_metadata.proto.v0 import schema_pb2
from tfx.components.trainer.executor import TrainerFnArgs
from tfx.utils import io_utils

_FEATURE_KEYS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
_LABEL_KEY = 'variety'


# Tf.Transform considers these features as "raw"
def _get_raw_feature_spec(schema):
  return schema_utils.schema_as_feature_spec(schema).feature_spec


def _gzip_reader_fn(filenames):
  """Small utility returning a record reader that can read gzip'ed files."""
  return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def _get_serve_tf_examples_fn(model, schema):
  """Returns a function that parses a serialized tf.Example."""

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    """Parses features an invoke model with inputs."""
    feature_spec = _get_raw_feature_spec(schema)
    feature_spec.pop(_LABEL_KEY)

    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
    inputs = {f: parsed_features[f] for f in _FEATURE_KEYS}
    outputs = model(inputs)
    return {'outputs': outputs}

  return serve_tf_examples_fn


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


def _build_keras_model() -> tf.keras.Model:
  """Creates a DNN Keras model for classifying iris data.

  Returns:
    A Keras Model.
  """
  # The model below is built with Functional API, please refer to
  # https://www.tensorflow.org/guide/keras/overview for all API options.
  inputs = [keras.layers.Input(shape=(1,), name=f) for f in _FEATURE_KEYS]
  d = keras.layers.concatenate(inputs)
  for _ in range(3):
    d = keras.layers.Dense(8, activation='relu')(d)
  output = keras.layers.Dense(3, activation='softmax')(d)
  model = keras.Model(inputs=inputs, outputs=output)
  model.compile(
      optimizer=keras.optimizers.Adam(lr=0.001),
      loss='sparse_categorical_crossentropy',
      metrics=[keras.metrics.SparseCategoricalAccuracy(name='accuracy')])
  absl.logging.info(model.summary())
  return model


# TFX will call this function.
def run_fn(fn_args: TrainerFnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """
  schema = io_utils.parse_pbtxt_file(fn_args.schema_file, schema_pb2.Schema())

  train_dataset = _input_fn(fn_args.train_files, schema, 40)
  eval_dataset = _input_fn(fn_args.eval_files, schema, 40)

  model = _build_keras_model()

  model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps)

  signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(model, schema).get_concrete_function(
              tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')),
  }
  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
