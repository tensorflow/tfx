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

from typing import Dict, Text, Tuple
import absl
import kerastuner
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import keras
from tensorflow_transform.tf_metadata import schema_utils
from tuner_component import component

from tensorflow_metadata.proto.v0 import schema_pb2

_FEATURE_KEYS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
_LABEL_KEY = 'variety'


# Tf.Transform considers these features as "raw"
def _get_raw_feature_spec(schema):
  return schema_utils.schema_as_feature_spec(schema).feature_spec


def _gzip_reader_fn(filenames):
  """Small utility returning a record reader that can read gzip'ed files."""
  return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def _pack(features: Dict[Text, tf.Tensor],
          label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
  """For reshaping the features and label to fit into the model.

  For example, assume batch size is 5,
    input features: Dict('feature1': tensor [[a], [b], [b], [a], [a]],
                         'feature2': tensor [[x], [y], [y], [z], [z]])
    input label: tensor [[1], [0], [0], [1], [1]]
  after pack (stack and reshape), the tuple will be:
    ( feature tensor [[a, x], [b, y], [b, y], [a, z], [a, z]],
      label tensor [1, 0, 0, 1, 1])

  Args:
    features: dict of feature tensor.
    label: label tensor.

  Returns:
    (features, label) tensor tuple.
  """

  return tf.reshape(
      tf.stack([features[key] for key in _FEATURE_KEYS], axis=1),
      [-1, len(_FEATURE_KEYS)]), tf.reshape(label, [-1])


def _make_input_dataset(file_pattern: Text,
                        schema: schema_pb2.Schema,
                        batch_size: int = 200) -> tf.data.Dataset:
  """Generates features and label for tuning/training.

  Args:
    file_pattern: input tfrecord file pattern.
    schema: Schema of the input data.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

  Returns:
    A tf.data.Dataset that contains features-label tuples.
  """
  feature_spec = _get_raw_feature_spec(schema)

  dataset = tf.data.experimental.make_batched_features_dataset(
      file_pattern=file_pattern,
      batch_size=batch_size,
      features=feature_spec,
      reader=_gzip_reader_fn,
      label_key=_LABEL_KEY,
      num_epochs=1)

  # The packed dataset contains ceil(record_num/batch_size) of tuples in format:
  #   (feature tensor with shape(batch_size, features_size),
  #    label tensor with shape(batch_size,))
  pack_dataset = dataset.map(_pack)

  return pack_dataset


def _build_keras_model(hparams: kerastuner.HyperParameters) -> tf.keras.Model:
  """Creates a DNN Keras model for classifying iris data.

  Args:
    hparams: Holds HyperParameters for tuning.

  Returns:
    A Keras Model.
  """
  model = keras.Sequential()
  model.add(
      keras.layers.Dense(
          8, activation='relu', input_shape=(len(_FEATURE_KEYS),)))
  for _ in range(hparams.get('num_layers')):  # pytype: disable=wrong-arg-types
    model.add(keras.layers.Dense(8, activation='relu'))
  model.add(keras.layers.Dense(3, activation='softmax'))
  model.compile(
      optimizer=keras.optimizers.Adam(hparams.get('learning_rate')),
      loss='categorical_crossentropy',
      metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')])
  absl.logging.info(model.summary())
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
      train_dataset=_make_input_dataset(train_data_pattern, schema, 10),
      eval_dataset=_make_input_dataset(eval_data_pattern, schema, 10))
