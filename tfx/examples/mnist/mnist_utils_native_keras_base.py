# Lint as: python2, python3
# Copyright 2020 Google LLC. All Rights Reserved.
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
"""Base Python source file for MNIST utils.

This file is used by both mnist_utils_native_keras and
mnist_util_native_keras_lite to build Keras and TFLite models, respectively.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Text

import absl
import tensorflow as tf
import tensorflow_transform as tft

# MNIST dataset consists of an image of the handwritten digits,
# and it's label which is the class indicating digits 0 through 9.
IMAGE_KEY = 'image_floats'
LABEL_KEY = 'image_class'


def transformed_name(key):
  return key + '_xf'


def _gzip_reader_fn(filenames):
  """Small utility returning a record reader that can read gzip'ed files."""
  return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def input_fn(file_pattern: List[Text],
             tf_transform_output: tft.TFTransformOutput,
             batch_size: int = 200) -> tf.data.Dataset:
  """Generates features and label for tuning/training.

  Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    tf_transform_output: A TFTransformOutput.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
  transformed_feature_spec = (
      tf_transform_output.transformed_feature_spec().copy())

  dataset = tf.data.experimental.make_batched_features_dataset(
      file_pattern=file_pattern,
      batch_size=batch_size,
      features=transformed_feature_spec,
      reader=_gzip_reader_fn,
      label_key=transformed_name(LABEL_KEY))

  return dataset


def build_keras_model() -> tf.keras.Model:
  """Creates a DNN Keras model for classifying MNIST data.

  Returns:
    A Keras Model.
  """
  # The model below is built with Sequential API, please refer to
  # https://www.tensorflow.org/guide/keras/overview for all API options.
  model = tf.keras.Sequential()
  model.add(
      tf.keras.layers.InputLayer(
          input_shape=(784,), name=transformed_name(IMAGE_KEY)))
  model.add(tf.keras.layers.Dense(64, activation='relu'))
  model.add(tf.keras.layers.Dropout(0.2))
  model.add(tf.keras.layers.Dense(64, activation='relu'))
  model.add(tf.keras.layers.Dropout(0.2))
  model.add(tf.keras.layers.Dense(10, activation='softmax'))
  model.compile(
      loss='sparse_categorical_crossentropy',
      optimizer=tf.keras.optimizers.RMSprop(lr=0.0015),
      metrics=['sparse_categorical_accuracy'])
  model.summary(print_fn=absl.logging.info)
  return model


# TFX Transform will call this function.
def preprocessing_fn(inputs):
  """tf.transform's callback function for preprocessing inputs.

  Args:
    inputs: map from feature keys to raw not-yet-transformed features.

  Returns:
    Map from string feature key to transformed feature operations.
  """
  outputs = {}

  # The input float values for the image encoding are in the range [-0.5, 0.5].
  # So scale_by_min_max is a identity operation, since the range is preserved.
  outputs[transformed_name(IMAGE_KEY)] = (
      tft.scale_by_min_max(inputs[IMAGE_KEY], -0.5, 0.5))
  # TODO(b/157064428): Support label transformation for Keras.
  # Do not apply label transformation as it will result in wrong evaluation.
  outputs[transformed_name(LABEL_KEY)] = inputs[LABEL_KEY]

  return outputs
