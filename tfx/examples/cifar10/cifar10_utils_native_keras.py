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
"""Python source file includes CIFAR10 utils for Keras model.

The utilities in this file are used to build a model with native Keras.
This module file will be used in Transform and generic Trainer.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Text

import absl
import tensorflow as tf
import tensorflow_transform as tft

from tfx.components.trainer.executor import TrainerFnArgs

# cifar10 dataset has 50000 train records, and 10000 val records
_TRAIN_DATA_SIZE = 50000
_EVAL_DATA_SIZE = 10000
_TRAIN_BATCH_SIZE = 64
_EVAL_BATCH_SIZE = 64

IMAGE_KEY = 'image'
LABEL_KEY = 'label'

def transformed_name(key):
  return key + '_xf'

def _gzip_reader_fn(filenames):
  """Small utility returning a record reader that can read gzip'ed files."""
  return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def _get_serve_tf_examples_fn(model, tf_transform_output):
  """Returns a function that parses a serialized tf.Example."""

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    """Returns the output to be used in the serving signature."""
    feature_spec = tf_transform_output.raw_feature_spec()
    feature_spec.pop(LABEL_KEY)
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

    transformed_features = tf_transform_output.transform_raw_features(
        parsed_features)
    # TODO(b/148082271): Remove this line once TFT 0.22 is used.
    transformed_features.pop(transformed_name(LABEL_KEY), None)

    return model(transformed_features)

  return serve_tf_examples_fn

def _input_fn(file_pattern: List[Text],
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

def _build_keras_model() -> tf.keras.Model:
  """Creates a MobileNet model pretrained on ImageNet for classifying
  CIFAR10 images.

  Returns:
    A Keras Model.
  """
  tf.keras.backend.set_image_data_format('channels_last')
  base_model = tf.keras.applications.MobileNet(
      input_shape=(224, 224, 3), include_top=False, weights='imagenet',
      pooling='avg')

  # Freeze all layers in the base model except last conv block
  for layer in base_model.layers:
    if '13' not in layer.name:
      layer.trainable = False

  model = tf.keras.Sequential([
      tf.keras.layers.InputLayer(
          input_shape=(224, 224, 3), name=transformed_name(IMAGE_KEY)),
      base_model,
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Dense(10, activation='softmax')
  ])

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

  image_features = tf.map_fn(lambda x: tf.io.decode_png(x[0], channels=3), inputs[IMAGE_KEY], dtype=tf.uint8)
  image_features = tf.cast(image_features, tf.float32)

  # The MobileNet we use was trained on ImageNet, which has image size 224 x 224.
  # We resize CIFAR10 images to match that size
  image_features = tf.image.resize(image_features, [224, 224])

  image_features = tf.ensure_shape(image_features, (None, 224, 224, 3))
  image_features = tf.map_fn(tf.keras.applications.mobilenet.preprocess_input,
                             image_features, dtype=tf.float32)

  outputs[transformed_name(IMAGE_KEY)] = (image_features)
  # TODO(b/157064428): Support label transformation for Keras.
  # Do not apply label transformation as it will result in wrong evaluation.
  outputs[transformed_name(LABEL_KEY)] = inputs[LABEL_KEY]

  return outputs

# TFX Trainer will call this function.
def run_fn(fn_args: TrainerFnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  train_dataset = _input_fn(fn_args.train_files, tf_transform_output,
                            _TRAIN_BATCH_SIZE)
  eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output,
                           _EVAL_BATCH_SIZE)

  mirrored_strategy = tf.distribute.MirroredStrategy()
  with mirrored_strategy.scope():
    model = _build_keras_model()

  steps_per_epoch = _TRAIN_DATA_SIZE / _TRAIN_BATCH_SIZE
  validation_steps = _EVAL_DATA_SIZE / _EVAL_BATCH_SIZE

  model.fit(
      train_dataset,
      epochs=int(fn_args.train_steps / steps_per_epoch),
      steps_per_epoch=steps_per_epoch,
      validation_data=eval_dataset,
      validation_steps=validation_steps
      )

  signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(
              model, tf_transform_output).get_concrete_function(
                  tf.TensorSpec(shape=[None], dtype=tf.string, name='examples'))
  }
  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
