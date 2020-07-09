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

import os
import absl
import tensorflow as tf
import tensorflow_transform as tft

from tfx.components.trainer.rewriting import converters
from tfx.components.trainer.rewriting import rewriter
from tfx.components.trainer.rewriting import rewriter_factory

from tfx.components.trainer.executor import TrainerFnArgs

# CIFAR10 dataset has 50000 train records, and 10000 test records
_TRAIN_DATA_SIZE = 50000
_EVAL_DATA_SIZE = 10000
_TRAIN_BATCH_SIZE = 64
_EVAL_BATCH_SIZE = 64

# _TRAIN_DATA_SIZE = 100
# _EVAL_DATA_SIZE = 100
# _TRAIN_BATCH_SIZE = 32
# _EVAL_BATCH_SIZE = 32

IMAGE_KEY = 'image'
LABEL_KEY = 'label'

def transformed_name(key):
  return key + '_xf'

def _gzip_reader_fn(filenames):
  """Small utility returning a record reader that can read gzip'ed files."""
  return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def _get_serve_tf_examples_fn(model):
  """Returns a function that feeds the input tensor into the model."""

  @tf.function
  def serve_tf_examples_fn(image_tensor):
    """Returns the output to be used in the serving signature."""
    return model(image_tensor)

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

def _freeze_model_by_percentage(model: tf.keras.Model,
                                percentage: float):
  """Freeze part of the model based on specified percentage

  Args:
    model: The keras model need to be partially frozen
    percentage: the percentage of layers to freeze
  """
  if percentage < 0 or percentage > 1:
    raise Exception("freeze percentage should between 0.0 and 1.0")

  num_layers = len(model.layers)
  num_layers_to_freeze = int(num_layers * percentage)
  for idx, layer in enumerate(model.layers):
    if idx < num_layers_to_freeze:
      layer.trainable = False
    else:
      layer.trainable = True

def _build_keras_model() -> tf.keras.Model:
  """Creates a MobileNet model pretrained on ImageNet for finetuning

  Args:
    model: The keras model need to be partially frozen
  Returns:
    A Keras Model.
  """
  base_model = tf.keras.applications.MobileNet(
      input_shape=(224, 224, 3), include_top=False, weights='imagenet',
      pooling='avg')

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

def _decode_and_preprocess_image(image_string):
  """Decodes the raw image string and preprocess it

  Args:
    image_string: The encoded raw image string
  Returns:
    The preprocessed image tensor
  """
  image = tf.io.decode_png(image_string, channels=3)
  image = tf.cast(image, tf.float32)

  # The MobileNet we use was trained on ImageNet, which has image size 224 x 224.
  # We resize CIFAR10 images to match that size
  image = tf.image.resize(image, [224, 224])
  image = tf.keras.applications.mobilenet.preprocess_input(image)
  return image

# TFX Transform will call this function.
def preprocessing_fn(inputs):
  """tf.transform's callback function for preprocessing inputs.

  Args:
    inputs: map from feature keys to raw not-yet-transformed features.

  Returns:
    Map from string feature key to transformed feature operations.
  """
  outputs = {}

  image_features = tf.map_fn(lambda x: _decode_and_preprocess_image(x[0]),
                             inputs[IMAGE_KEY], dtype=tf.float32)

  outputs[transformed_name(IMAGE_KEY)] = image_features
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
  total_epochs = int(fn_args.train_steps / steps_per_epoch)
  classifier_epochs = int(total_epochs / 2)
  finetune_epochs = total_epochs - classifier_epochs

  # Freeze the whole MobileNet backbone and train the top classifer only
  base_model = model.get_layer('mobilenet_1.00_224')
  _freeze_model_by_percentage(base_model, 1.0)
  # We need to recompile the model because layer properties have changed
  model.compile(
      loss='sparse_categorical_crossentropy',
      optimizer=tf.keras.optimizers.RMSprop(lr=0.0015),
      metrics=['sparse_categorical_accuracy'])
  model.summary(print_fn=absl.logging.info)

  model.fit(
      train_dataset,
      epochs=classifier_epochs,
      steps_per_epoch=steps_per_epoch,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps
  )

  # Optional
  # Unfreeze the top MobileNet layers and finetune the whole model
  base_model = model.get_layer('mobilenet_1.00_224')
  _freeze_model_by_percentage(base_model, 0.9)
  # We need to recompile the model because layer properties have changed
  model.compile(
      loss='sparse_categorical_crossentropy',
      optimizer=tf.keras.optimizers.RMSprop(lr=0.0005),
      metrics=['sparse_categorical_accuracy'])
  model.summary(print_fn=absl.logging.info)

  model.fit(
      train_dataset,
      epochs=finetune_epochs,
      steps_per_epoch=steps_per_epoch,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps
  )

  signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(
              model).get_concrete_function(
                  tf.TensorSpec(
                      shape=[None, 224, 224, 3],
                      dtype=tf.float32,
                      name=transformed_name(IMAGE_KEY)
                      ))
  }

  temp_saving_model_dir = os.path.join(fn_args.serving_model_dir, 'temp')
  model.save(temp_saving_model_dir, save_format='tf', signatures=signatures)

  tfrw = rewriter_factory.create_rewriter(
      rewriter_factory.TFLITE_REWRITER, name='tflite_rewriter', #filename='cifar10.tflite',
      enable_experimental_new_converter=True)
  converters.rewrite_saved_model(temp_saving_model_dir,
                                 fn_args.serving_model_dir,
                                 tfrw,
                                 rewriter.ModelType.TFLITE_MODEL
                                 )

  tf.io.gfile.rmtree(temp_saving_model_dir)
