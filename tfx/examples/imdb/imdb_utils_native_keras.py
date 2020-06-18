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
"""Python source file include IMDB pipeline functions and necessary utils.

The utilities in this file are used to build a model with native Keras.
This module file will be used in Transform and generic Trainer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Text

import tensorflow as tf
from tensorflow import keras
import tensorflow_transform as tft

from tfx.components.trainer.executor import TrainerFnArgs

# There are 100 entries in the imdb_small dataset. ExampleGen splits the dataset
# with a 2:1 train-eval ratio. Batch_size is an empirically sound
# configuration.
# To train on the entire imdb dataset, please refer to imdb_dataset_utils.py
# and change the batch configuration accordingly.
_DROPOUT_RATE = 0.2
_EVAL_BATCH_SIZE = 5 
_HIDDEN_UNITS = 21 
_LABEL_KEY = "label"
_LEARNING_RATE = 1e-4
_MAX_FEATURES = 8000
_MAX_LEN = 128
_TRAIN_BATCH_SIZE = 10 
_TRAIN_EPOCHS = 10

def _gzip_reader_fn(filenames):
  """Small utility returning a record reader that can read gzip'ed files."""
  return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def _tokenize_review(review):
  """Tokenize the reviews by spliting the reviews, then constructing a
  vocabulary. Map the words to their frequency index in the vocabulary."""
  review_sparse = tf.strings.split(tf.reshape(review, [-1])).to_sparse()
  # tft.apply_vocabulary doesn't reserve 0 for oov words. In order to comply
  # with convention and use mask_zero in keras.embedding layer, manually set
  # default value to -1 and add 1 to every index.
  review_indices = tft.compute_and_apply_vocabulary(
      review_sparse,
      default_value=-1,
      top_k=_MAX_FEATURES)
  dense = tf.sparse.reset_shape(review_indices, None)
  dense = tf.sparse.to_dense(review_indices, default_value=-1)
  # TFX transform expects the transform result to be FixedLenFeature.
  padding_config = [[0, 0], [0, _MAX_LEN]]
  dense = tf.pad(
      dense,
      padding_config,
      'CONSTANT',
      -1)

  padded = tf.slice(dense, [0, 0], [-1, _MAX_LEN])
  padded += 1
  return padded

# TFX Transform will call this function.
def preprocessing_fn(inputs):
  """tf.transform's callback function for preprocessing inputs.

  Args:
    inputs: map from feature keys to raw not-yet-transformed features.

  Returns:
    Map from string feature key to transformed feature operations.
  """
  label = inputs['label']
  text = inputs['text']
  return {
      'label': label,
      'embedding_input': _tokenize_review(text)}

def _input_fn(file_pattern: List[Text],
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 200) -> tf.data.Dataset:
  """Generates features and label for tuning/training.

  Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    tf_transform_output: A TFTransformOutput.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch.

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
      label_key=_LABEL_KEY)

  return dataset

def _build_keras_model() -> keras.Model:
  """Creates a DNN Keras model for classifying imdb data.

  Returns:
    A Keras Model.
  """
  # The model below is built with Functional API, please refer to
  # https://www.tensorflow.org/guide/keras/overview for all API options.
  model = keras.Sequential([
      keras.layers.Embedding(
          _MAX_FEATURES+1,
          _HIDDEN_UNITS,
          mask_zero=True),
      keras.layers.Bidirectional(keras.layers.LSTM(
          _HIDDEN_UNITS,
          dropout=_DROPOUT_RATE,
          recurrent_dropout=_DROPOUT_RATE)),
      keras.layers.Dense(1, activation='sigmoid')
  ])

  model.compile(
      loss='binary_crossentropy',
      optimizer=keras.optimizers.Adam(_LEARNING_RATE),
      metrics=['AUC', 'binary_accuracy'])

  model.summary()
  return model

def _get_serve_tf_examples_fn(model, tf_transform_output):
  """Returns a function that parses a serialized tf.Example."""
  model.tft_layer = tf_transform_output.transform_features_layer()

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    """Returns the output to be used in the serving signature."""
    feature_spec = tf_transform_output.raw_feature_spec()
    feature_spec.pop(_LABEL_KEY)
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

    transformed_features = model.tft_layer(parsed_features)
    # TODO(b/148082271): Remove this line once TFT 0.22 is used.
    transformed_features.pop(_LABEL_KEY, None)

    return model(transformed_features)

  return serve_tf_examples_fn


# TFX Trainer will call this function.
def run_fn(fn_args: TrainerFnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  train_dataset = _input_fn(
      fn_args.train_files,
      tf_transform_output,
      batch_size=_TRAIN_BATCH_SIZE)

  eval_dataset = _input_fn(
      fn_args.eval_files,
      tf_transform_output,
      batch_size=_EVAL_BATCH_SIZE)

  mirrored_strategy = tf.distribute.MirroredStrategy()
  with mirrored_strategy.scope():
    model = _build_keras_model()

  model.fit(
      train_dataset,
      epochs=_TRAIN_EPOCHS,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps)

  signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(model,
                                    tf_transform_output).get_concrete_function(
                                        tf.TensorSpec(
                                            shape=[None],
                                            dtype=tf.string,
                                            name='examples')),
  }

  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
