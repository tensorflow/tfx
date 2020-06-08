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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Text

import absl
import tensorflow as tf
from tensorflow import keras
import tensorflow_transform as tft
import tensorflow_hub as hub

from tfx.components.trainer.executor import TrainerFnArgs

_TRAIN_BATCH_SIZE = 64 
_TRAIN_DATA_SIZE = int(50000 * 2 / 3) 
_EVAL_BATCH_SIZE = 64
_EVAL_DATA_SIZE = int(50000 / 3)
_LABEL_KEY = "sentiment"
DELIMITERS = '.,!?() '
_MAX_FEATURES = 8000 
_MAX_LEN = 100 

def _gzip_reader_fn(filenames):
  """Small utility returning a record reader that can read gzip'ed files."""
  return tf.data.TFRecordDataset(filenames, compression_type='GZIP'
          )

def _sentiment_to_int(sentiment):
    """Converting labels from string to integer"""
    ints = tf.cast(sentiment, tf.int64)
    return ints

def _tokenize_review(review):
    review_sparse = tf.compat.v1.string_split(tf.reshape(review, shape=[-1]), 
            DELIMITERS)
    review_indices = tft.compute_and_apply_vocabulary(review_sparse, 
            default_value=_MAX_FEATURES,
            top_k=_MAX_FEATURES)
    dense = tf.sparse.to_dense(review_indices, default_value=_MAX_FEATURES)
    padding_config = [[0, 0], [0, _MAX_LEN]]
    dense = tf.pad(dense, padding_config, 'CONSTANT', constant_values = _MAX_FEATURES)
    padded = tf.slice(dense, [0, 0], [-1, _MAX_LEN])
    return padded 

def preprocessing_fn(inputs):
  """tf.transform's callback function for preprocessing inputs.

  Args:
    inputs: map from feature keys to raw not-yet-transformed features.

  Returns:
    Map from string feature key to transformed feature operations.
  """
  sentiment = inputs['sentiment']
  review = inputs['review']
  return {
          'sentiment':_sentiment_to_int(sentiment),
          'embedding_input':_tokenize_review(review),
          }

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
      label_key=_LABEL_KEY)

  return dataset

def _build_keras_model() -> tf.keras.Model:
  """Creates a DNN Keras model for classifying imdb data.

  Returns:
    A Keras Model.
  """
  # The model below is built with Functional API, please refer to
  # https://www.tensorflow.org/guide/keras/overview for all API options.
  model = tf.keras.Sequential([
      tf.keras.layers.Embedding(_MAX_FEATURES+1, 64),
      tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, 
          dropout=0.2,
          recurrent_dropout=0.2)),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(1)
  ])
  model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])
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

  train_dataset = _input_fn(fn_args.train_files, tf_transform_output,
                            batch_size=_TRAIN_BATCH_SIZE)
  eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output,
                           batch_size=_EVAL_BATCH_SIZE)
  steps_per_epoch = _TRAIN_DATA_SIZE / _TRAIN_BATCH_SIZE
  eval_steps = _EVAL_DATA_SIZE / _EVAL_BATCH_SIZE

  mirrored_strategy = tf.distribute.MirroredStrategy()
  with mirrored_strategy.scope():
      model = _build_keras_model()

  model.fit(
      train_dataset,
      epochs=10,
      steps_per_epoch=steps_per_epoch,
      validation_data=eval_dataset,
      validation_steps=eval_steps)

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
