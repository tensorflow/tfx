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
This module file will be used in Transform and generic Trainer.
"""

from typing import List

import absl
import tensorflow as tf
from tensorflow import keras
import tensorflow_transform as tft

from tfx.components.trainer.fn_args_utils import DataAccessor
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx_bsl.tfxio import dataset_options

_FEATURE_KEY = 'text'
_LABEL_KEY = 'label'

# There are 100 entries in the imdb_small dataset. ExampleGen splits the dataset
# with a 2:1 train-eval ratio. Batch_size is an empirically sound
# configuration.
# To train on the entire imdb dataset, please refer to imdb_dataset_utils.py
# and change the batch configuration accordingly.
_DROPOUT_RATE = 0.2
_EMBEDDING_UNITS = 64
_EVAL_BATCH_SIZE = 5
_HIDDEN_UNITS = 64
_LEARNING_RATE = 1e-4
_LSTM_UNITS = 64
_VOCAB_SIZE = 8000
_MAX_LEN = 400
_TRAIN_BATCH_SIZE = 10


def _transformed_name(key, is_input=False):
  return key + ('_xf_input' if is_input else '_xf')


def _tokenize_review(review):
  """Tokenize the reviews by spliting the reviews.

  Constructing a vocabulary. Map the words to their frequency index in the
  vocabulary.

  Args:
    review: tensors containing the reviews. (batch_size/None, 1)

  Returns:
    Tokenized and padded review tensors. (batch_size/None, _MAX_LEN)
  """
  review_sparse = tf.strings.split(tf.reshape(review, [-1])).to_sparse()
  # tft.apply_vocabulary doesn't reserve 0 for oov words. In order to comply
  # with convention and use mask_zero in keras.embedding layer, set oov value
  # to _VOCAB_SIZE and padding value to -1. Then add 1 to all the tokens.
  review_indices = tft.compute_and_apply_vocabulary(
      review_sparse, default_value=_VOCAB_SIZE, top_k=_VOCAB_SIZE)
  dense = tf.sparse.to_dense(review_indices, default_value=-1)
  # TFX transform expects the transform result to be FixedLenFeature.
  padding_config = [[0, 0], [0, _MAX_LEN]]
  dense = tf.pad(dense, padding_config, 'CONSTANT', -1)
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
  return {
      _transformed_name(_LABEL_KEY):
          inputs[_LABEL_KEY],
      _transformed_name(_FEATURE_KEY, True):
          _tokenize_review(inputs[_FEATURE_KEY])
  }


def _input_fn(file_pattern: List[str],
              data_accessor: DataAccessor,
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 200) -> tf.data.Dataset:
  """Generates features and label for tuning/training.

  Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    data_accessor: DataAccessor for converting input to RecordBatch.
    tf_transform_output: A TFTransformOutput.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch.

  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
  dataset = data_accessor.tf_dataset_factory(
      file_pattern,
      dataset_options.TensorFlowDatasetOptions(
          batch_size=batch_size, label_key=_transformed_name(_LABEL_KEY)),
      tf_transform_output.transformed_metadata.schema)

  return dataset.repeat()


def _build_keras_model() -> keras.Model:
  """Creates a LSTM Keras model for classifying imdb data.

  Reference: https://www.tensorflow.org/tutorials/text/text_classification_rnn

  Returns:
    A Keras Model.
  """
  # The model below is built with Sequential API, please refer to
  # https://www.tensorflow.org/guide/keras/sequential_model
  model = keras.Sequential([
      keras.layers.Embedding(
          _VOCAB_SIZE + 2,
          _EMBEDDING_UNITS,
          name=_transformed_name(_FEATURE_KEY)),
      keras.layers.Bidirectional(
          keras.layers.LSTM(_LSTM_UNITS, dropout=_DROPOUT_RATE)),
      keras.layers.Dense(_HIDDEN_UNITS, activation='relu'),
      keras.layers.Dense(1)
  ])

  model.compile(
      loss=keras.losses.BinaryCrossentropy(from_logits=True),
      optimizer=keras.optimizers.Adam(_LEARNING_RATE),
      metrics=['accuracy'])

  model.summary(print_fn=absl.logging.info)
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
    return model(transformed_features)

  return serve_tf_examples_fn


# TFX Trainer will call this function.
def run_fn(fn_args: FnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  train_dataset = _input_fn(
      fn_args.train_files,
      fn_args.data_accessor,
      tf_transform_output,
      batch_size=_TRAIN_BATCH_SIZE)

  eval_dataset = _input_fn(
      fn_args.eval_files,
      fn_args.data_accessor,
      tf_transform_output,
      batch_size=_EVAL_BATCH_SIZE)

  mirrored_strategy = tf.distribute.MirroredStrategy()
  with mirrored_strategy.scope():
    model = _build_keras_model()

  # In distributed training, it is common to use num_steps instead of num_epochs
  # to control training.
  # Reference: https://stackoverflow.com/questions/45989971/
  # /distributed-training-with-tf-estimator-resulting-in-more-training-steps

  model.fit(
      train_dataset,
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
