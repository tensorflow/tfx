# Copyright 2021 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Python source file includes pipeline functions and necessary utils."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import List, Text

import absl
import tensorflow as tf
from tensorflow import keras
import tensorflow_transform as tft

from tfx.components.trainer.rewriting import converters
from tfx.components.trainer.rewriting import rewriter
from tfx.components.trainer.rewriting import rewriter_factory
from tfx.dsl.io import fileio

from tfx import v1 as tfx  # pylint: disable=g-bad-import-order

from tfx_bsl.public import tfxio

_CUR_PAGE_FEATURE_KEY = 'cur_page'
_SESSION_INDEX_FEATURE_KEY = 'session_index'
_LABEL_KEY = 'label'
_VOCAB_FILENAME = 'vocab'

_TOP_K = 100
_EMBEDDING_DIM = 10
_UNITS = 50

_TRAIN_BATCH_SIZE = 32
_EVAL_BATCH_SIZE = 16


# TFX Transform will call this function.
def preprocessing_fn(inputs):
  """Callback function for preprocessing inputs.

  Args:
    inputs: map from feature keys to raw not-yet-transformed features.

  Returns:
    Map from string feature key to transformed feature operations.
  """
  outputs = inputs.copy()

  # Compute a vocabulary based on the TOP-K current pages and labels seen in
  # the dataset.
  vocab = tft.vocabulary(
      tf.concat([inputs[_CUR_PAGE_FEATURE_KEY], inputs[_LABEL_KEY]], axis=0),
      top_k=_TOP_K,
      vocab_filename=_VOCAB_FILENAME)

  # Apply the vocabulary to both the current page feature and the label,
  # converting the strings into integers.
  for k in [_CUR_PAGE_FEATURE_KEY, _LABEL_KEY]:
    # Out-of-vocab strings will be assigned the _TOP_K value.
    outputs[k] = tft.apply_vocabulary(inputs[k], vocab, default_value=_TOP_K)
  return outputs


def _input_fn(file_pattern: List[Text],
              data_accessor: tfx.components.DataAccessor,
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
      tfxio.TensorFlowDatasetOptions(
          batch_size=batch_size, label_key=_LABEL_KEY),
      tf_transform_output.transformed_metadata.schema)

  return dataset.repeat()


def _build_keras_model() -> keras.Model:
  """Creates a Keras model for predicting the next page.

  Returns:
    A Keras Model.
  """
  # This model has two inputs: (i) current page and (ii) session index.
  cur_page_input = keras.Input(shape=(), name=_CUR_PAGE_FEATURE_KEY)
  session_index_input = keras.Input(shape=(1,), name=_SESSION_INDEX_FEATURE_KEY)
  inputs = [cur_page_input, session_index_input]

  # Create an embedding for the current page.
  cur_page_emb = keras.layers.Embedding(
      _TOP_K + 1, _EMBEDDING_DIM, input_length=1)(
          cur_page_input)
  x = keras.layers.Concatenate()([cur_page_emb, session_index_input])
  x = keras.layers.Dense(_UNITS, activation='relu')(x)
  outputs = keras.layers.Dense(_TOP_K + 1, activation='softmax')(x)
  model = keras.Model(inputs=inputs, outputs=outputs)
  model.compile(
      loss=keras.losses.SparseCategoricalCrossentropy(),
      optimizer=keras.optimizers.Adam(0.0001),
      metrics=[
          'sparse_categorical_accuracy', 'sparse_top_k_categorical_accuracy'
      ])

  model.summary(print_fn=absl.logging.info)
  return model


# The inference function assumes that the mapping from string to integer for
# the current page has been done outside of the model. We store the vocabulary
# file with the output tfjs model to simplify this process.
def _get_inference_fn(model, tf_transform_output):
  """Defines the function used for inference."""
  model.tft_layer = tf_transform_output.transform_features_layer()

  @tf.function
  def inference_fn(cur_page, session_index):
    """Returns the output to be used in the serving signature."""
    return model({
        _CUR_PAGE_FEATURE_KEY: cur_page,
        _SESSION_INDEX_FEATURE_KEY: session_index
    })

  return inference_fn


# TFX Trainer will call this function.
def run_fn(fn_args: tfx.components.FnArgs):
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

  model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps,
      verbose=2)

  signatures = {
      'serving_default':
          _get_inference_fn(model, tf_transform_output).get_concrete_function(
              tf.TensorSpec(
                  shape=[None], dtype=tf.int64, name=_CUR_PAGE_FEATURE_KEY),
              tf.TensorSpec(
                  shape=[None], dtype=tf.int64,
                  name=_SESSION_INDEX_FEATURE_KEY)),
  }

  # Create the saved_model in a temporary directory.
  temp_saving_model_dir = os.path.join(fn_args.serving_model_dir, 'temp')
  model.save(temp_saving_model_dir, save_format='tf', signatures=signatures)

  # Convert the saved_model to a tfjs model and store it in the final directory.
  tfrw = rewriter_factory.create_rewriter(
      rewriter_factory.TFJS_REWRITER, name='tfjs_rewriter')
  converters.rewrite_saved_model(temp_saving_model_dir,
                                 fn_args.serving_model_dir, tfrw,
                                 rewriter.ModelType.TFJS_MODEL)

  # Copy the vocabulary computed by transform to the final directory.
  # The vocabulary is not included in the original savedmodel because vocab
  # lookups are currently not supported in TFJS and are expected to be done
  # independently by client code.
  fileio.copy(
      tf_transform_output.vocabulary_file_by_name(_VOCAB_FILENAME),
      os.path.join(fn_args.serving_model_dir, _VOCAB_FILENAME))

  fileio.rmtree(temp_saving_model_dir)
