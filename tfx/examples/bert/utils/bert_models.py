# Lint as: python2, python3
# Copyright 2020 Google LLC. All Rights Reserved.
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
<<<<<<< HEAD
"""Configurable fine-tuning BERT models for various tasks"""

import tensorflow.keras as keras
import tensorflow as tf

def build_bert_classifier(
    bert_layer,
    max_len,
    num_classes,
    dropout=0.1,
    activation=None):
=======
"""Configurable fine-tuning BERT models for various tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Text, Optional, List, Union

import tensorflow as tf
import tensorflow.keras as keras


def build_bert_classifier(bert_layer: tf.keras.layers.Layer,
                          max_len: int,
                          num_classes: int,
                          dropout: float = 0.1,
                          activation: Optional[Text] = None):
>>>>>>> c7a67f4a400574a6347e8042bca7ff5cd6452f43
  """BERT Keras model for classification.

  Connect configurable fully connected layers on top of the BERT
  pooled_output.

  Args:
    bert_layer: A tensorflow_hub.KerasLayer intence of BERT layer.
    max_len: The maximum length of preprocessed tokens.
    num_classes: Number of unique classes in the labels. Determines the output
      shape of the classification layer.
<<<<<<< HEAD
    drop_out_rate: Dropout rate to be used for the classification layer.
=======
    dropout: Dropout rate to be used for the classification layer.
>>>>>>> c7a67f4a400574a6347e8042bca7ff5cd6452f43
    activation: Activation function to use. If you don't specify anything, no
      activation is applied (ie. "linear" activation: a(x) = x).

  Returns:
    A Keras model.
  """
<<<<<<< HEAD
  input_layer_names = [
      "input_word_ids",
      "input_mask",
      "segment_ids"]

  input_layers = [
      keras.layers.Input(
          shape=(max_len,),
          dtype=tf.int32,
          name=name) for name in input_layer_names]

  pooled_output, _ = bert_layer(input_layers)
  output = keras.layers.Dropout(dropout)(pooled_output)
  output = keras.layers.Dense(
      num_classes,
      activation=activation)(output)
  model = keras.Model(input_layers, output)
  return model

def compile_bert_classifier(
    model,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    learning_rate=2e-5,
    metrics=None):
=======
  input_layer_names = ["input_word_ids", "input_mask", "segment_ids"]

  input_layers = [
      keras.layers.Input(shape=(max_len,), dtype=tf.int64, name=name)
      for name in input_layer_names
  ]

  converted_layers = [tf.cast(k, tf.int32) for k in input_layers]

  pooled_output, _ = bert_layer(converted_layers)
  output = keras.layers.Dropout(dropout)(pooled_output)
  output = keras.layers.Dense(num_classes, activation=activation)(output)
  model = keras.Model(input_layers, output)
  return model


def compile_bert_classifier(
    model: tf.keras.Model,
    loss: tf.keras.losses = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True),
    learning_rate: float = 2e-5,
    metrics: List[Union[Text, tf.keras.metrics.Metric]] = None):
>>>>>>> c7a67f4a400574a6347e8042bca7ff5cd6452f43
  """Compile the BERT classifier using suggested parameters.

  Args:
    model: A keras model. Most likely the output of build_bert_classifier.
    loss: tf.keras.losses. The suggested loss function expects integer labels
      (e.g. 0, 1, 2). If the labels are one-hot encoded, consider using
      tf.keras.lossesCategoricalCrossEntropy with from_logits set to true.
    learning_rate: Suggested learning rate to be used in
      tf.keras.optimizer.Adam. The three suggested learning_rates for
      fine-tuning are [2e-5, 3e-5, 5e-5].
<<<<<<< HEAD
    metrics: Default None will use ['accuracy']. An array of strings or
      tf.keras.metrics.
=======
    metrics: Default None will use ['sparse_categorical_accuracy']. An array of
      strings or tf.keras.metrics.
>>>>>>> c7a67f4a400574a6347e8042bca7ff5cd6452f43

  Returns:
    None.
  """
  if metrics is None:
<<<<<<< HEAD
    metrics = ['accuracy']
=======
    metrics = ["sparse_categorical_accuracy"]
>>>>>>> c7a67f4a400574a6347e8042bca7ff5cd6452f43

  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate),
      loss=loss,
<<<<<<< HEAD
      metrics=metrics
  )

def build_and_compile_bert_classifier(
    bert_layer,
    max_len,
    num_classes,
    learning_rate=5e-5,
    metrics=None):
=======
      metrics=metrics)


def build_and_compile_bert_classifier(
    bert_layer: tf.keras.layers.Layer,
    max_len: int,
    num_classes: int,
    learning_rate: float = 5e-5,
    metrics: List[Union[Text, tf.keras.metrics.Metric]] = None):
>>>>>>> c7a67f4a400574a6347e8042bca7ff5cd6452f43
  """Build and compile keras BERT classification model.

  Apart from the necessary inputs, use default/suggested parameters in build
  and compile BERT classifier functions.

  Args:
    bert_layer: A tensorflow_hub.KerasLayer intence of BERT layer.
    max_len: The maximum length of preprocessed tokens.
    num_classes: Number of unique classes in the labels. Determines the output
      shape of the classification layer.
    learning_rate: Suggested learning rate to be used in
      tf.keras.optimizer.Adam. The three suggested learning_rates for
      fine-tuning are [2e-5, 3e-5,5e-5]
<<<<<<< HEAD
    metrics: Default None will use ['accuracy']. An array of strings or
      tf.keras.metrics.
=======
    metrics: Default None will use ['sparse_categorical_accuracy']. An array of
      strings or tf.keras.metrics.
>>>>>>> c7a67f4a400574a6347e8042bca7ff5cd6452f43

  Returns:
      A compiled keras BERT Classification model.
  """
  if metrics is None:
<<<<<<< HEAD
    metrics = ['accuracy']

  model = build_bert_classifier(
      bert_layer,
      max_len,
      num_classes)

  compile_bert_classifier(
      model,
      learning_rate=learning_rate,
      metrics=metrics
  )
  return model

def build_bert_question_answering(
    bert_layer,
    max_len,
    dropout=0.1,
    activation=None):
  
  input_layer_names = [
      "input_word_ids",
      "input_mask",
      "segment_ids"]

  input_layers = [
      keras.layers.Input(
          shape=(max_len,),
          dtype=tf.int32,
          name=name) for name in input_layer_names]

  _, sequence_output = bert_layer(input_layers)
  output = keras.layers.Dropout(dropout)(sequence_output)
  output = tf.keras.layers.Dense(2)(output)
  model = tf.keras.Model(input_layers, output)
  return model
=======
    metrics = ["sparse_categorical_accuracy"]

  model = build_bert_classifier(bert_layer, max_len, num_classes)

  compile_bert_classifier(model, learning_rate=learning_rate, metrics=metrics)
  return model
>>>>>>> c7a67f4a400574a6347e8042bca7ff5cd6452f43
