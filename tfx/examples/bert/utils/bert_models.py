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
"""Configurable fine-tuning BERT models for various tasks"""

import tensorflow.keras as keras
import tensorflow as tf

def build_bert_classifier(
    bert_layer,
    max_len,
    num_classes,
    dropout=0.1,
    activation=None):
  """BERT Keras model for classification.

  Connect configurable fully connected layers on top of the BERT
  pooled_output.

  Args:
    bert_layer: A tensorflow_hub.KerasLayer intence of BERT layer.
    max_len: The maximum length of preprocessed tokens.
    num_classes: Number of unique classes in the labels. Determines the output
      shape of the classification layer.
    drop_out_rate: Dropout rate to be used for the classification layer.
    activation: Activation function to use. If you don't specify anything, no
      activation is applied (ie. "linear" activation: a(x) = x).

  Returns:
    A Keras model.
  """
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
  """Compile the BERT classifier using suggested parameters.

  Args:
    model: A keras model. Most likely the output of build_bert_classifier.
    loss: tf.keras.losses. The suggested loss function expects integer labels
      (e.g. 0, 1, 2). If the labels are one-hot encoded, consider using
      tf.keras.lossesCategoricalCrossEntropy with from_logits set to true.
    learning_rate: Suggested learning rate to be used in
      tf.keras.optimizer.Adam. The three suggested learning_rates for
      fine-tuning are [2e-5, 3e-5, 5e-5].
    metrics: Default None will use ['accuracy']. An array of strings or
      tf.keras.metrics.

  Returns:
    None.
  """
  if metrics is None:
    metrics = ['accuracy']

  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate),
      loss=loss,
      metrics=metrics
  )

def build_and_compile_bert_classifier(
    bert_layer,
    max_len,
    num_classes,
    learning_rate=5e-5,
    metrics=None):
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
    metrics: Default None will use ['accuracy']. An array of strings or
      tf.keras.metrics.

  Returns:
      A compiled keras BERT Classification model.
  """
  if metrics is None:
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
  output = tf.keras.layers.Dense(2, dropout=dropout)(sequence_output)
  model = tf.keras.Model(input_layers, output)
  return model