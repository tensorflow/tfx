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
"""Build configurable fine-tuning Bert models for various tasks"""

import tensorflow.keras as keras
import tensorflow as tf

def build_bert_classifier(
    bert_layer,
    max_len,
    num_classes,
    drop_out_rate=0.1,
    activation='linear'):
  """Bert Keras model for classification.

  Connect configurable fully connected layers on top of the Bert
  pooled_output.

  Args:
    bert_layer: A tensorflow_hub.KerasLayer intence of Bert layer.
    max_len: The maximum length of preprocessed tokens.
    num_classes: Number of unique classes in the labels. Determines the output
      shape of the classification layer.
    drop_out_rate: Dropout rate to be used for the classification layer.
    activation: The activation function used for the classification layer.
      Default to linear because use from_logits in the loss function ensures
      numerical stability.

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
  fully_connected = keras.layers.Dropout(drop_out_rate)(pooled_output)
  output = keras.layers.Dense(
      num_classes,
      activation=activation)(fully_connected)
  model = keras.Model(input_layers, output)
  return model

def compile_bert_classifier(
    model,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    learning_rate=2e-5,
    metrics=None):
  """Compile the bert classifier using suggested parameters.

  Args:
    model: A keras model. Most likely the output of build_bert_classifier.
    loss: tf.keras.losses. The suggested loss function here expects the lables
      to be sparseCategorial i.e 0, 1, 2. If the labels are one-hot encoding,
      should consider using tf.keras.losses.CategoricalCrossentropy. Output of
      the model is expected to be dim=num_classes.
    learning_rate: Suggested learning rate to be used in
      tf.keras.optimizer.Adam. The three suggested learning_rates for
      fine-tuning are [2e-5, 3e-5,5e-5].
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
  """Build and compile keras bert classification model.

  Apart from the necessary inputs, use default/suggested parameters in build
  and compile bert classifier functions.

  Args:
    bert_layer: A tensorflow_hub.KerasLayer intence of Bert layer.
    max_len: The maximum length of preprocessed tokens.
    num_classes: Number of unique classes in the labels. Determines the output
      shape of the classification layer.
    learning_rate: Suggested learning rate to be used in
      tf.keras.optimizer.Adam. The three suggested learning_rates for
      fine-tuning are [2e-5, 3e-5,5e-5]
    metrics: Default None will use ['accuracy']. An array of strings or
      tf.keras.metrics.

  Returns:
      A compile keras Bert Classification model.
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
