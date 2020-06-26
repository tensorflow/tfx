# Lint as: python2, python3
# Copyright 2019 Google LLC. All Rights Reserved.
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
import tensorflow_addons as tfa

def BertForSingleSentenceClassification(
  bert_layer,
  max_len,
  fully_connected_layers=None):
  """Keras model for single sentence classification.
  Connect configurable fully connected layers on top of the Bert
  pooled_output.

  Args:
    bert_layer: A tensroflow_hub.KerasLayer intence of Bert layer.
    max_len: The maximum length of preprocessed tokens.
    hidden_layers: List of configurations for fine-tuning hidden layers
      after the pooled_output. [(#of hidden units, activation)].

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
      name=name) for name in input_layer_names
    ]

  pooled_output, _ = bert_layer(input_layers)

  fully_connected = pooled_output
  if fully_connected_layers is not None:
    for (i, activation) in fully_connected_layers:
      fully_connected = keras.layers.Dense(
        i,
        activation=activation
        )(fully_connected)

  output = keras.layers.Dense(1, activation='sigmoid')(fully_connected)
  model = keras.Model(input_layers, output)
  model.compile(
    optimizer=tf.keras.optimizers.Adam(5e-5),
    loss=tf.keras.losses.binary_crossentropy,
    metrics=[tfa.metrics.MatthewsCorrelationCoefficient(1)])
  return model
