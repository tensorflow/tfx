# Lint as: python3
# Copyright 2020 Google LLC. All Rights Reserved.
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
"""Python source file include Iris pipeline functions and necessary utils.

The utilities in this file are used to build a model with scikit-learn.
This module file will be used in Transform and generic Trainer.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
from typing import Text, Tuple

import absl
import numpy as np
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
import tensorflow_transform as tft

from tfx.components.trainer.executor import TrainerFnArgs

_FEATURE_KEYS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
_LABEL_KEY = 'variety'

# Iris dataset has 150 records, and is divided to train and eval splits in 2:1
# ratio.
_TRAIN_DATA_SIZE = 100
_TRAIN_BATCH_SIZE = 20
_SHUFFLE_BUFFER = 10000


def _transformed_name(key):
  return key + '_xf'


# TODO(b/153996019): This function will no longer be needed once feature is
# added to return entire dataset in pyarrow format.
def _tf_dataset_to_numpy(dataset: tf.data.Dataset,
                         ) -> Tuple[np.ndarray, np.ndarray]:
  """Converts a tf.data.dataset into features and labels.

  Args:
    dataset: A tf.data.dataset that contains (features, indices) tuple where
      features is a dictionary of Tensors, and indices is a single Tensor of
      label indices.

  Returns:
    A (features, indices) tuple where features is a matrix of features, and
      indices is a single vector of label indices.
  """
  feature_list = []
  label_list = []
  for feature_dict, labels in dataset:
    features = [feature_dict[_transformed_name(key)].numpy()
                for key in _FEATURE_KEYS]
    features = np.concatenate(features).T
    feature_list.append(features)
    label_list.append(labels)
  return np.vstack(feature_list), np.concatenate(label_list)


# TODO(b/157054136): Use TFXIO to read examples in the input function.
def _input_fn(file_pattern: Text, tf_transform_output: tft.TFTransformOutput,
              ) -> Tuple[np.ndarray, np.ndarray]:
  """Generates features and label for tuning/training.

  Args:
    file_pattern: input tfrecord file pattern.
    tf_transform_output: A TFTransformOutput.

  Returns:
    A (features, indices) tuple where features is a matrix of features, and
      indices is a single vector of label indices.
  """
  def _parse_example(example):
    """Parses a tfrecord into a (features, indices) tuple of Tensors."""
    parsed_example = tf.io.parse_single_example(
        serialized=example,
        features=tf_transform_output.transformed_feature_spec())
    label = parsed_example.pop(_transformed_name(_LABEL_KEY))
    return parsed_example, label

  filenames = tf.data.Dataset.list_files(file_pattern)
  dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP')
  # TODO(b/157598676): Make AUTOTUNE the default.
  dataset = dataset.map(
      _parse_example,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.shuffle(_SHUFFLE_BUFFER)
  return _tf_dataset_to_numpy(dataset)


# TFX Transform will call this function.
def preprocessing_fn(inputs):
  """tf.transform's callback function for preprocessing inputs.

  Args:
    inputs: map from feature keys to raw not-yet-transformed features.

  Returns:
    Map from string feature key to transformed feature operations.
  """
  outputs = {}

  for key in _FEATURE_KEYS:
    std_input = tft.scale_to_z_score(inputs[key])
    assert isinstance(std_input, tf.Tensor)
    outputs[_transformed_name(key)] = std_input
  outputs[_transformed_name(_LABEL_KEY)] = inputs[_LABEL_KEY]

  return outputs


# TFX Trainer will call this function.
def run_fn(fn_args: TrainerFnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  x_train, y_train = _input_fn(fn_args.train_files, tf_transform_output)
  x_eval, y_eval = _input_fn(fn_args.eval_files, tf_transform_output)

  steps_per_epoch = _TRAIN_DATA_SIZE / _TRAIN_BATCH_SIZE

  model = MLPClassifier(
      hidden_layer_sizes=[8, 8, 8],
      activation='relu',
      solver='adam',
      batch_size=_TRAIN_BATCH_SIZE,
      learning_rate_init=0.0005,
      max_iter=int(fn_args.train_steps / steps_per_epoch),
      verbose=True)
  model.fit(x_train, y_train)
  absl.logging.info(model)

  score = model.score(x_eval, y_eval)
  absl.logging.info('Accuracy: %f', score)

  os.makedirs(fn_args.serving_model_dir)

  # TODO(humichael): Export TFT graph for serving once a solution for serving is
  # determined.
  model_path = os.path.join(fn_args.serving_model_dir, 'model.pkl')
  with tf.io.gfile.GFile(model_path, 'wb+') as f:
    pickle.dump(model, f)
