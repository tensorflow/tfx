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

import os
import pickle
from typing import List, Text, Tuple, Union

import absl
import numpy as np
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow_transform.tf_metadata import schema_utils
from tfx.components.trainer.executor import TrainerFnArgs
from tfx.utils import io_utils
from tensorflow_metadata.proto.v0 import schema_pb2

_FEATURE_KEYS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
_LABEL_KEY = 'variety'

# Iris dataset has 150 records, and is divided to train and eval splits in 2:1
# ratio.
_TRAIN_DATA_SIZE = 100
_TRAIN_BATCH_SIZE = 20
_SHUFFLE_BUFFER = 10000


def _get_raw_feature_spec(schema):
  return schema_utils.schema_as_feature_spec(schema).feature_spec


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
    features = [feature_dict[key].numpy()
                for key in _FEATURE_KEYS]
    features = np.concatenate(features).T
    feature_list.append(features)
    label_list.append(labels)
  return np.vstack(feature_list), np.concatenate(label_list)


def _input_fn(file_pattern: Union[Text, List[Text]],
              schema: schema_pb2.Schema) -> Tuple[np.ndarray, np.ndarray]:
  """Generates features and label for tuning/training.

  Args:
    file_pattern: input tfrecord file pattern.
    schema: schema of the input data.

  Returns:
    A (features, indices) tuple where features is a matrix of features, and
      indices is a single vector of label indices.
  """
  def _parse_example(example, feature_spec):
    """Parses a tfrecord into a (features, indices) tuple of Tensors."""
    parsed_example = tf.io.parse_single_example(
        serialized=example,
        features=feature_spec)
    label = parsed_example.pop(_LABEL_KEY)
    return parsed_example, label

  filenames = tf.data.Dataset.list_files(file_pattern)
  dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP')
  feature_spec = _get_raw_feature_spec(schema)
  # TODO(b/157598676): Make AUTOTUNE the default.
  dataset = dataset.map(
      lambda x: _parse_example(x, feature_spec),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.shuffle(_SHUFFLE_BUFFER)
  return _tf_dataset_to_numpy(dataset)


# TFX Trainer will call this function.
def run_fn(fn_args: TrainerFnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """
  schema = io_utils.parse_pbtxt_file(fn_args.schema_file, schema_pb2.Schema())

  x_train, y_train = _input_fn(fn_args.train_files, schema)
  x_eval, y_eval = _input_fn(fn_args.eval_files, schema)

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

  model_path = os.path.join(fn_args.serving_model_dir, 'model.pkl')
  with tf.io.gfile.GFile(model_path, 'wb+') as f:
    pickle.dump(model, f)
