# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Python source file include Penguin pipeline functions and necessary utils.

The utilities in this file are used to build a model with scikit-learn.
This module file will be used in Transform and generic Trainer.
"""

import os
import pickle
from typing import Tuple

import absl
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tfx.components.trainer.fn_args_utils import DataAccessor
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.dsl.io import fileio
from tfx.utils import io_utils
from tfx_bsl.tfxio import dataset_options

from tensorflow_metadata.proto.v0 import schema_pb2

_FEATURE_KEYS = [
    'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g'
]
_LABEL_KEY = 'species'

# The Penguin dataset has 342 records, and is divided into train and eval
# splits in a 2:1 ratio.
_TRAIN_DATA_SIZE = 228
_TRAIN_BATCH_SIZE = 20


def _input_fn(
    file_pattern: str,
    data_accessor: DataAccessor,
    schema: schema_pb2.Schema,
    batch_size: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
  """Generates features and label for tuning/training.

  Args:
    file_pattern: input tfrecord file pattern.
    data_accessor: DataAccessor for converting input to RecordBatch.
    schema: schema of the input data.
    batch_size: An int representing the number of records to combine in a single
      batch.

  Returns:
    A (features, indices) tuple where features is a matrix of features, and
      indices is a single vector of label indices.
  """
  record_batch_iterator = data_accessor.record_batch_factory(
      file_pattern,
      dataset_options.RecordBatchesOptions(batch_size=batch_size, num_epochs=1),
      schema)

  feature_list = []
  label_list = []
  for record_batch in record_batch_iterator:
    record_dict = {}
    for column, field in zip(record_batch, record_batch.schema):
      record_dict[field.name] = column.flatten()

    label_list.append(record_dict[_LABEL_KEY])
    features = [record_dict[key] for key in _FEATURE_KEYS]
    feature_list.append(np.stack(features, axis=-1))

  return np.concatenate(feature_list), np.concatenate(label_list)


# TFX Trainer will call this function.
def run_fn(fn_args: FnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """
  schema = io_utils.parse_pbtxt_file(fn_args.schema_file, schema_pb2.Schema())

  x_train, y_train = _input_fn(fn_args.train_files, fn_args.data_accessor,
                               schema)
  x_eval, y_eval = _input_fn(fn_args.eval_files, fn_args.data_accessor, schema)

  steps_per_epoch = _TRAIN_DATA_SIZE / _TRAIN_BATCH_SIZE

  estimator = MLPClassifier(
      hidden_layer_sizes=[8, 8, 8],
      activation='relu',
      solver='adam',
      batch_size=_TRAIN_BATCH_SIZE,
      learning_rate_init=0.0005,
      max_iter=int(fn_args.train_steps / steps_per_epoch),
      verbose=True)

  # Create a pipeline that standardizes the input data before passing it to an
  # estimator. Once the scaler is fit, it will use the same mean and stdev to
  # transform inputs at both training and serving time.
  model = Pipeline([
      ('scaler', StandardScaler()),
      ('estimator', estimator),
  ])
  model.feature_keys = _FEATURE_KEYS
  model.label_key = _LABEL_KEY
  model.fit(x_train, y_train)
  absl.logging.info(model)

  score = model.score(x_eval, y_eval)
  absl.logging.info('Accuracy: %f', score)

  # Export the model as a pickle named model.pkl. AI Platform Prediction expects
  # sklearn model artifacts to follow this naming convention.
  os.makedirs(fn_args.serving_model_dir)

  model_path = os.path.join(fn_args.serving_model_dir, 'model.pkl')
  with fileio.open(model_path, 'wb+') as f:
    pickle.dump(model, f)
