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
"""Python source file include Iris pipeline functions and necessary utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Text, Union

import absl
import kerastuner
import tensorflow as tf
from tensorflow import keras
from tensorflow_transform.tf_metadata import schema_utils

from tensorflow_metadata.proto.v0 import schema_pb2
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.components.tuner.component import TunerFnResult
from tfx.utils import io_utils

_FEATURE_KEYS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
_LABEL_KEY = 'variety'


def _gzip_reader_fn(filenames):
  """Small utility returning a record reader that can read gzip'ed files."""
  return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def _input_fn(file_pattern: Union[Text, List[Text]],
              schema: schema_pb2.Schema,
              batch_size: int = 20) -> tf.data.Dataset:
  """Generates features and label for tuning/training.

  Args:
    file_pattern: string or list of strings, contains pattern(s) of input
      tfrecord files.
    schema: Schema of the input data.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch.

  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
  feature_spec = schema_utils.schema_as_feature_spec(schema).feature_spec

  dataset = tf.data.experimental.make_batched_features_dataset(
      file_pattern=file_pattern,
      batch_size=batch_size,
      features=feature_spec,
      reader=_gzip_reader_fn,
      label_key=_LABEL_KEY)

  return dataset


def _build_keras_model(hparams: kerastuner.HyperParameters) -> tf.keras.Model:
  """Creates a DNN Keras model for classifying iris data.

  Args:
    hparams: Holds HyperParameters for tuning.

  Returns:
    A Keras Model.
  """
  # The model below is built with Functional API, please refer to
  # https://www.tensorflow.org/guide/keras/overview for all API options.
  inputs = [keras.layers.Input(shape=(1,), name=f) for f in _FEATURE_KEYS]
  d = keras.layers.concatenate(inputs)
  for _ in range(int(hparams.get('num_layers'))):
    d = keras.layers.Dense(8, activation='relu')(d)
  outputs = keras.layers.Dense(3, activation='softmax')(d)

  model = keras.Model(inputs=inputs, outputs=outputs)
  model.compile(
      optimizer=keras.optimizers.Adam(hparams.get('learning_rate')),
      loss='sparse_categorical_crossentropy',
      metrics=[keras.metrics.SparseCategoricalAccuracy()])

  model.summary(print_fn=absl.logging.info)
  return model


# This will be called by TFX Tuner.
def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
  """Build the tuner using the KerasTuner API.

  Args:
    fn_args: Holds args as name/value pairs.
      - working_dir: working dir for tuning.
      - train_files: List of file paths containing training tf.Example data.
      - eval_files: List of file paths containing eval tf.Example data.
      - train_steps: number of train steps.
      - eval_steps: number of eval steps.
      - schema_path: optional schema of the input data.
      - transform_graph_path: optional transform graph produced by TFT.

  Returns:
    A namedtuple contains the following:
      - tuner: A BaseTuner that will be used for tuning.
      - fit_kwargs: Args to pass to tuner's run_trial function for fitting the
                    model , e.g., the training and validation dataset. Required
                    args depend on the above tuner's implementation.
  """
  hp = kerastuner.HyperParameters()
  # Defines search space.
  hp.Choice('learning_rate', [1e-1, 1e-3])
  hp.Int('num_layers', 1, 5)

  # RandomSearch is a subclass of Keras model Tuner.
  tuner = kerastuner.RandomSearch(
      _build_keras_model,
      max_trials=5,
      hyperparameters=hp,
      allow_new_entries=False,
      objective='val_sparse_categorical_accuracy',
      directory=fn_args.working_dir,
      project_name='test')

  schema = schema_pb2.Schema()
  io_utils.parse_pbtxt_file(fn_args.schema_path, schema)
  train_dataset = _input_fn(fn_args.train_files, schema)
  eval_dataset = _input_fn(fn_args.eval_files, schema)

  return TunerFnResult(
      tuner=tuner,
      fit_kwargs={
          'x': train_dataset,
          'validation_data': eval_dataset,
          'steps_per_epoch': fn_args.train_steps,
          'validation_steps': fn_args.eval_steps
      })
