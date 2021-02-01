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
"""TFX template penguin model.

A DNN keras model which uses features defined in features.py and network
parameters defined in constants.py.
"""

from typing import List, Text
from absl import logging
import tensorflow as tf
from tensorflow import keras
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import schema_utils

from tfx.components.trainer.executor import TrainerFnArgs
from tfx.components.trainer.fn_args_utils import DataAccessor
from tfx.experimental.templates.penguin.models import constants
from tfx.experimental.templates.penguin.models import features
from tfx.utils import io_utils
from tfx_bsl.tfxio import dataset_options

from tensorflow_metadata.proto.v0 import schema_pb2


def _get_serve_tf_examples_fn(model, schema, tf_transform_output):
  """Returns a function that parses a serialized tf.Example."""
  if tf_transform_output is None:  # Transform component is not used.
    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
      """Returns the output to be used in the serving signature."""
      feature_spec = schema_utils.schema_as_feature_spec(schema).feature_spec
      feature_spec.pop(features.LABEL_KEY)
      parsed_features = tf.io.parse_example(serialized_tf_examples,
                                            feature_spec)
      return model(parsed_features)

  else:  # Transform component exists.
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
      """Returns the output to be used in the serving signature."""
      feature_spec = tf_transform_output.raw_feature_spec()
      feature_spec.pop(features.LABEL_KEY)
      parsed_features = tf.io.parse_example(serialized_tf_examples,
                                            feature_spec)
      transformed_features = model.tft_layer(parsed_features)
      return model(transformed_features)

  return serve_tf_examples_fn


def _input_fn(file_pattern: List[Text],
              data_accessor: DataAccessor,
              schema: schema_pb2.Schema,
              label: Text,
              batch_size: int = 200) -> tf.data.Dataset:
  """Generates features and label for tuning/training.

  Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    data_accessor: DataAccessor for converting input to RecordBatch.
    schema: A schema proto of input data.
    label: Name of the label.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
  return data_accessor.tf_dataset_factory(
      file_pattern,
      dataset_options.TensorFlowDatasetOptions(
          batch_size=batch_size, label_key=label), schema).repeat()


def _build_keras_model(feature_list: List[Text]) -> tf.keras.Model:
  """Creates a DNN Keras model for classifying penguin data.

  Args:
    feature_list: List of feature names.

  Returns:
    A Keras Model.
  """
  # The model below is built with Functional API, please refer to
  # https://www.tensorflow.org/guide/keras/overview for all API options.
  inputs = [keras.layers.Input(shape=(1,), name=f) for f in feature_list]
  d = keras.layers.concatenate(inputs)
  for _ in range(constants.NUM_LAYERS):
    d = keras.layers.Dense(constants.HIDDEN_LAYER_UNITS, activation='relu')(d)
  outputs = keras.layers.Dense(
      constants.OUTPUT_LAYER_UNITS, activation='softmax')(
          d)

  model = keras.Model(inputs=inputs, outputs=outputs)
  model.compile(
      optimizer=keras.optimizers.Adam(constants.LEARNING_RATE),
      loss='sparse_categorical_crossentropy',
      metrics=[keras.metrics.SparseCategoricalAccuracy()])

  model.summary(print_fn=logging.info)
  return model


# TFX Trainer will call this function.
# TODO(step 4): Construct, train and save your model in this function.
def run_fn(fn_args: TrainerFnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """
  if fn_args.transform_output is None:  # Transform is not used.
    tf_transform_output = None
    schema = io_utils.parse_pbtxt_file(fn_args.schema_file, schema_pb2.Schema())
    feature_list = features.FEATURE_KEYS
    label_key = features.LABEL_KEY
  else:
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    schema = tf_transform_output.transformed_metadata.schema
    feature_list = [features.transformed_name(f) for f in features.FEATURE_KEYS]
    label_key = features.transformed_name(features.LABEL_KEY)

  mirrored_strategy = tf.distribute.MirroredStrategy()
  train_batch_size = (
      constants.TRAIN_BATCH_SIZE * mirrored_strategy.num_replicas_in_sync)
  eval_batch_size = (
      constants.EVAL_BATCH_SIZE * mirrored_strategy.num_replicas_in_sync)

  train_dataset = _input_fn(
      fn_args.train_files,
      fn_args.data_accessor,
      schema,
      label_key,
      batch_size=train_batch_size)
  eval_dataset = _input_fn(
      fn_args.eval_files,
      fn_args.data_accessor,
      schema,
      label_key,
      batch_size=eval_batch_size)

  with mirrored_strategy.scope():
    model = _build_keras_model(feature_list)

  # Write logs to path
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=fn_args.model_run_dir, update_freq='batch')

  model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps,
      callbacks=[tensorboard_callback])

  signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(model, schema,
                                    tf_transform_output).get_concrete_function(
                                        tf.TensorSpec(
                                            shape=[None],
                                            dtype=tf.string,
                                            name='examples')),
  }
  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
