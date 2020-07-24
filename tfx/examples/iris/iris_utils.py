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
"""Python source file include Iris pipeline functions and necessary utils.

The utilities in this file are used to build a model with Keras Layers, but
uses model_to_estimator for Trainer component adaption.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import absl
import tensorflow as tf
from tensorflow import keras
import tensorflow_model_analysis as tfma
from tensorflow_transform.tf_metadata import schema_utils

from tfx.components.trainer import executor
from tfx.utils import io_utils
from tfx.utils import path_utils
from tensorflow_metadata.proto.v0 import schema_pb2

_FEATURE_KEYS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
_LABEL_KEY = 'variety'


# Tf.Transform considers these features as "raw"
def _get_raw_feature_spec(schema):
  return schema_utils.schema_as_feature_spec(schema).feature_spec


def _gzip_reader_fn(filenames):
  """Small utility returning a record reader that can read gzip'ed files."""
  return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def _serving_input_receiver_fn(schema):
  """Build the serving inputs.

  Args:
    schema: the schema of the input data.

  Returns:
    serving_input_receiver_fn for serving this model, since no transformation is
    required in this case it does not include a tf-transform graph.
  """
  raw_feature_spec = _get_raw_feature_spec(schema)
  raw_feature_spec.pop(_LABEL_KEY)

  raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
      raw_feature_spec, default_batch_size=None)
  return raw_input_fn()


def _eval_input_receiver_fn(schema):
  """Build the evalution inputs for the tf-model-analysis to run the model.

  Args:
    schema: the schema of the input data.

  Returns:
    EvalInputReceiver function, which contains:
      - Features (dict of Tensors) to be passed to the model.
      - Raw features as serialized tf.Examples.
      - Labels
  """
  # Notice that the inputs are raw features, not transformed features here.
  raw_feature_spec = _get_raw_feature_spec(schema)

  raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
      raw_feature_spec, default_batch_size=None)
  serving_input_receiver = raw_input_fn()

  labels = serving_input_receiver.features.pop(_LABEL_KEY)
  return tfma.export.EvalInputReceiver(
      features=serving_input_receiver.features,
      labels=labels,
      receiver_tensors=serving_input_receiver.receiver_tensors)


def _input_fn(filenames, schema, batch_size=200):
  """Input function for training and evaluation.

  Args:
    filenames: [str] list of CSV files to read data from.
    schema: Schema of the input data.
    batch_size: int First dimension size of the Tensors returned by input_fn

  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """

  feature_spec = _get_raw_feature_spec(schema)

  dataset = tf.data.experimental.make_batched_features_dataset(
      filenames, batch_size, feature_spec, reader=_gzip_reader_fn)

  # We pop the label because we do not want to use it as a feature while we're
  # training.
  return dataset.map(lambda features: (features, features.pop(_LABEL_KEY)))


def _keras_model_builder():
  """Creates a DNN Keras model  for classifying iris data.

  Returns:
    A keras Model.
  """
  # The model below is built with Functional API, please refer to
  # https://www.tensorflow.org/guide/keras/overview for all API options.
  inputs = [tf.keras.layers.Input(shape=(1,), name=f) for f in _FEATURE_KEYS]
  d = keras.layers.concatenate(inputs)
  for _ in range(3):
    d = keras.layers.Dense(8, activation='relu')(d)
  outputs = keras.layers.Dense(3, activation='softmax')(d)

  model = keras.Model(inputs=inputs, outputs=outputs)
  model.compile(
      optimizer=keras.optimizers.Adam(lr=0.0005),
      loss='sparse_categorical_crossentropy',
      metrics=[keras.metrics.SparseCategoricalAccuracy()])

  model.summary(print_fn=absl.logging.info)
  return model


def _trainer_fn(trainer_fn_args, schema):
  """Build the estimator using the high level API.

  Args:
    trainer_fn_args: Holds args used to train the model as name/value pairs.
    schema: Holds the schema of the training examples.

  Returns:
    A dict of the following:
      - estimator: The estimator that will be used for training and eval.
      - train_spec: Spec for training.
      - eval_spec: Spec for eval.
      - eval_input_receiver_fn: Input function for eval.
  """

  train_batch_size = 20
  eval_batch_size = 10

  train_input_fn = lambda: _input_fn(  # pylint: disable=g-long-lambda
      trainer_fn_args.train_files,
      schema,
      batch_size=train_batch_size)

  eval_input_fn = lambda: _input_fn(  # pylint: disable=g-long-lambda
      trainer_fn_args.eval_files,
      schema,
      batch_size=eval_batch_size)

  train_spec = tf.estimator.TrainSpec(
      train_input_fn, max_steps=trainer_fn_args.train_steps)

  serving_receiver_fn = lambda: _serving_input_receiver_fn(schema)

  exporter = tf.estimator.FinalExporter('iris', serving_receiver_fn)
  eval_spec = tf.estimator.EvalSpec(
      eval_input_fn,
      steps=trainer_fn_args.eval_steps,
      exporters=[exporter],
      name='iris-eval')

  run_config = tf.estimator.RunConfig(
      save_checkpoints_steps=999, keep_checkpoint_max=1)

  export_dir = path_utils.serving_model_dir(trainer_fn_args.model_run_dir)
  run_config = run_config.replace(model_dir=export_dir)

  estimator = tf.keras.estimator.model_to_estimator(
      keras_model=_keras_model_builder(), config=run_config)

  # Create an input receiver for TFMA processing
  eval_receiver_fn = lambda: _eval_input_receiver_fn(schema)

  return {
      'estimator': estimator,
      'train_spec': train_spec,
      'eval_spec': eval_spec,
      'eval_input_receiver_fn': eval_receiver_fn
  }


# TFX generic trainer will call this function instead of train_fn.
def run_fn(fn_args: executor.TrainerFnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """
  schema = io_utils.parse_pbtxt_file(fn_args.schema_file, schema_pb2.Schema())

  training_spec = _trainer_fn(fn_args, schema)

  # Train the model
  absl.logging.info('Training model.')
  tf.estimator.train_and_evaluate(training_spec['estimator'],
                                  training_spec['train_spec'],
                                  training_spec['eval_spec'])
  absl.logging.info('Training complete.  Model written to %s',
                    fn_args.serving_model_dir)

  # Export an eval savedmodel for TFMA
  # NOTE: When trained in distributed training cluster, eval_savedmodel must be
  # exported only by the chief worker (check TF_CONFIG).
  absl.logging.info('Exporting eval_savedmodel for TFMA.')
  eval_export_dir = path_utils.eval_model_dir(fn_args.model_run_dir)
  tfma.export.export_eval_savedmodel(
      estimator=training_spec['estimator'],
      export_dir_base=eval_export_dir,
      eval_input_receiver_fn=training_spec['eval_input_receiver_fn'])

  absl.logging.info('Exported eval_savedmodel to %s.', fn_args.eval_model_dir)

  # TODO(b/160795287): Deprecate estimator based executor.
  # Copy serving and eval model from model_run to model artifact directory.
  serving_source = path_utils.serving_model_path(fn_args.model_run_dir)
  io_utils.copy_dir(serving_source, fn_args.serving_model_dir)
  absl.logging.info('Serving model copied to: %s.', fn_args.serving_model_dir)

  eval_source = path_utils.eval_model_path(fn_args.model_run_dir)
  io_utils.copy_dir(eval_source, fn_args.eval_model_dir)
  absl.logging.info('Eval model copied to: %s.', fn_args.eval_model_dir)
