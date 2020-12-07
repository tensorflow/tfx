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
"""Python source file include taxi pipeline functions and necesasry utils.

For a TFX pipeline to successfully run, a preprocessing_fn and a
_build_estimator function needs to be provided.  This file contains both.

This file is equivalent to examples/chicago_taxi/trainer/model.py and
examples/chicago_taxi/preprocess.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import absl
import tensorflow as tf
import tensorflow_model_analysis as tfma
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import schema_utils
from tfx.components.trainer import executor
from tfx.utils import io_utils
from tfx.utils import path_utils
from tensorflow_metadata.proto.v0 import schema_pb2

# TODO(b/162532757): clean up once tfx-bsl post-0.22 is released.
try:
  from tfx_bsl.public.tfxio import TensorFlowDatasetOptions  # pylint:disable=g-import-not-at-top
except ImportError:
  TensorFlowDatasetOptions = None

# Categorical features are assumed to each have a maximum value in the dataset.
_MAX_CATEGORICAL_FEATURE_VALUES = [24, 31, 12]

_CATEGORICAL_FEATURE_KEYS = [
    'trip_start_hour', 'trip_start_day', 'trip_start_month',
    'pickup_census_tract', 'dropoff_census_tract', 'pickup_community_area',
    'dropoff_community_area'
]

_DENSE_FLOAT_FEATURE_KEYS = ['trip_miles', 'fare', 'trip_seconds']

# Number of buckets used by tf.transform for encoding each feature.
_FEATURE_BUCKET_COUNT = 10

_BUCKET_FEATURE_KEYS = [
    'pickup_latitude', 'pickup_longitude', 'dropoff_latitude',
    'dropoff_longitude'
]

# Number of vocabulary terms used for encoding VOCAB_FEATURES by tf.transform
_VOCAB_SIZE = 1000

# Count of out-of-vocab buckets in which unrecognized VOCAB_FEATURES are hashed.
_OOV_SIZE = 10

_VOCAB_FEATURE_KEYS = [
    'payment_type',
    'company',
]

# Keys
_LABEL_KEY = 'tips'
_FARE_KEY = 'fare'


def _transformed_name(key):
  return key + '_xf'


def _transformed_names(keys):
  return [_transformed_name(key) for key in keys]


# Tf.Transform considers these features as "raw"
def _get_raw_feature_spec(schema):
  return schema_utils.schema_as_feature_spec(schema).feature_spec


def _gzip_reader_fn(filenames):
  """Small utility returning a record reader that can read gzip'ed files."""
  return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def _build_estimator(config, hidden_units=None, warm_start_from=None):
  """Build an estimator for predicting the tipping behavior of taxi riders.

  Args:
    config: tf.estimator.RunConfig defining the runtime environment for the
      estimator (including model_dir).
    hidden_units: [int], the layer sizes of the DNN (input layer first)
    warm_start_from: Optional directory to warm start from.

  Returns:
    A dict of the following:
      - estimator: The estimator that will be used for training and eval.
      - train_spec: Spec for training.
      - eval_spec: Spec for eval.
      - eval_input_receiver_fn: Input function for eval.
  """
  real_valued_columns = [
      tf.feature_column.numeric_column(key, shape=())
      for key in _transformed_names(_DENSE_FLOAT_FEATURE_KEYS)
  ]
  categorical_columns = [
      tf.feature_column.categorical_column_with_identity(
          key, num_buckets=_VOCAB_SIZE + _OOV_SIZE, default_value=0)
      for key in _transformed_names(_VOCAB_FEATURE_KEYS)
  ]
  categorical_columns += [
      tf.feature_column.categorical_column_with_identity(
          key, num_buckets=_FEATURE_BUCKET_COUNT, default_value=0)
      for key in _transformed_names(_BUCKET_FEATURE_KEYS)
  ]
  categorical_columns += [
      tf.feature_column.categorical_column_with_identity(  # pylint: disable=g-complex-comprehension
          key,
          num_buckets=num_buckets,
          default_value=0) for key, num_buckets in zip(
              _transformed_names(_CATEGORICAL_FEATURE_KEYS),
              _MAX_CATEGORICAL_FEATURE_VALUES)
  ]
  return tf.estimator.DNNLinearCombinedClassifier(
      config=config,
      linear_feature_columns=categorical_columns,
      dnn_feature_columns=real_valued_columns,
      dnn_hidden_units=hidden_units or [100, 70, 50, 25],
      warm_start_from=warm_start_from)


def _example_serving_receiver_fn(tf_transform_output, schema):
  """Build the serving in inputs.

  Args:
    tf_transform_output: A TFTransformOutput.
    schema: the schema of the input data.

  Returns:
    Tensorflow graph which parses examples, applying tf-transform to them.
  """
  raw_feature_spec = _get_raw_feature_spec(schema)
  raw_feature_spec.pop(_LABEL_KEY)

  raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
      raw_feature_spec, default_batch_size=None)
  serving_input_receiver = raw_input_fn()

  transformed_features = tf_transform_output.transform_raw_features(
      serving_input_receiver.features)

  return tf.estimator.export.ServingInputReceiver(
      transformed_features, serving_input_receiver.receiver_tensors)


def _eval_input_receiver_fn(tf_transform_output, schema):
  """Build everything needed for the tf-model-analysis to run the model.

  Args:
    tf_transform_output: A TFTransformOutput.
    schema: the schema of the input data.

  Returns:
    EvalInputReceiver function, which contains:
      - Tensorflow graph which parses raw untransformed features, applies the
        tf-transform preprocessing operators.
      - Set of raw, untransformed features.
      - Label against which predictions will be compared.
  """
  # Notice that the inputs are raw features, not transformed features here.
  raw_feature_spec = _get_raw_feature_spec(schema)

  serialized_tf_example = tf.compat.v1.placeholder(
      dtype=tf.string, shape=[None], name='input_example_tensor')

  # Add a parse_example operator to the tensorflow graph, which will parse
  # raw, untransformed, tf examples.
  features = tf.io.parse_example(
      serialized=serialized_tf_example, features=raw_feature_spec)

  # Now that we have our raw examples, process them through the tf-transform
  # function computed during the preprocessing step.
  transformed_features = tf_transform_output.transform_raw_features(
      features)

  # The key name MUST be 'examples'.
  receiver_tensors = {'examples': serialized_tf_example}

  # NOTE: Model is driven by transformed features (since training works on the
  # materialized output of TFT, but slicing will happen on raw features.
  features.update(transformed_features)

  return tfma.export.EvalInputReceiver(
      features=features,
      receiver_tensors=receiver_tensors,
      labels=transformed_features[_transformed_name(_LABEL_KEY)])


def _input_fn(
    filenames, data_accessor, tf_transform_output, batch_size=200):
  """Generates features and labels for training or evaluation.

  Args:
    filenames: [str] list of CSV files to read data from.
    data_accessor: Unused. Only to keep the signature the same as
      _tfxio_input_fn()
    tf_transform_output: A TFTransformOutput.
    batch_size: int First dimension size of the Tensors returned by input_fn

  Returns:
    A (features, indices) tuple where features is a dictionary of
      Tensors, and indices is a single Tensor of label indices.
  """
  del data_accessor
  transformed_feature_spec = (
      tf_transform_output.transformed_feature_spec().copy())

  dataset = tf.data.experimental.make_batched_features_dataset(
      filenames, batch_size, transformed_feature_spec, reader=_gzip_reader_fn)

  transformed_features = tf.compat.v1.data.make_one_shot_iterator(
      dataset).get_next()
  # We pop the label because we do not want to use it as a feature while we're
  # training.
  return transformed_features, transformed_features.pop(
      _transformed_name(_LABEL_KEY))


def _tfxio_input_fn(
    filenames, data_accessor, tf_transform_output, batch_size=200):
  """Generates features and labels for training or evaluation.

  Args:
    filenames: [str] list of CSV files to read data from.
    data_accessor: fn_args_utils.DataAccessor.
    tf_transform_output: A TFTransformOutput.
    batch_size: int First dimension size of the Tensors returned by input_fn

  Returns:
    A (features, indices) tuple where features is a dictionary of
      Tensors, and indices is a single Tensor of label indices.
  """
  dataset = data_accessor.tf_dataset_factory(
      filenames,
      TensorFlowDatasetOptions(
          batch_size=batch_size,
          label_key=_transformed_name(_LABEL_KEY)),
      tf_transform_output.transformed_metadata.schema)

  return tf.compat.v1.data.make_one_shot_iterator(
      dataset).get_next()


# TFX will call this function
def trainer_fn(trainer_fn_args, schema):
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
  if trainer_fn_args.hyperparameters:
    hp = trainer_fn_args.hyperparameters
    first_dnn_layer_size = hp.get('first_dnn_layer_size')
    num_dnn_layers = hp.get('num_dnn_layers')
    dnn_decay_factor = hp.get('dnn_decay_factor')
  else:
    # Number of nodes in the first layer of the DNN
    first_dnn_layer_size = 100
    num_dnn_layers = 4
    dnn_decay_factor = 0.7

  train_batch_size = 40
  eval_batch_size = 40

  # TODO(b/162532757): use _tfxio_input_fn exclusively once tfx-bsl post-0.22 is
  # released.
  use_tfxio_input_fn = trainer_fn_args.get('use_tfxio_input_fn', False)
  input_fn = _tfxio_input_fn if use_tfxio_input_fn else _input_fn
  tf_transform_output = tft.TFTransformOutput(trainer_fn_args.transform_output)

  train_input_fn = lambda: input_fn(  # pylint: disable=g-long-lambda
      trainer_fn_args.train_files,
      trainer_fn_args.data_accessor,
      tf_transform_output,
      batch_size=train_batch_size)

  eval_input_fn = lambda: input_fn(  # pylint: disable=g-long-lambda
      trainer_fn_args.eval_files,
      trainer_fn_args.data_accessor,
      tf_transform_output,
      batch_size=eval_batch_size)

  train_spec = tf.estimator.TrainSpec(  # pylint: disable=g-long-lambda
      train_input_fn,
      max_steps=trainer_fn_args.train_steps)

  serving_receiver_fn = lambda: _example_serving_receiver_fn(  # pylint: disable=g-long-lambda
      tf_transform_output, schema)

  exporter = tf.estimator.FinalExporter('chicago-taxi', serving_receiver_fn)
  eval_spec = tf.estimator.EvalSpec(
      eval_input_fn,
      steps=trainer_fn_args.eval_steps,
      exporters=[exporter],
      name='chicago-taxi-eval')

  run_config = tf.estimator.RunConfig(
      save_checkpoints_steps=999,
      # keep_checkpoint_max must be more than the number of worker replicas
      # nodes if training distributed, in order to avoid race condition.
      keep_checkpoint_max=5)

  export_dir = path_utils.serving_model_dir(trainer_fn_args.model_run_dir)
  run_config = run_config.replace(model_dir=export_dir)
  warm_start_from = trainer_fn_args.base_model

  estimator = _build_estimator(
      # Construct layers sizes with exponetial decay
      hidden_units=[
          max(2, int(first_dnn_layer_size * dnn_decay_factor**i))
          for i in range(num_dnn_layers)
      ],
      config=run_config,
      warm_start_from=warm_start_from)

  # Create an input receiver for TFMA processing
  receiver_fn = lambda: _eval_input_receiver_fn(  # pylint: disable=g-long-lambda
      tf_transform_output, schema)

  return {
      'estimator': estimator,
      'train_spec': train_spec,
      'eval_spec': eval_spec,
      'eval_input_receiver_fn': receiver_fn
  }


# TFX generic trainer will call this function
def run_fn(fn_args: executor.TrainerFnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """
  schema = io_utils.parse_pbtxt_file(fn_args.schema_file, schema_pb2.Schema())

  training_spec = trainer_fn(fn_args, schema)

  # Train the model
  absl.logging.info('Training model.')
  tf.estimator.train_and_evaluate(training_spec['estimator'],
                                  training_spec['train_spec'],
                                  training_spec['eval_spec'])

  # Export an eval savedmodel for TFMA
  # NOTE: When trained in distributed training cluster, eval_savedmodel must be
  # exported only by the chief worker.
  absl.logging.info('Exporting eval_savedmodel for TFMA.')
  tfma.export.export_eval_savedmodel(
      estimator=training_spec['estimator'],
      export_dir_base=path_utils.eval_model_dir(fn_args.model_run_dir),
      eval_input_receiver_fn=training_spec['eval_input_receiver_fn'])

  # TODO(b/160795287): Deprecate estimator based executor.
  # Copy serving and eval model from model_run to model artifact directory.
  serving_source = path_utils.serving_model_path(fn_args.model_run_dir)
  io_utils.copy_dir(serving_source, fn_args.serving_model_dir)

  eval_source = path_utils.eval_model_path(fn_args.model_run_dir)
  io_utils.copy_dir(eval_source, fn_args.eval_model_dir)

  absl.logging.info('Training complete. Model written to %s',
                    fn_args.serving_model_dir)
  absl.logging.info('Exported eval_savedmodel to %s.', fn_args.eval_model_dir)
