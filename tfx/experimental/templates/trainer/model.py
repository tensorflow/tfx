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
"""Implementation of Trainer."""

from __future__ import division
from __future__ import print_function

import six

import tensorflow as tf
import tensorflow_model_analysis as tfma
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import schema_utils

from tfx.experimental.templates import common


def _build_estimator(config, hparams):
  """Build an estimator for the model."""

  # Developer TODO: Adjust this with model specification.

  tf_transform_output = tft.TFTransformOutput(hparams.transform_output)

  real_valued_columns = []
  for key in common.NUMERIC_FEATURES:
    tf.feature_column.numeric_column(key)

  categorical_columns = []
  for key in common.CATEGORICAL_FEATURES:
    categorical_columns.append(
        tf.feature_column.categorical_column_with_vocabulary_file(
            key,
            vocabulary_file=tf_transform_output.vocabulary_file_by_name(
                common.vocabulary_name(key))))

  return tf.estimator.DNNLinearCombinedEstimator(
      head=tf.contrib.estimator.binary_classification_head(),
      config=config,
      dnn_feature_columns=real_valued_columns,
      dnn_hidden_units=hparams.dnn_hidden_units,
      linear_feature_columns=categorical_columns,
  )


def _serving_receiver_fn(hparams, schema):
  """Build the serving input function for serving that receives examples."""
  raw_feature_spec = schema_utils.schema_as_feature_spec(schema).feature_spec
  raw_feature_spec.pop(common.LABEL)

  raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
      raw_feature_spec, default_batch_size=None)
  serving_input_receiver = raw_input_fn()

  tf_transform_output = tft.TFTransformOutput(hparams.transform_output)
  transformed_features = tf_transform_output.transform_raw_features(
      serving_input_receiver.features)

  return tf.estimator.export.ServingInputReceiver(
      transformed_features, serving_input_receiver.receiver_tensors)


def _eval_input_receiver_fn(hparams, schema):
  """Build everything needed for the Evaluator to run the model."""
  raw_feature_spec = schema_utils.schema_as_feature_spec(schema).feature_spec

  serialized_tf_example = tf.placeholder(
      dtype=tf.string, shape=[None], name='input_example_tensor')

  features = tf.parse_example(serialized_tf_example, raw_feature_spec)

  tf_transform_output = tft.TFTransformOutput(hparams.transform_output)
  transformed_features = tf_transform_output.transform_raw_features(features)
  features.update(transformed_features)

  return tfma.export.EvalInputReceiver(
      # Eval input functions consumes raw features
      features=features,
      # The key name MUST be 'examples'.
      receiver_tensors={'examples': serialized_tf_example},
      labels=transformed_features[common.transformed_name(common.LABEL)])


def _input_fn(filenames, hparams):
  """Generates features and labels for training or evaluation."""

  tf_transform_output = tft.TFTransformOutput(hparams.transform_output)

  transformed_feature_spec = (
      tf_transform_output.transformed_feature_spec().copy())

  dataset = tf.data.experimental.make_batched_features_dataset(
      filenames,
      hparams.batch_size,
      transformed_feature_spec,
      reader=tf.data.TFRecordDataset(filenames, compression_type='GZIP'))

  transformed_features = dataset.make_one_shot_iterator().get_next()

  return transformed_features, transformed_features.pop(
      common.transformed_name(common.LABEL))


# TFX Trainer Component will call this function.
def trainer_fn(hparams, schema):
  """Build the estimator for the Trainer."""

  for k, v in six.iteritems(common.HPARAMS.values()):
    hparams.add_hparam(k, v)

  train_input_fn = lambda: _input_fn(hparams.train_files, hparams)
  eval_input_fn = lambda: _input_fn(hparams.eval_files, hparams)

  train_spec = tf.estimator.TrainSpec(
      train_input_fn, max_steps=hparams.train_steps)

  serving_receiver_fn = lambda: _serving_receiver_fn(hparams, schema)

  eval_spec = tf.estimator.EvalSpec(
      eval_input_fn,
      steps=hparams.eval_steps,
      exporters=[
          tf.estimator.FinalExporter('trainer', serving_receiver_fn),
      ],
      name='trainer-eval')

  run_config = tf.estimator.RunConfig(
      save_checkpoints_steps=999, keep_checkpoint_max=1)
  run_config = run_config.replace(model_dir=hparams.serving_model_dir)

  estimator = _build_estimator(hparams=hparams, config=run_config)

  # Create an input receiver for TFMA processing
  eval_input_receiver_fn = lambda: _eval_input_receiver_fn(hparams, schema)

  return {
      'estimator': estimator,
      'train_spec': train_spec,
      'eval_spec': eval_spec,
      'eval_input_receiver_fn': eval_input_receiver_fn
  }
