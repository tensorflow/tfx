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
# pylint: disable=line-too-long
# pylint: disable=unused-argument
# pylint: disable=unused-import
"""Python source file include taxi pipeline functions and necesasry utils.

The utilities in this file are used to build a model with native Keras.
This module file will be used in Transform and generic Trainer.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Text

import absl
import tensorflow as tf
# import tensorflow_transform as tft # Step 4

from tfx.components.trainer.executor import TrainerFnArgs

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

# START Step 4 -----------------------------------------------------------
# def _transformed_name(key):
#   return key + '_xf'
#
#
# def _transformed_names(keys):
#   return [_transformed_name(key) for key in keys]
#
#
# def _gzip_reader_fn(filenames):
#   """Small utility returning a record reader that can read gzip'ed files."""
#   return tf.data.TFRecordDataset(
#       filenames,
#       compression_type='GZIP')
#
#
# def _fill_in_missing(x):
#   """Replace missing values in a SparseTensor.
#
#   Fills in missing values of `x` with '' or 0, and converts to a dense tensor.
#
#   Args:
#     x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
#       in the second dimension.
#
#   Returns:
#     A rank 1 tensor where missing values of `x` have been filled in.
#   """
#   default_value = '' if x.dtype == tf.string else 0
#   return tf.squeeze(
#       tf.sparse.to_dense(
#           tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
#           default_value),
#       axis=1)
#
#
#   # TFX Transform will call this function.
# def preprocessing_fn(inputs):
#   """tf.transform's callback function for preprocessing inputs.
#
#   Args:
#     inputs: map from feature keys to raw not-yet-transformed features.
#
#   Returns:
#     Map from string feature key to transformed feature operations.
#   """
#   outputs = {}
#   for key in _DENSE_FLOAT_FEATURE_KEYS:
#     # Preserve this feature as a dense float, setting nan's to the mean.
#     outputs[_transformed_name(key)] = tft.scale_to_z_score(
#         _fill_in_missing(inputs[key]))
#
#   for key in _VOCAB_FEATURE_KEYS:
#     # Build a vocabulary for this feature.
#     outputs[_transformed_name(key)] = tft.compute_and_apply_vocabulary(
#         _fill_in_missing(inputs[key]),
#         top_k=_VOCAB_SIZE,
#         num_oov_buckets=_OOV_SIZE)
#
#   for key in _BUCKET_FEATURE_KEYS:
#     outputs[_transformed_name(key)] = tft.bucketize(
#         _fill_in_missing(inputs[key]),
#         _FEATURE_BUCKET_COUNT,
#         always_return_num_quantiles=False)
#
#   for key in _CATEGORICAL_FEATURE_KEYS:
#     outputs[_transformed_name(key)] = _fill_in_missing(inputs[key])
#
#   # Was this passenger a big tipper?
#   taxi_fare = _fill_in_missing(inputs[_FARE_KEY])
#   tips = _fill_in_missing(inputs[_LABEL_KEY])
#   outputs[_transformed_name(_LABEL_KEY)] = tf.where(
#       tf.math.is_nan(taxi_fare),
#       tf.cast(tf.zeros_like(taxi_fare), tf.int64),
#       # Test if the tip was > 20% of the fare.
#       tf.cast(
#           tf.greater(tips, tf.multiply(taxi_fare, tf.constant(0.2))),
#                      tf.int64))
#
#   return outputs
# END Step 4 -------------------------------------------------------------

# START Step 5 -----------------------------------------------------------
# def _get_serve_tf_examples_fn(model, tf_transform_output):
#   """Returns a function that parses a serialized tf.Example and applies TFT"""
#
#   model.tft_layer = tf_transform_output.transform_features_layer()
#
#   @tf.function
#   def serve_tf_examples_fn(serialized_tf_examples):
#     """Returns the output to be used in the serving signature."""
#     feature_spec = tf_transform_output.raw_feature_spec()
#     feature_spec.pop(_LABEL_KEY)
#     parsed_features = tf.io.parse_example(serialized_tf_examples,
#                                           feature_spec)
#
#     transformed_features = model.tft_layer(parsed_features)
#
#     return model(transformed_features)
#
#   return serve_tf_examples_fn
#
#
# def _input_fn(file_pattern: List[Text],
#               tf_transform_output: tft.TFTransformOutput,
#               batch_size: int = 200) -> tf.data.Dataset:
#   """Generates features and label for tuning/training.
#
#   Args:
#     file_pattern: List of paths or patterns of input tfrecord files.
#     tf_transform_output: A TFTransformOutput.
#     batch_size: representing the number of consecutive elements of returned
#       dataset to combine in a single batch
#
#   Returns:
#     A dataset that contains (features, indices) tuple where features is a
#       dictionary of Tensors, and indices is a single Tensor of label indices.
#   """
#   transformed_feature_spec = (
#       tf_transform_output.transformed_feature_spec().copy())
#
#   dataset = tf.data.experimental.make_batched_features_dataset(
#       file_pattern=file_pattern,
#       batch_size=batch_size,
#       features=transformed_feature_spec,
#       reader=_gzip_reader_fn,
#       label_key=_transformed_name(_LABEL_KEY))
#
#   return dataset
#
#
# def _build_keras_model(hidden_units: List[int] = None) -> tf.keras.Model:
#   """Creates a DNN Keras model for classifying taxi data.
#
#   Args:
#     hidden_units: [int], the layer sizes of the DNN (input layer first).
#
#   Returns:
#     A keras Model.
#   """
#   real_valued_columns = [
#       tf.feature_column.numeric_column(key, shape=())
#       for key in _transformed_names(_DENSE_FLOAT_FEATURE_KEYS)
#   ]
#   categorical_columns = [
#       tf.feature_column.categorical_column_with_identity(
#           key, num_buckets=_VOCAB_SIZE + _OOV_SIZE, default_value=0)
#       for key in _transformed_names(_VOCAB_FEATURE_KEYS)
#   ]
#   categorical_columns += [
#       tf.feature_column.categorical_column_with_identity(
#           key, num_buckets=_FEATURE_BUCKET_COUNT, default_value=0)
#       for key in _transformed_names(_BUCKET_FEATURE_KEYS)
#   ]
#   categorical_columns += [
#       tf.feature_column.categorical_column_with_identity(  # pylint: disable=g-complex-comprehension
#           key,
#           num_buckets=num_buckets,
#           default_value=0) for key, num_buckets in zip(
#               _transformed_names(_CATEGORICAL_FEATURE_KEYS),
#               _MAX_CATEGORICAL_FEATURE_VALUES)
#   ]
#   indicator_column = [
#       tf.feature_column.indicator_column(categorical_column)
#       for categorical_column in categorical_columns
#   ]
#
#   model = _wide_and_deep_classifier(
#       # TODO(b/139668410) replace with premade wide_and_deep keras model
#       wide_columns=indicator_column,
#       deep_columns=real_valued_columns,
#       dnn_hidden_units=hidden_units or [100, 70, 50, 25])
#   return model
#
#
# def _wide_and_deep_classifier(wide_columns, deep_columns, dnn_hidden_units):
#   """Build a simple keras wide and deep model.
#
#   Args:
#     wide_columns: Feature columns wrapped in indicator_column for wide
#       (linear) part of the model.
#     deep_columns: Feature columns for deep part of the model.
#     dnn_hidden_units: [int], the layer sizes of the hidden DNN.
#
#   Returns:
#     A Wide and Deep Keras model
#   """
#   # Following values are hard coded for simplicity in this example,
#   # However prefarably they should be passsed in as hparams.
#
#   # Keras needs the feature definitions at compile time.
#   # TODO(b/139081439): Automate generation of input layers from FeatureColumn.
#   input_layers = {
#       colname: tf.keras.layers.Input(name=colname, shape=(), dtype=tf.float32)
#       for colname in _transformed_names(_DENSE_FLOAT_FEATURE_KEYS)
#   }
#   input_layers.update({
#       colname: tf.keras.layers.Input(name=colname, shape=(), dtype='int32')
#       for colname in _transformed_names(_VOCAB_FEATURE_KEYS)
#   })
#   input_layers.update({
#       colname: tf.keras.layers.Input(name=colname, shape=(), dtype='int32')
#       for colname in _transformed_names(_BUCKET_FEATURE_KEYS)
#   })
#   input_layers.update({
#       colname: tf.keras.layers.Input(name=colname, shape=(), dtype='int32')
#       for colname in _transformed_names(_CATEGORICAL_FEATURE_KEYS)
#   })
#
#   # TODO(b/144500510): SparseFeatures for feature columns + Keras.
#   deep = tf.keras.layers.DenseFeatures(deep_columns)(input_layers)
#   for numnodes in dnn_hidden_units:
#     deep = tf.keras.layers.Dense(numnodes)(deep)
#   wide = tf.keras.layers.DenseFeatures(wide_columns)(input_layers)
#
#   output = tf.keras.layers.Dense(
#       1, activation='sigmoid')(
#           tf.keras.layers.concatenate([deep, wide]))
#
#   model = tf.keras.Model(input_layers, output)
#   model.compile(
#       loss='binary_crossentropy',
#       optimizer=tf.keras.optimizers.Adam(lr=0.001),
#       metrics=[tf.keras.metrics.BinaryAccuracy()])
#   model.summary(print_fn=absl.logging.info)
#   return model
#
#
# # TFX Trainer will call this function.
# def run_fn(fn_args: TrainerFnArgs):
#   """Train the model based on given args.
#
#   Args:
#     fn_args: Holds args used to train the model as name/value pairs.
#   """
#   # Number of nodes in the first layer of the DNN
#   first_dnn_layer_size = 100
#   num_dnn_layers = 4
#   dnn_decay_factor = 0.7
#
#   tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
#
#   train_dataset = _input_fn(fn_args.train_files, tf_transform_output, 40)
#   eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output, 40)
#
#   # If no GPUs are found, CPU is used.
#   mirrored_strategy = tf.distribute.MirroredStrategy()
#   with mirrored_strategy.scope():
#     model = _build_keras_model(
#         # Construct layers sizes with exponetial decay
#         hidden_units=[
#             max(2, int(first_dnn_layer_size * dnn_decay_factor**i))
#             for i in range(num_dnn_layers)
#         ])
#
#   model.fit(
#       train_dataset,
#       steps_per_epoch=fn_args.train_steps,
#       validation_data=eval_dataset,
#       validation_steps=fn_args.eval_steps)
#
#   signatures = {
#       'serving_default':
#           _get_serve_tf_examples_fn(model,
#                                     tf_transform_output).get_concrete_function(
#                                         tf.TensorSpec(
#                                             shape=[None],
#                                             dtype=tf.string,
#                                             name='examples')),
#   }
#   model.save(fn_args.serving_model_dir, save_format='tf',
#              signatures=signatures)
# END Step 5 -------------------------------------------------------------
