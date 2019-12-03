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
"""Python source file include CIFAR-10 pipeline functions and necessary utils.

For a TFX pipeline to successfully run, a preprocessing_fn and a train_fn
function needs to be provided. This file contains both.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import absl
import tensorflow as tf
import tensorflow_model_analysis as tfma
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import schema_utils

# Keys
_LABEL_KEY = 'label'
_IMAGE_KEY = 'image_raw'


def _transformed_name(key):
  return key + '_xf_input'


def _get_raw_feature_spec(schema):
  return schema_utils.schema_as_feature_spec(schema).feature_spec


def _gzip_reader_fn(filenames):
  """Small utility returning a record reader that can read gzip'ed files."""
  return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def _image_parser(image_str):
  image = tf.image.decode_image(image_str, channels=3)
  image = tf.reshape(image, (32, 32, 3))
  image = tf.cast(image, tf.float32) / 255.
  return image


def _label_parser(label_id):
  label = tf.one_hot(label_id, 10)
  return label


def preprocessing_fn(inputs):
  """tf.transform's callback function for preprocessing inputs.

  Args:
    inputs: map from feature keys to raw not-yet-transformed features.

  Returns:
    Map from string feature key to transformed feature operations.
  """
  outputs = {
      _transformed_name(_IMAGE_KEY):
          tf.compat.v2.map_fn(
              _image_parser,
              tf.squeeze(inputs[_IMAGE_KEY], axis=1),
              dtype=tf.float32),
      _transformed_name(_LABEL_KEY):
          tf.compat.v2.map_fn(
              _label_parser,
              tf.squeeze(inputs[_LABEL_KEY], axis=1),
              dtype=tf.float32)
  }
  return outputs


def _keras_model_builder():
  """Build a keras model for image classification on cifar10 dataset."""
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(
          64,
          kernel_size=3,
          activation='relu',
          input_shape=(32, 32, 3),
          name='image_raw_xf'),
      tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu'),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(10, activation='softmax')
  ])

  model.compile(
      optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9),
      loss='categorical_crossentropy',
      metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')])

  absl.logging.info(model.summary())
  return model


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
  transformed_features.pop(_transformed_name(_LABEL_KEY))

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

  transformed_features = tf_transform_output.transform_raw_features(features)
  labels = transformed_features.pop(_transformed_name(_LABEL_KEY))

  receiver_tensors = {'examples': serialized_tf_example}

  return tfma.export.EvalInputReceiver(
      features=transformed_features,
      receiver_tensors=receiver_tensors,
      labels=labels)


def _input_fn(filenames, tf_transform_output, batch_size):
  """Generates features and labels for training or evaluation.

  Args:
    filenames: [str] list of CSV files to read data from.
    tf_transform_output: A TFTransformOutput.
    batch_size: int First dimension size of the Tensors returned by input_fn

  Returns:
    A (features, indices) tuple where features is a dictionary of
      Tensors, and indices is a single Tensor of label indices.
  """
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


# TFX will call this function
def trainer_fn(hparams, schema):
  """Build the estimator using the high level API.

  Args:
    hparams: Holds hyperparameters used to train the model as name/value pairs.
    schema: Holds the schema of the training examples.

  Returns:
    A dict of the following:
      - estimator: The estimator that will be used for training and eval.
      - train_spec: Spec for training.
      - eval_spec: Spec for eval.
      - eval_input_receiver_fn: Input function for eval.
  """
  train_batch_size = 32
  eval_batch_size = 32

  tf_transform_output = tft.TFTransformOutput(hparams.transform_output)

  train_input_fn = lambda: _input_fn(  # pylint: disable=g-long-lambda
      hparams.train_files,
      tf_transform_output,
      batch_size=train_batch_size)

  eval_input_fn = lambda: _input_fn(  # pylint: disable=g-long-lambda
      hparams.eval_files,
      tf_transform_output,
      batch_size=eval_batch_size)

  train_spec = tf.estimator.TrainSpec(  # pylint: disable=g-long-lambda
      train_input_fn,
      max_steps=hparams.train_steps)

  serving_receiver_fn = lambda: _example_serving_receiver_fn(  # pylint: disable=g-long-lambda
      tf_transform_output, schema)

  exporter = tf.estimator.FinalExporter('cifar-10', serving_receiver_fn)
  eval_spec = tf.estimator.EvalSpec(
      eval_input_fn,
      steps=hparams.eval_steps,
      exporters=[exporter],
      name='cifar-10')

  run_config = tf.estimator.RunConfig(
      save_checkpoints_steps=999, keep_checkpoint_max=1)

  run_config = run_config.replace(model_dir=hparams.serving_model_dir)

  estimator = tf.keras.estimator.model_to_estimator(
      keras_model=_keras_model_builder(), config=run_config)

  # Create an input receiver for TFMA processing
  receiver_fn = lambda: _eval_input_receiver_fn(  # pylint: disable=g-long-lambda
      tf_transform_output, schema)

  return {
      'estimator': estimator,
      'train_spec': train_spec,
      'eval_spec': eval_spec,
      'eval_input_receiver_fn': receiver_fn
  }
