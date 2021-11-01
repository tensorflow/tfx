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
"""Python source which includes pipeline functions for Cloud Tuner example.

The utilities in this file are used to build a model with native Keras.
This module file will be used in the Transform, Tuner and generic Trainer
components.
CloudTuner (a subclass of keras_tuner.Tuner) creates a seamless integration with
Cloud AI Platform Vizier as a backend to get suggestions of hyperparameters
and run trials. DistributingCloudTuner is a subclass of CloudTuner which
launches remote distributed training job for each trial on Cloud AI Platform
Training Service. More details see:
https://github.com/tensorflow/cloud/blob/master/src/python/tensorflow_cloud/tuner/tuner.py
"""

import datetime
import os
from typing import List

from absl import logging
import keras_tuner
import tensorflow as tf
from tensorflow import keras
import tensorflow_transform as tft
import tfx.v1 as tfx
from tfx_bsl.public import tfxio

from tensorflow_cloud.core import machine_config
from tensorflow_cloud.tuner import tuner as cloud_tuner


_FEATURE_KEYS = [
    'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g'
]
_LABEL_KEY = 'species'

_TRAIN_BATCH_SIZE = 20
_EVAL_BATCH_SIZE = 10

# Base image to use for AI Platform Training. This image must follow
# cloud_fit image with a cloud_fit.remote() as entry point. Refer to
# cloud_fit documentation for more details at
# https://github.com/tensorflow/cloud/blob/master/src/python/tensorflow_cloud/tuner/cloud_fit_readme.md
#  TODO(b/184093307) Push official cloud_fit images for future release of
# tensorflow-cloud.
_CLOUD_FIT_IMAGE = 'gcr.io/my-project-id/cloud_fit'


def _transformed_name(key):
  return key + '_xf'


def _get_tf_examples_serving_signature(model, tf_transform_output):
  """Returns a serving signature that accepts `tensorflow.Example`."""

  # We need to track the layers in the model in order to save it.
  # TODO(b/162357359): Revise once the bug is resolved.
  model.tft_layer_inference = tf_transform_output.transform_features_layer()

  @tf.function(input_signature=[
      tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
  ])
  def serve_tf_examples_fn(serialized_tf_example):
    """Returns the output to be used in the serving signature."""
    raw_feature_spec = tf_transform_output.raw_feature_spec()
    # Remove label feature since these will not be present at serving time.
    raw_feature_spec.pop(_LABEL_KEY)
    raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
    transformed_features = model.tft_layer_inference(raw_features)
    logging.info('serve_transformed_features = %s', transformed_features)

    outputs = model(transformed_features)
    # TODO(b/154085620): Convert the predicted labels from the model using a
    # reverse-lookup (opposite of transform.py).
    return {'outputs': outputs}

  return serve_tf_examples_fn


def _get_transform_features_signature(model, tf_transform_output):
  """Returns a serving signature that applies tf.Transform to features."""

  # We need to track the layers in the model in order to save it.
  # TODO(b/162357359): Revise once the bug is resolved.
  model.tft_layer_eval = tf_transform_output.transform_features_layer()

  @tf.function(input_signature=[
      tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
  ])
  def transform_features_fn(serialized_tf_example):
    """Returns the transformed_features to be fed as input to evaluator."""
    raw_feature_spec = tf_transform_output.raw_feature_spec()
    raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
    transformed_features = model.tft_layer_eval(raw_features)
    logging.info('eval_transformed_features = %s', transformed_features)
    return transformed_features

  return transform_features_fn


def _input_fn(file_pattern: List[str],
              data_accessor: tfx.components.DataAccessor,
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 200) -> tf.data.Dataset:
  """Generates features and label for tuning/training.

  Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    data_accessor: DataAccessor for converting input to RecordBatch.
    tf_transform_output: A `TFTransformOutput` object, containing statistics
      and metadata from TFTransform component.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
  return data_accessor.tf_dataset_factory(
      file_pattern,
      tfxio.TensorFlowDatasetOptions(
          batch_size=batch_size, label_key=_transformed_name(_LABEL_KEY)),
      tf_transform_output.transformed_metadata.schema).repeat()


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
    # Nothing to transform for the penguin dataset. This code is just to
    # show how the preprocessing function for Transform should be defined.
    # We just assign original values to the transformed feature.
    outputs[_transformed_name(key)] = inputs[key]
  # TODO(b/157064428): Support label transformation for Keras.
  # Do not apply label transformation as it will result in wrong evaluation.
  outputs[_transformed_name(_LABEL_KEY)] = inputs[_LABEL_KEY]

  return outputs


def _get_hyperparameters() -> keras_tuner.HyperParameters:
  """Returns hyperparameters for building Keras model."""
  hp = keras_tuner.HyperParameters()
  # Defines search space.
  hp.Choice('learning_rate', [1e-5, 1e-4, 1e-3, 1e-2], default=1e-2)
  hp.Int('num_layers', 1, 4, default=2)
  return hp


def _build_keras_model(hparams: keras_tuner.HyperParameters) -> tf.keras.Model:
  """Creates a DNN Keras model for classifying penguin data.

  Args:
    hparams: Holds HyperParameters for tuning.

  Returns:
    A Keras Model.
  """
  # The model below is built with Functional API, please refer to
  # https://www.tensorflow.org/guide/keras/overview for all API options.
  inputs = [
      keras.layers.Input(shape=(1,), name=_transformed_name(f))
      for f in _FEATURE_KEYS
  ]
  d = keras.layers.concatenate(inputs)
  for _ in range(int(hparams.get('num_layers'))):
    d = keras.layers.Dense(8, activation='relu')(d)
  outputs = keras.layers.Dense(3, activation='softmax')(d)

  model = keras.Model(inputs=inputs, outputs=outputs)
  model.compile(
      optimizer=keras.optimizers.Adam(hparams.get('learning_rate')),
      loss='sparse_categorical_crossentropy',
      metrics=[keras.metrics.SparseCategoricalAccuracy()])

  model.summary(print_fn=logging.info)
  return model


# TFX Tuner will call this function.
def tuner_fn(fn_args: tfx.components.FnArgs) -> tfx.components.TunerFnResult:
  """Build the tuner using the CloudTuner API.

  Args:
    fn_args: Holds args as name/value pairs. See
      https://www.tensorflow.org/tfx/api_docs/python/tfx/components/trainer/fn_args_utils/FnArgs.
      - transform_graph_path: optional transform graph produced by TFT.
      - custom_config: An optional dictionary passed to the component. In this
        example, it contains the dict ai_platform_tuning_args.
      - working_dir: working dir for tuning.
      - train_files: List of file paths containing training tf.Example data.
      - eval_files: List of file paths containing eval tf.Example data.
      - train_steps: number of train steps.
      - eval_steps: number of eval steps.

  Returns:
    A namedtuple contains the following:
      - tuner: A BaseTuner that will be used for tuning.
      - fit_kwargs: Args to pass to tuner's run_trial function for fitting the
                    model , e.g., the training and validation dataset. Required
                    args depend on the above tuner's implementation. For
                    DistributingCloudTuner, we generate datasets at the remote
                    jobs rather than serialize and then deserialize them.
  """

  # study_id should be the same across multiple tuner workers which starts
  # approximately at the same time.
  study_id = 'DistributingCloudTuner_study_{}'.format(
            datetime.datetime.now().strftime('%Y%m%d%H'))

  if _CLOUD_FIT_IMAGE == 'gcr.io/my-project-id/cloud_fit':
    raise ValueError('Build your own cloud_fit image, ' +
                     'default dummy one is used!')

  tuner = cloud_tuner.DistributingCloudTuner(
      _build_keras_model,
      # The project/region configuations for Cloud Vizier service and its trial
      # executions. Note: this example uses the same configuration as the
      # CAIP Training service for distributed tuning flock management to view
      # all of the pipeline's jobs and resources in the same project. It can
      # also be configured separately.
      project_id=fn_args.custom_config['ai_platform_tuning_args']['project'],
      region=fn_args.custom_config['ai_platform_tuning_args']['region'],
      objective=keras_tuner.Objective('val_sparse_categorical_accuracy', 'max'),
      hyperparameters=_get_hyperparameters(),
      max_trials=5,  # Optional.
      directory=os.path.join(fn_args.custom_config['remote_trials_working_dir'],
                             study_id),
      study_id=study_id,
      container_uri=_CLOUD_FIT_IMAGE,
      # Optional `MachineConfig` that represents the configuration for the
      # general workers in a distribution cluster. More options see:
      # https://github.com/tensorflow/cloud/blob/master/src/python/tensorflow_cloud/core/machine_config.py
      replica_config=machine_config.COMMON_MACHINE_CONFIGS['K80_1X'],
      # Optional total number of workers in a distribution cluster including a
      # chief worker.
      replica_count=2)

  return tfx.components.TunerFnResult(
      tuner=tuner,
      fit_kwargs={
          'steps_per_epoch': fn_args.train_steps,
          'validation_steps': fn_args.eval_steps,
          'train_files': fn_args.train_files,
          'eval_files': fn_args.eval_files,
          'transform_graph_path': fn_args.transform_graph_path,
          'label_key': _LABEL_KEY,
          'train_batch_size': _TRAIN_BATCH_SIZE,
          'eval_batch_size': _EVAL_BATCH_SIZE,
      })


# TFX Trainer will call this function.
def run_fn(fn_args: tfx.components.FnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args as name/value pairs. See
      https://www.tensorflow.org/tfx/api_docs/python/tfx/components/trainer/fn_args_utils/FnArgs.
      - train_files: List of file paths containing training tf.Example data.
      - eval_files: List of file paths containing eval tf.Example data.
      - data_accessor: Contains factories that can create tf.data.Datasets or
        other means to access the train/eval data. They provide a uniform way of
        accessing data, regardless of how the data is stored on disk.
      - train_steps: number of train steps.
      - eval_steps: number of eval steps.
      - transform_output: A uri to a path containing statistics and metadata
        from TFTransform component. produced by TFT. Will be None if not
        specified.
      - model_run_dir: A single uri for the output directory of model training
        related files.
      - hyperparameters: An optional keras_tuner.HyperParameters config.
  """
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  train_dataset = _input_fn(
      fn_args.train_files,
      fn_args.data_accessor,
      tf_transform_output,
      batch_size=_TRAIN_BATCH_SIZE)

  eval_dataset = _input_fn(
      fn_args.eval_files,
      fn_args.data_accessor,
      tf_transform_output,
      batch_size=_EVAL_BATCH_SIZE)

  if fn_args.hyperparameters:
    hparams = keras_tuner.HyperParameters.from_config(fn_args.hyperparameters)
  else:
    # This is a shown case when hyperparameters is decided and Tuner is removed
    # from the pipeline. User can also inline the hyperparameters directly in
    # _build_keras_model.
    hparams = _get_hyperparameters()
  logging.info('HyperParameters for training: %s', hparams.get_config())

  mirrored_strategy = tf.distribute.MirroredStrategy()
  with mirrored_strategy.scope():
    model = _build_keras_model(hparams)

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
          _get_tf_examples_serving_signature(model, tf_transform_output),
      'transform_features':
          _get_transform_features_signature(model, tf_transform_output),
  }
  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
