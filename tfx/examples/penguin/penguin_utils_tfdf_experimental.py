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
"""Supplement for penguin_utils_base.py for TF Decision Forests models.

**The TFDF support in TFX is currently experimental.**

This module file will be used in the Transform, Tuner and generic Trainer
components.
"""

from typing import List

import keras_tuner as kt

import tensorflow as tf
import tensorflow_decision_forests as tfdf
import tensorflow_transform as tft
from tfx import v1 as tfx
from tfx.examples.penguin import penguin_utils_base as base

from tfx_bsl.public import tfxio

# TFX Transform will call this function.
# Note: many decision tree algorithms do not benefit from feature preprocessing.
# For more info, please refer to
# https://github.com/tensorflow/decision-forests/blob/main/documentation/migration.md#do-not-preprocess-the-features.
preprocessing_fn = base.preprocessing_fn


def _get_hyperparameters() -> kt.HyperParameters:
  """Returns hyperparameters for building a TFDF model."""
  hp = kt.HyperParameters()
  # Defines search space.
  # For a complete list of configuration parameters of learners, please refer to
  # https://github.com/google/yggdrasil-decision-forests/blob/main/documentation/learners.md
  hp.Int('max_depth', 3, 8, default=6)
  hp.Float('shrinkage', 0.01, 0.1, default=0.1)
  hp.Boolean('use_hessian_gain', default=False)
  return hp


def _make_keras_model(hparams: kt.HyperParameters) -> tf.keras.Model:
  """Creates a TFDF Keras model for classifying penguin data.

  Args:
    hparams: Holds HyperParameters for tuning.

  Returns:
    A Keras Model.
  """
  return tfdf.keras.GradientBoostedTreesModel(
      max_depth=hparams.get('max_depth'),
      shrinkage=hparams.get('shrinkage'),
      use_hessian_gain=hparams.get('use_hessian_gain'))


def input_fn(file_pattern: List[str],
             data_accessor: tfx.components.DataAccessor,
             tf_transform_output: tft.TFTransformOutput,
             batch_size: int) -> tf.data.Dataset:
  """Generates features and label for tuning/training for a single epoch.

  Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    data_accessor: DataAccessor for converting input to RecordBatch.
    tf_transform_output: A TFTransformOutput.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
  return data_accessor.tf_dataset_factory(
      file_pattern,
      tfxio.TensorFlowDatasetOptions(
          batch_size=batch_size, num_epochs=1,
          label_key=base.transformed_name(base._LABEL_KEY)),  # pylint: disable=protected-access
      tf_transform_output.transformed_metadata.schema)


# TFX Tuner will call this function.
def tuner_fn(fn_args: tfx.components.FnArgs) -> tfx.components.TunerFnResult:
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
  # RandomSearch is a subclass of kt.Tuner which inherits from
  # BaseTuner.
  tuner = kt.RandomSearch(
      _make_keras_model,
      max_trials=6,
      hyperparameters=_get_hyperparameters(),
      allow_new_entries=False,
      objective=kt.Objective('val_loss', 'min'),
      directory=fn_args.working_dir,
      project_name='penguin_tuning')

  transform_graph = tft.TFTransformOutput(fn_args.transform_graph_path)

  train_dataset = input_fn(fn_args.train_files, fn_args.data_accessor,
                           transform_graph, base.TRAIN_BATCH_SIZE)

  eval_dataset = input_fn(fn_args.eval_files, fn_args.data_accessor,
                          transform_graph, base.EVAL_BATCH_SIZE)

  return tfx.components.TunerFnResult(
      tuner=tuner,
      fit_kwargs={
          'x': train_dataset,
          'validation_data': eval_dataset,
      })


# TFX Trainer will call this function.
def run_fn(fn_args: tfx.components.FnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  train_dataset = input_fn(fn_args.train_files, fn_args.data_accessor,
                           tf_transform_output, base.TRAIN_BATCH_SIZE)

  eval_dataset = input_fn(fn_args.eval_files, fn_args.data_accessor,
                          tf_transform_output, base.EVAL_BATCH_SIZE)

  if fn_args.hyperparameters:
    hparams = kt.HyperParameters.from_config(fn_args.hyperparameters)
  else:
    # This is a shown case when hyperparameters is decided and Tuner is removed
    # from the pipeline. User can also inline the hyperparameters directly in
    # _make_keras_model.
    hparams = _get_hyperparameters()

  model = _make_keras_model(hparams)

  # Write logs to path
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=fn_args.model_run_dir, update_freq='batch')

  model.fit(
      train_dataset,
      validation_data=eval_dataset,
      callbacks=[tensorboard_callback])

  signatures = base.make_serving_signatures(model, tf_transform_output)
  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
