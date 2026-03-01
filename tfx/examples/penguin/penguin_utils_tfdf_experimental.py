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

**TensorFlow Decision Forests (TF-DF) support in TFX is currently
experimental.**

TensorFlow Decision Forests (https://www.tensorflow.org/decision_forests) is a
collection collection of state-of-the-art algorithms for the training, serving
and interpretation of Decision Forest models. The library is a collection of
Keras models and supports classification, regression and ranking.

To learn about TF-DF, and about the difference between Decision Forests and
Neural Networks, check the beginner colab
(https://www.tensorflow.org/decision_forests/tutorials/beginner_colab) for an
introduction to TF-DF.

Important remark:

TF-DF relies on TensorFlow Custom Ops
(https://www.tensorflow.org/lite/guide/ops_custom) for training and serving
(soon, only for training). To run correctly, the various TFX component that
handle the model (e.g., training, evaluation) need to be configured with the
TF-DF ops.

- Training & evaluation & validation: In python code, the ops are loaded when
  importing the TF-DF library: `import tensorflow_decision_forests as tfdf`.

- Serving: TF-DF requires a version of TensorFlow Serving compiled with
  TensorFlow Decision Forests ops
  (https://github.com/tensorflow/decision-forests/releases/tag/serving-0.2.6).
  Alternatively, you can follow the instructions
  (https://www.tensorflow.org/decision_forests/tensorflow_serving) or use the
  TFServing+TF-DF compilation script
  (https://github.com/tensorflow/decision-forests/tree/main/tools/tf_serving)
  to compile TF Serving with TF-DF support yourself.

Note: If a TFX component is not configured with TF-DF custom ops, you will see
errors such as: "Op type not registered 'SimpleMLInferenceOpWithHandle'".

In case of issues, you can ask for help on the TensorFlow Forum
(https://discuss.tensorflow.org). This particular discussion
(https://discuss.tensorflow.org/t/tensorflow-decision-forests-with-tfx-model-serving-and-evaluation/2137)
also contains some good pointers.

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

_KEY_MODEL_TYPE = 'model_type'

# Random Forest and Gradient Boosted trees are the two most popular algorithm to
# train decision forests.
_KEY_RANDOM_FOREST = 'RANDOM_FOREST'
_KEY_GRADIENT_BOOSTED_TREES = 'GRADIENT_BOOSTED_TREES'


def _get_hyperparameters() -> kt.HyperParameters:
  """Creates a small hyperparameter search space for TF-DF.

  TF-DF offers multiple learning algorithms, and each one has its own
  hyperparameters. In this example, we consider the Random Forest and Gradient
  Boosted Trees models.

  Their hyperparameters are described at:

  https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/GradientBoostedTreesModel
  https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/RandomForestModel
  https://github.com/google/yggdrasil-decision-forests/blob/main/documentation/learners.md

  See the user manual to get an idea of which hyperparameters are best suited
  for tuning:
  https://github.com/google/yggdrasil-decision-forests/blob/main/documentation/user_manual.md#manual-tuning-of-hyper-parameters

  Returns:
    Valid range and default value of few of the GBT hyperparameters.
  """
  hp = kt.HyperParameters()

  # Select the decision forest learning algorithm.
  hp.Choice(
      _KEY_MODEL_TYPE, [
          _KEY_RANDOM_FOREST,
          _KEY_GRADIENT_BOOSTED_TREES,
      ],
      default=_KEY_GRADIENT_BOOSTED_TREES)

  # Hyperparameter for the Random Forest
  hp.Int(
      'max_depth',
      10,
      24,
      default=16,
      parent_name=_KEY_MODEL_TYPE,
      parent_values=[_KEY_RANDOM_FOREST])
  hp.Int(
      'min_examples',
      2,
      20,
      default=6,
      parent_name=_KEY_MODEL_TYPE,
      parent_values=[_KEY_RANDOM_FOREST])

  # Hyperparameters for the Gradient Boosted Trees
  hp.Float(
      'num_candidate_attributes_ratio',
      0.5,
      1.0,
      default=1.0,
      parent_name=_KEY_MODEL_TYPE,
      parent_values=[_KEY_GRADIENT_BOOSTED_TREES])
  hp.Boolean(
      'use_hessian_gain',
      default=False,
      parent_name=_KEY_MODEL_TYPE,
      parent_values=[_KEY_GRADIENT_BOOSTED_TREES])
  hp.Choice(
      'growing_strategy', ['LOCAL', 'BEST_FIRST_GLOBAL'],
      default='LOCAL',
      parent_name=_KEY_MODEL_TYPE,
      parent_values=[_KEY_GRADIENT_BOOSTED_TREES])

  return hp


def _make_keras_model(hparams: kt.HyperParameters) -> tf.keras.Model:
  """Creates a TF-DF Keras model.

  Args:
    hparams: Hyperparameters of the model.

  Returns:
    A Keras Model.
  """

  # Note: The input features are not specified. Therefore, all the columns
  # specified in the Transform are used as input features, and their semantic
  # (e.g. numerical, categorical) is inferred automatically.
  common_args = {
      'verbose': 2,
      'task': tfdf.keras.Task.CLASSIFICATION,
  }

  if hparams.get(_KEY_MODEL_TYPE) == _KEY_RANDOM_FOREST:
    return tfdf.keras.RandomForestModel(
        max_depth=hparams.get('max_depth'),
        min_examples=hparams.get('min_examples'),
        **common_args)

  elif hparams.get(_KEY_MODEL_TYPE) == _KEY_GRADIENT_BOOSTED_TREES:
    return tfdf.keras.GradientBoostedTreesModel(
        num_candidate_attributes_ratio=hparams.get(
            'num_candidate_attributes_ratio'),
        use_hessian_gain=hparams.get('use_hessian_gain'),
        growing_strategy=hparams.get('growing_strategy'),
        **common_args)

  else:
    raise ValueError('Unknown model type')


def input_fn(file_pattern: List[str],
             data_accessor: tfx.components.DataAccessor,
             tf_transform_output: tft.TFTransformOutput,
             batch_size: int) -> tf.data.Dataset:
  """Creates a tf.Dataset for training or evaluation.

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
          # The batch size has not impact on the model quality. However, small
          # batch size might be slower.
          batch_size=batch_size,
          # TF-DF models should be trained on exactly one epoch.
          num_epochs=1,
          # Datasets should not be shuffled.
          shuffle=False,
          label_key=base.transformed_name(base._LABEL_KEY)),  # pylint: disable=protected-access
      tf_transform_output.transformed_metadata.schema)


# TFX Tuner will call this function.
def tuner_fn(fn_args: tfx.components.FnArgs) -> tfx.components.TunerFnResult:
  """Builds a Keras Tuner for the model.

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
      # The model is tuned on the loss computed on the TFX validation dataset.
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
  model.fit(train_dataset, validation_data=eval_dataset)

  print('Trained model:')
  model.summary()

  # Export the tensorboard logs.
  model.make_inspector().export_to_tensorboard(fn_args.model_run_dir)

  signatures = base.make_serving_signatures(model, tf_transform_output)
  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
