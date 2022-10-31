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
"""Supplement for penguin_utils_base.py with specifics for Flax models.

**The Flax support in TFX is currently experimental.**

Using TFX with Flax instead of Keras requires only a few changes for the
Trainer component. However, Flax is simpler than Keras and does not come
pre-packaged with a training loop, model saving, and other bells and whistles,
hence the training code with Flax is more verbose. This file contains the
definition of the Flax model for Penguin dataset, the training loop, and the
model creation and saving.

The code in this file is structures in three parts: Part 1 contains
standard Flax code to define and train the model, independent of any TFX
specifics; Part 2 contains the customization of
TFX components to use the Flax model.
"""
import functools
from typing import Callable, Dict, List

import absl

from flax import linen as nn
from flax.metrics import tensorboard

import jax
from jax import numpy as jnp
from jax.experimental import jax2tf
import numpy as np
import optax

import tensorflow as tf
import tensorflow_transform as tft

from tfx import v1 as tfx
from tfx.examples.penguin import penguin_utils_base as base


# The transformed feature names
_FEATURE_KEYS_XF = list(map(base.transformed_name, base.FEATURE_KEYS))

# Type abbreviations: (B is the batch size)
_Array = np.ndarray
_InputBatch = Dict[str,
                   _Array]  # keys are _FEATURE_KEYS_XF and values f32[B, 1]
_LogitBatch = _Array  # of shape f32[B, 3]
_LabelBatch = _Array  # of shape int64[B, 1]
_Params = Dict[str, _Array]


### Part 1: Definition of the Flax model and its training loop.
#
# This part is standard Flax code, independent of any TFX specifics.
#
def _make_trained_model(train_data: tf.data.Dataset,
                        eval_data: tf.data.Dataset,
                        num_epochs: int, steps_per_epoch: int,
                        eval_steps_per_epoch: int, tensorboard_log_dir: str):
  """Execute model training and evaluation loop.

  Args:
    train_data: a dataset with training pairs (_InputBatch, _LabelBatch).
    eval_data: a dataset with evaluation pairs (_InputBatch, _LabelBatch).
    num_epochs: number of training epochs.
    steps_per_epoch: number of steps for a training epoch. Should be the number
       of samples in your train_data divided by the batch size.
    eval_steps_per_epoch: number of steps for evaluation at the end of each
       training epoch. Should be the number of samples in your eval_data
       divided by the batch size.
    tensorboard_log_dir: Directory where the tensorboard summaries are written.

  Returns:
    An instance of tf.Model.
  """
  learning_rate = 1e-2

  rng = jax.random.PRNGKey(0)

  summary_writer = tensorboard.SummaryWriter(tensorboard_log_dir)
  summary_writer.hparams(
      dict(
          learning_rate=learning_rate,
          num_epochs=num_epochs,
          steps_per_epoch=steps_per_epoch,
          eval_steps_per_epoch=eval_steps_per_epoch))

  rng, init_rng = jax.random.split(rng)
  # Initialize with some fake data of the proper shape.
  init_val = dict((feature, jnp.array([[1.]], dtype=jnp.float32))
                  for feature in _FEATURE_KEYS_XF)
  model = _FlaxPenguinModel()
  params = model.init(init_rng, init_val)['params']

  tx = optax.adam(learning_rate=learning_rate)
  opt_state = tx.init(params)

  for epoch in range(1, num_epochs + 1):
    params, opt_state, train_metrics = _train_epoch(model, tx,
                                                    params, opt_state,
                                                    train_data, steps_per_epoch)
    absl.logging.info('Flax train epoch: %d, loss: %.4f, accuracy: %.2f', epoch,
                      train_metrics['loss'], train_metrics['accuracy'] * 100)

    eval_metrics = _eval_epoch(model, params, eval_data,
                               eval_steps_per_epoch)
    absl.logging.info('Flax eval epoch: %d, loss: %.4f, accuracy: %.2f', epoch,
                      eval_metrics['loss'], eval_metrics['accuracy'] * 100)
    summary_writer.scalar('epoch_train_loss', train_metrics['loss'], epoch)
    summary_writer.scalar('epoch_train_accuracy', train_metrics['accuracy'],
                          epoch)
    summary_writer.scalar('epoch_eval_loss', eval_metrics['loss'], epoch)
    summary_writer.scalar('epoch_eval_accuracy', eval_metrics['accuracy'],
                          epoch)

  summary_writer.flush()

  # The prediction function for the trained model
  def predict(params: _Params, inputs: _InputBatch):
    return model.apply({'params': params}, inputs)

  trained_params = params

  # Convert the prediction function to TF, with a variable batch dimension
  # for all inputs.
  tf_fn = jax2tf.convert(predict, with_gradient=False, enable_xla=True,
                         polymorphic_shapes=(None, '(b, 1)'))

  # Create tf.Variables for the parameters. If you want more useful variable
  # names, you can use `tree.map_structure_with_path` from the `dm-tree`
  # package.
  param_vars = tf.nest.map_structure(
      # Due to a bug in SavedModel it is not possible to use tf.GradientTape
      # on a function converted with jax2tf and loaded from SavedModel.
      # Thus, we mark the variables as non-trainable to ensure that users of
      # the SavedModel will not try to fine tune them.
      lambda param: tf.Variable(param, trainable=False),
      trained_params)
  tf_graph = tf.function(
      lambda inputs: tf_fn(param_vars, inputs),
      autograph=False,
      experimental_compile=True)
  return _SavedModelWrapper(tf_graph, param_vars)


class _FlaxPenguinModel(nn.Module):
  """The model definition."""

  @nn.compact
  def __call__(self, x_dict: _InputBatch) -> _LogitBatch:
    # Each feature is of shape f32[B, 1]
    x_tuple = tuple(x_dict[feature] for feature in _FEATURE_KEYS_XF)
    x_array = jnp.concatenate(x_tuple, axis=-1)  # shape: f32[B, 4]
    assert x_array.ndim == 2
    assert x_array.shape[1] == 4
    x = x_array

    x = nn.Dense(features=8)(x)
    x = nn.relu(x)
    x = nn.Dense(features=8)(x)
    x = nn.relu(x)
    x = nn.Dense(features=3)(x)
    x = nn.log_softmax(x, axis=-1)
    return x


def _train_epoch(model: _FlaxPenguinModel, tx: optax.GradientTransformation,
                 params: _Params, opt_state: optax.OptState,
                 train_data: tf.data.Dataset,
                 steps_per_epoch: int):
  """Train for a single epoch."""
  batch_metrics = []
  steps = 0
  for inputs, labels in train_data.as_numpy_iterator():
    params, opt_state, metrics = _train_step(model, tx, params, opt_state,
                                             inputs, labels)
    batch_metrics.append(metrics)
    steps += 1
    if steps == steps_per_epoch:
      break

  # compute mean of metrics across each batch in epoch.
  epoch_metrics_np = _mean_epoch_metrics(jax.device_get(batch_metrics))
  return params, opt_state, epoch_metrics_np


def _eval_epoch(model: _FlaxPenguinModel, params: _Params,
                eval_data: tf.data.Dataset,
                steps_per_epoch: int):
  """Validate for a single epoch."""
  batch_metrics = []
  steps = 0
  for inputs, labels in eval_data.as_numpy_iterator():
    metrics = _eval_step(model, params, inputs, labels)
    batch_metrics.append(metrics)
    steps += 1
    if steps == steps_per_epoch:
      break

  # compute mean of metrics across each batch in epoch.
  return _mean_epoch_metrics(jax.device_get(batch_metrics))


@functools.partial(jax.jit, static_argnums=(0, 1))
def _train_step(model: _FlaxPenguinModel, tx: optax.GradientTransformation,
                params: _Params, opt_state: optax.OptState,
                inputs: _InputBatch, labels: _LabelBatch):
  """Train for a single step, given a batch of inputs and labels."""

  def loss_fn(params):
    logits = model.apply({'params': params}, inputs)
    loss = _categorical_cross_entropy_loss(logits, labels)
    return loss, logits

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grads = grad_fn(params)
  updates, opt_state = tx.update(grads, opt_state)
  params = optax.apply_updates(params, updates)
  metrics = _compute_metrics(logits, labels)
  return params, opt_state, metrics


@functools.partial(jax.jit, static_argnums=0)
def _eval_step(model: _FlaxPenguinModel, params: _Params, inputs: _InputBatch,
               labels: _LabelBatch):
  logits = model.apply({'params': params}, inputs)
  return _compute_metrics(logits, labels)


def _categorical_cross_entropy_loss(logits: _LogitBatch, labels: _LabelBatch):
  # assumes that the logits use log_softmax activations.
  onehot_labels = (labels == jnp.arange(3)[None]).astype(jnp.float32)
  # onehot_labels: f32[B, 3]
  z = -jnp.sum(onehot_labels * logits, axis=-1)  # f32[B]
  return jnp.mean(z)  # f32


def _compute_metrics(logits: _LogitBatch, labels: _LabelBatch):
  # assumes that the logits use log_softmax activations.
  loss = _categorical_cross_entropy_loss(logits, labels)
  accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels[..., 0])
  return {
      'loss': loss,
      'accuracy': accuracy,
  }


def _mean_epoch_metrics(
    batch_metrics: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
  batch_metrics_np = jax.device_get(batch_metrics)
  return {
      metric_name:
      np.mean([metrics[metric_name] for metrics in batch_metrics_np])
      for metric_name in batch_metrics_np[0]
  }


class _SavedModelWrapper(tf.train.Checkpoint):
  """Wraps a function and its parameters for saving to a SavedModel.

  Implements the interface described at
  https://www.tensorflow.org/hub/reusable_saved_models.

  This class contains all the code needed to convert a Flax model to a
  TensorFlow saved model.
  """

  def __init__(self,
               tf_graph: Callable[[_InputBatch], _Array],
               param_vars: Dict[str, tf.Variable]):
    """Builds the tf.Module.

    Args:
      tf_graph: a tf.function taking one argument (the inputs), which can be be
        tuples/lists/dictionaries of np.ndarray or tensors. The function may
        have references to the tf.Variables in `param_vars`.
      param_vars: the parameters, as tuples/lists/dictionaries of tf.Variable,
        to be saved as the variables of the SavedModel.
    """
    super().__init__()
    # Implement the interface from
    # https://www.tensorflow.org/hub/reusable_saved_models
    self.variables = tf.nest.flatten(param_vars)
    self.trainable_variables = [v for v in self.variables if v.trainable]
    self._tf_graph = tf_graph

  @tf.function
  def __call__(self, inputs):
    return self._tf_graph(inputs)


### Part 2: Customization of TFX components

# TFX Transform will call this function.
preprocessing_fn = base.preprocessing_fn


# TFX Trainer will call this function.
def run_fn(fn_args: tfx.components.FnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  train_dataset = base.input_fn(
      fn_args.train_files,
      fn_args.data_accessor,
      tf_transform_output,
      base.TRAIN_BATCH_SIZE)

  eval_dataset = base.input_fn(
      fn_args.eval_files,
      fn_args.data_accessor,
      tf_transform_output,
      base.EVAL_BATCH_SIZE)

  model = _make_trained_model(
      train_dataset,
      eval_dataset,
      num_epochs=1,
      steps_per_epoch=fn_args.train_steps,
      eval_steps_per_epoch=fn_args.eval_steps,
      tensorboard_log_dir=fn_args.model_run_dir)

  signatures = base.make_serving_signatures(model, tf_transform_output)
  tf.saved_model.save(model, fn_args.serving_model_dir, signatures=signatures)
