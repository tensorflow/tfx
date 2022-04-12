# Copyright 2021 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module file."""

import tensorflow as tf
import tensorflow_ranking as tfr
import tensorflow_transform as tft
from tfx.examples.ranking import features
from tfx.examples.ranking import struct2tensor_parsing_utils
from tfx_bsl.public import tfxio


def make_decoder():
  """Creates a data decoder that that decodes ELWC records to tensors.

  A DataView (see "TfGraphDataViewProvider" component in the pipeline)
  will refer to this decoder. And any components that consumes the data
  with the DataView applied will use this decoder.

  Returns:
    A ELWC decoder.
  """
  context_features, example_features, label_feature = features.get_features()

  return struct2tensor_parsing_utils.ELWCDecoder(
      name='ELWCDecoder',
      context_features=context_features,
      example_features=example_features,
      size_feature_name=features.LIST_SIZE_FEATURE_NAME,
      label_feature=label_feature)


def preprocessing_fn(inputs):
  """Transform preprocessing_fn."""

  # generate a shared vocabulary.
  _ = tft.vocabulary(
      tf.concat([
          inputs[features.QUERY_TOKENS].flat_values,
          inputs[features.DOCUMENT_TOKENS].flat_values
      ],
                axis=0),
      vocab_filename='shared_vocab')
  return inputs


def run_fn(trainer_fn_args):
  """TFX trainer entry point."""

  tf_transform_output = tft.TFTransformOutput(trainer_fn_args.transform_output)
  hparams = dict(
      batch_size=32,
      embedding_dimension=20,
      learning_rate=0.05,
      dropout_rate=0.8,
      hidden_layer_dims=[64, 32, 16],
      loss='approx_ndcg_loss',
      use_batch_norm=True,
      batch_norm_moment=0.99
  )

  train_dataset = _input_fn(trainer_fn_args.train_files,
                            trainer_fn_args.data_accessor,
                            hparams['batch_size'])
  eval_dataset = _input_fn(trainer_fn_args.eval_files,
                           trainer_fn_args.data_accessor,
                           hparams['batch_size'])

  model = _create_ranking_model(tf_transform_output, hparams)
  model.summary()
  log_dir = trainer_fn_args.model_run_dir
  # Write logs to path
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir, update_freq='batch')
  model.fit(
      train_dataset,
      steps_per_epoch=trainer_fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=trainer_fn_args.eval_steps,
      callbacks=[tensorboard_callback])

  # TODO(zhuo): Add support for Regress signature.
  @tf.function(input_signature=[tf.TensorSpec([None], tf.string)],
               autograph=False)
  def predict_serving_fn(serialized_elwc_records):
    decode_fn = trainer_fn_args.data_accessor.data_view_decode_fn
    decoded = decode_fn(serialized_elwc_records)
    decoded.pop(features.LABEL)
    return {tf.saved_model.PREDICT_OUTPUTS: model(decoded)}

  model.save(
      trainer_fn_args.serving_model_dir,
      save_format='tf',
      signatures={
          'serving_default':
              predict_serving_fn.get_concrete_function(),
      })


def _input_fn(file_patterns,
              data_accessor,
              batch_size) -> tf.data.Dataset:
  """Returns a dataset of decoded tensors."""

  def prepare_label(parsed_ragged_tensors):
    label = parsed_ragged_tensors.pop(features.LABEL)
    # Convert labels to a dense tensor.
    label = label.to_tensor(default_value=features.LABEL_PADDING_VALUE)
    return parsed_ragged_tensors, label

  # NOTE: this dataset already contains RaggedTensors from the Decoder.
  dataset = data_accessor.tf_dataset_factory(
      file_patterns,
      tfxio.TensorFlowDatasetOptions(batch_size=batch_size),
      schema=None)
  return dataset.map(prepare_label).repeat()


def _preprocess_keras_inputs(context_keras_inputs, example_keras_inputs,
                             tf_transform_output, hparams):
  """Preprocesses the inputs, including vocab lookup and embedding."""
  lookup_layer = tf.keras.layers.experimental.preprocessing.StringLookup(
      max_tokens=(
          tf_transform_output.vocabulary_size_by_name('shared_vocab') + 1),
      vocabulary=tf_transform_output.vocabulary_file_by_name('shared_vocab'),
      num_oov_indices=1,
      oov_token='[UNK#]',
      mask_token=None)
  embedding_layer = tf.keras.layers.Embedding(
      input_dim=(
          tf_transform_output.vocabulary_size_by_name('shared_vocab') + 1),
      output_dim=hparams['embedding_dimension'],
      embeddings_initializer=None,
      embeddings_constraint=None)
  def embedding(input_tensor):
    # TODO(b/158673891): Support weighted features.
    embedded_tensor = embedding_layer(lookup_layer(input_tensor))
    mean_embedding = tf.reduce_mean(embedded_tensor, axis=-2)
    # mean_embedding could be a dense tensor (context feature) or a ragged
    # tensor (example feature). if it's ragged, we densify it first.
    if isinstance(mean_embedding.type_spec, tf.RaggedTensorSpec):
      return struct2tensor_parsing_utils.make_ragged_densify_layer()(
          mean_embedding)
    return mean_embedding
  preprocessed_context_features, preprocessed_example_features = {}, {}
  context_features, example_features, _ = features.get_features()
  for feature in context_features:
    preprocessed_context_features[feature.name] = embedding(
        context_keras_inputs[feature.name])
  for feature in example_features:
    preprocessed_example_features[feature.name] = embedding(
        example_keras_inputs[feature.name])
  list_size = struct2tensor_parsing_utils.make_ragged_densify_layer()(
      context_keras_inputs[features.LIST_SIZE_FEATURE_NAME])
  list_size = tf.reshape(list_size, [-1])
  mask = tf.sequence_mask(list_size)

  return preprocessed_context_features, preprocessed_example_features, mask


def _create_ranking_model(tf_transform_output, hparams) -> tf.keras.Model:
  """Creates a Keras ranking model."""
  context_feature_specs, example_feature_specs, _ = features.get_features()
  context_keras_inputs, example_keras_inputs = (
      struct2tensor_parsing_utils.create_keras_inputs(
          context_feature_specs, example_feature_specs,
          features.LIST_SIZE_FEATURE_NAME))
  context_features, example_features, mask = _preprocess_keras_inputs(
      context_keras_inputs, example_keras_inputs, tf_transform_output, hparams)

  # Since argspec inspection is expensive, for keras layer,
  # layer_obj._call_spec.arg_names is a property that uses cached argspec for
  # call. We use this to determine whether the layer expects `inputs` as first
  # argument.
  # TODO(b/185176464): update tfr dependency to remove this branch.
  flatten_list = tfr.keras.layers.FlattenList()

  # TODO(kathywu): remove the except branch once changes to the call function
  # args in the Keras Layer have been released.
  try:
    first_arg_name = flatten_list._call_spec.arg_names[0]  # pylint: disable=protected-access
  except AttributeError:
    first_arg_name = flatten_list._call_fn_args[0]  # pylint: disable=protected-access
  if first_arg_name == 'inputs':
    (flattened_context_features, flattened_example_features) = flatten_list(
        inputs=(context_features, example_features, mask))
  else:
    (flattened_context_features,
     flattened_example_features) = flatten_list(context_features,
                                                example_features, mask)

  # Concatenate flattened context and example features along `list_size` dim.
  context_input = [
      tf.keras.layers.Flatten()(flattened_context_features[name])
      for name in sorted(flattened_context_features)
  ]
  example_input = [
      tf.keras.layers.Flatten()(flattened_example_features[name])
      for name in sorted(flattened_example_features)
  ]
  input_layer = tf.concat(context_input + example_input, 1)
  dnn = tf.keras.Sequential()
  if hparams['use_batch_norm']:
    dnn.add(
        tf.keras.layers.BatchNormalization(
            momentum=hparams['batch_norm_moment']))
  for layer_size in hparams['hidden_layer_dims']:
    dnn.add(tf.keras.layers.Dense(units=layer_size))
    if hparams['use_batch_norm']:
      dnn.add(tf.keras.layers.BatchNormalization(
          momentum=hparams['batch_norm_moment']))
    dnn.add(tf.keras.layers.Activation(activation=tf.nn.relu))
    dnn.add(tf.keras.layers.Dropout(rate=hparams['dropout_rate']))

  dnn.add(tf.keras.layers.Dense(units=1))

  # Since argspec inspection is expensive, for keras layer,
  # layer_obj._call_spec.arg_names is a property that uses cached argspec for
  # call. We use this to determine whether the layer expects `inputs` as first
  # argument.
  restore_list = tfr.keras.layers.RestoreList()

  # TODO(kathywu): remove the except branch once changes to the call function
  # args in the Keras Layer have been released.
  try:
    first_arg_name = flatten_list._call_spec.arg_names[0]  # pylint: disable=protected-access
  except AttributeError:
    first_arg_name = flatten_list._call_fn_args[0]  # pylint: disable=protected-access
  if first_arg_name == 'inputs':
    logits = restore_list(inputs=(dnn(input_layer), mask))
  else:
    logits = restore_list(dnn(input_layer), mask)

  model = tf.keras.Model(
      inputs={
          **context_keras_inputs,
          **example_keras_inputs
      },
      outputs=logits,
      name='dnn_ranking_model')
  model.compile(
      optimizer=tf.keras.optimizers.Adagrad(
          learning_rate=hparams['learning_rate']),
      loss=tfr.keras.losses.get(hparams['loss']),
      metrics=tfr.keras.metrics.default_keras_metrics())
  return model
