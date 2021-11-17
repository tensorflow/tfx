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
"""Python source file includes CIFAR10 utils for Keras model.

The utilities in this file are used to build a model with native Keras.
This module file will be used in Transform and generic Trainer.
"""

import os
from typing import List
import absl
import tensorflow as tf
import tensorflow_transform as tft

from tfx.components.trainer.fn_args_utils import DataAccessor
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.components.trainer.rewriting import converters
from tfx.components.trainer.rewriting import rewriter
from tfx.components.trainer.rewriting import rewriter_factory
from tfx.dsl.io import fileio
from tfx_bsl.tfxio import dataset_options

import flatbuffers
from tflite_support import metadata_schema_py_generated as _metadata_fb
from tflite_support import metadata as _metadata

# When training on the whole dataset use following constants instead.
# This setting should give ~91% accuracy on the whole test set
# _TRAIN_DATA_SIZE = 50000
# _EVAL_DATA_SIZE = 10000
# _TRAIN_BATCH_SIZE = 64
# _EVAL_BATCH_SIZE = 64
# _CLASSIFIER_LEARNING_RATE = 3e-4
# _FINETUNE_LEARNING_RATE = 5e-5
# _CLASSIFIER_EPOCHS = 12

_TRAIN_DATA_SIZE = 128
_EVAL_DATA_SIZE = 128
_TRAIN_BATCH_SIZE = 32
_EVAL_BATCH_SIZE = 32
_CLASSIFIER_LEARNING_RATE = 1e-3
_FINETUNE_LEARNING_RATE = 7e-6
_CLASSIFIER_EPOCHS = 30

_IMAGE_KEY = 'image'
_LABEL_KEY = 'label'

_TFLITE_MODEL_NAME = 'tflite'


def _transformed_name(key):
  return key + '_xf'


def _get_serve_image_fn(model):
  """Returns a function that feeds the input tensor into the model."""

  @tf.function
  def serve_image_fn(image_tensor):
    """Returns the output to be used in the serving signature.

    Args:
      image_tensor: A tensor represeting input image. The image should have 3
        channels.

    Returns:
      The model's predicton on input image tensor
    """
    return model(image_tensor)

  return serve_image_fn


def _image_augmentation(image_features):
  """Perform image augmentation on batches of images .

  Args:
    image_features: a batch of image features

  Returns:
    The augmented image features
  """
  batch_size = tf.shape(image_features)[0]
  image_features = tf.image.random_flip_left_right(image_features)
  image_features = tf.image.resize_with_crop_or_pad(image_features, 250, 250)
  image_features = tf.image.random_crop(image_features,
                                        (batch_size, 224, 224, 3))
  return image_features


def _data_augmentation(feature_dict):
  """Perform data augmentation on batches of data.

  Args:
    feature_dict: a dict containing features of samples

  Returns:
    The feature dict with augmented features
  """
  image_features = feature_dict[_transformed_name(_IMAGE_KEY)]
  image_features = _image_augmentation(image_features)
  feature_dict[_transformed_name(_IMAGE_KEY)] = image_features
  return feature_dict


def _input_fn(file_pattern: List[str],
              data_accessor: DataAccessor,
              tf_transform_output: tft.TFTransformOutput,
              is_train: bool = False,
              batch_size: int = 200) -> tf.data.Dataset:
  """Generates features and label for tuning/training.

  Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    data_accessor: DataAccessor for converting input to RecordBatch.
    tf_transform_output: A TFTransformOutput.
    is_train: Whether the input dataset is train split or not.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
  dataset = data_accessor.tf_dataset_factory(
      file_pattern,
      dataset_options.TensorFlowDatasetOptions(
          batch_size=batch_size, label_key=_transformed_name(_LABEL_KEY)),
      tf_transform_output.transformed_metadata.schema)
  # Apply data augmentation. We have to do data augmentation here because
  # we need to apply data agumentation on-the-fly during training. If we put
  # it in Transform, it will only be applied once on the whole dataset, which
  # will lose the point of data augmentation.
  if is_train:
    dataset = dataset.map(lambda x, y: (_data_augmentation(x), y))

  return dataset


def _freeze_model_by_percentage(model: tf.keras.Model, percentage: float):
  """Freeze part of the model based on specified percentage.

  Args:
    model: The keras model need to be partially frozen
    percentage: the percentage of layers to freeze

  Raises:
    ValueError: Invalid values.
  """
  if percentage < 0 or percentage > 1:
    raise ValueError('Freeze percentage should between 0.0 and 1.0')

  if not model.trainable:
    raise ValueError(
        'The model is not trainable, please set model.trainable to True')

  num_layers = len(model.layers)
  num_layers_to_freeze = int(num_layers * percentage)
  for idx, layer in enumerate(model.layers):
    if idx < num_layers_to_freeze:
      layer.trainable = False
    else:
      layer.trainable = True


def _build_keras_model() -> tf.keras.Model:
  """Creates a Image classification model with MobileNet backbone.

  Returns:
    The image classifcation Keras Model and the backbone MobileNet model
  """
  # We create a MobileNet model with weights pre-trained on ImageNet.
  # We remove the top classification layer of the MobileNet, which was
  # used for classifying ImageNet objects. We will add our own classification
  # layer for CIFAR10 later. We use average pooling at the last convolution
  # layer to get a 1D vector for classifcation, which is consistent with the
  # origin MobileNet setup
  base_model = tf.keras.applications.MobileNet(
      input_shape=(224, 224, 3),
      include_top=False,
      weights='imagenet',
      pooling='avg')
  base_model.input_spec = None

  # We add a Dropout layer at the top of MobileNet backbone we just created to
  # prevent overfiting, and then a Dense layer to classifying CIFAR10 objects
  model = tf.keras.Sequential([
      tf.keras.layers.InputLayer(
          input_shape=(224, 224, 3), name=_transformed_name(_IMAGE_KEY)),
      base_model,
      tf.keras.layers.Dropout(0.1),
      tf.keras.layers.Dense(10, activation='softmax')
  ])

  # Freeze the whole MobileNet backbone to first train the top classifer only
  _freeze_model_by_percentage(base_model, 1.0)

  model.compile(
      loss='sparse_categorical_crossentropy',
      optimizer=tf.keras.optimizers.RMSprop(lr=_CLASSIFIER_LEARNING_RATE),
      metrics=['sparse_categorical_accuracy'])
  model.summary(print_fn=absl.logging.info)

  return model, base_model


# TFX Transform will call this function.
def preprocessing_fn(inputs):
  """tf.transform's callback function for preprocessing inputs.

  Args:
    inputs: map from feature keys to raw not-yet-transformed features.

  Returns:
    Map from string feature key to transformed feature operations.
  """
  outputs = {}

  # tf.io.decode_png function cannot be applied on a batch of data.
  # We have to use tf.map_fn
  image_features = tf.map_fn(
      lambda x: tf.io.decode_png(x[0], channels=3),
      inputs[_IMAGE_KEY],
      dtype=tf.uint8)
  # image_features = tf.cast(image_features, tf.float32)
  image_features = tf.image.resize(image_features, [224, 224])
  image_features = tf.keras.applications.mobilenet.preprocess_input(
      image_features)

  outputs[_transformed_name(_IMAGE_KEY)] = image_features
  # TODO(b/157064428): Support label transformation for Keras.
  # Do not apply label transformation as it will result in wrong evaluation.
  outputs[_transformed_name(_LABEL_KEY)] = inputs[_LABEL_KEY]

  return outputs


def _write_metadata(model_path: str, label_map_path: str, mean: List[float],
                    std: List[float]):
  """Add normalization option and label map TFLite metadata to the model.

  Args:
    model_path: The path of the TFLite model
    label_map_path: The path of the label map file
    mean: The mean value used to normalize input image tensor
    std: The standard deviation used to normalize input image tensor
  """

  # Creates flatbuffer for model information.
  model_meta = _metadata_fb.ModelMetadataT()

  # Creates flatbuffer for model input metadata.
  # Here we add the input normalization info to input metadata.
  input_meta = _metadata_fb.TensorMetadataT()
  input_normalization = _metadata_fb.ProcessUnitT()
  input_normalization.optionsType = (
      _metadata_fb.ProcessUnitOptions.NormalizationOptions)
  input_normalization.options = _metadata_fb.NormalizationOptionsT()
  input_normalization.options.mean = mean
  input_normalization.options.std = std
  input_meta.processUnits = [input_normalization]

  # Creates flatbuffer for model output metadata.
  # Here we add label file to output metadata.
  output_meta = _metadata_fb.TensorMetadataT()
  label_file = _metadata_fb.AssociatedFileT()
  label_file.name = os.path.basename(label_map_path)
  label_file.type = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS
  output_meta.associatedFiles = [label_file]

  # Creates subgraph to contain input and output information,
  # and add subgraph to the model information.
  subgraph = _metadata_fb.SubGraphMetadataT()
  subgraph.inputTensorMetadata = [input_meta]
  subgraph.outputTensorMetadata = [output_meta]
  model_meta.subgraphMetadata = [subgraph]

  # Serialize the model metadata buffer we created above using flatbuffer
  # builder.
  b = flatbuffers.Builder(0)
  b.Finish(
      model_meta.Pack(b), _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
  metadata_buf = b.Output()

  # Populates metadata and label file to the model file.
  populator = _metadata.MetadataPopulator.with_model_file(model_path)
  populator.load_metadata_buffer(metadata_buf)
  populator.load_associated_files([label_map_path])
  populator.populate()


# TFX Trainer will call this function.
def run_fn(fn_args: FnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.

  Raises:
    ValueError: if invalid inputs.
  """
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  train_dataset = _input_fn(
      fn_args.train_files,
      fn_args.data_accessor,
      tf_transform_output,
      is_train=True,
      batch_size=_TRAIN_BATCH_SIZE)
  eval_dataset = _input_fn(
      fn_args.eval_files,
      fn_args.data_accessor,
      tf_transform_output,
      is_train=False,
      batch_size=_EVAL_BATCH_SIZE)

  model, base_model = _build_keras_model()

  absl.logging.info('Tensorboard logging to {}'.format(fn_args.model_run_dir))
  # Write logs to path
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=fn_args.model_run_dir, update_freq='batch')

  # Our training regime has two phases: we first freeze the backbone and train
  # the newly added classifier only, then unfreeze part of the backbone and
  # fine-tune with classifier jointly.
  steps_per_epoch = int(_TRAIN_DATA_SIZE / _TRAIN_BATCH_SIZE)
  total_epochs = int(fn_args.train_steps / steps_per_epoch)
  if _CLASSIFIER_EPOCHS > total_epochs:
    raise ValueError('Classifier epochs is greater than the total epochs')

  absl.logging.info('Start training the top classifier')
  model.fit(
      train_dataset,
      epochs=_CLASSIFIER_EPOCHS,
      steps_per_epoch=steps_per_epoch,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps,
      callbacks=[tensorboard_callback])

  absl.logging.info('Start fine-tuning the model')
  # Unfreeze the top MobileNet layers and do joint fine-tuning
  _freeze_model_by_percentage(base_model, 0.9)

  # We need to recompile the model because layer properties have changed
  model.compile(
      loss='sparse_categorical_crossentropy',
      optimizer=tf.keras.optimizers.RMSprop(lr=_FINETUNE_LEARNING_RATE),
      metrics=['sparse_categorical_accuracy'])
  model.summary(print_fn=absl.logging.info)

  model.fit(
      train_dataset,
      initial_epoch=_CLASSIFIER_EPOCHS,
      epochs=total_epochs,
      steps_per_epoch=steps_per_epoch,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps,
      callbacks=[tensorboard_callback])

  # Prepare the TFLite model used for serving in MLKit
  signatures = {
      'serving_default':
          _get_serve_image_fn(model).get_concrete_function(
              tf.TensorSpec(
                  shape=[None, 224, 224, 3],
                  dtype=tf.float32,
                  name=_transformed_name(_IMAGE_KEY)))
  }

  temp_saving_model_dir = os.path.join(fn_args.serving_model_dir, 'temp')
  model.save(temp_saving_model_dir, save_format='tf', signatures=signatures)

  tfrw = rewriter_factory.create_rewriter(
      rewriter_factory.TFLITE_REWRITER,
      name='tflite_rewriter')
  converters.rewrite_saved_model(temp_saving_model_dir,
                                 fn_args.serving_model_dir, tfrw,
                                 rewriter.ModelType.TFLITE_MODEL)

  # Add necessary TFLite metadata to the model in order to use it within MLKit
  # TODO(dzats@): Handle label map file path more properly, currently
  # hard-coded.
  tflite_model_path = os.path.join(fn_args.serving_model_dir,
                                   _TFLITE_MODEL_NAME)
  # TODO(dzats@): Extend the TFLite rewriter to be able to add TFLite metadata
  #@ to the model.
  _write_metadata(
      model_path=tflite_model_path,
      label_map_path=fn_args.custom_config['labels_path'],
      mean=[127.5],
      std=[127.5])

  fileio.rmtree(temp_saving_model_dir)
