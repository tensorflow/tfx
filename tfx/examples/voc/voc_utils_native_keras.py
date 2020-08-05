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
"""Python source file includes CIFAR10 utils for Keras model.

The utilities in this file are used to build a model with native Keras.
This module file will be used in Transform and generic Trainer.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import List, Text

import absl
import tensorflow as tf
import tensorflow_transform as tft

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.builders import model_builder
from object_detection.utils import visualization_utils as viz_utils

from tfx.components.trainer.rewriting import converters
from tfx.components.trainer.rewriting import rewriter
from tfx.components.trainer.rewriting import rewriter_factory

from tfx.components.trainer.executor import TrainerFnArgs

# cifar10 dataset has 50000 train records, and 10000 val records
_TRAIN_DATA_SIZE = 50000
_EVAL_DATA_SIZE = 10000
_TRAIN_BATCH_SIZE = 64
_EVAL_BATCH_SIZE = 64

_IMAGE_KEY = 'image'
_BOX_KEY = 'objects/bbox'
_LABEL_KEY = 'objects/label'

def _transformed_name(key):
  return key + '_xf'

def _gzip_reader_fn(filenames):
  """Small utility returning a record reader that can read gzip'ed files."""
  return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def _get_serve_image_fn(model):
  """Returns a function that feeds the input tensor into the model."""

  @tf.function
  def serve_image_fn(image_tensor):
    """Returns the output to be used in the serving signature.

    Args:
      image_tensor: A tensor represeting input image. The image should
        have 3 channels.

    Returns:
      The model's predicton on input image tensor
    """
    return model(image_tensor)

  return serve_image_fn

def prepare_ground_truth(feature_dict):
  bboxes = feature_dict[_transformed_name(_BOX_KEY)]
  batch_size = tf.shape(bboxes)[0]
  bboxes = tf.sparse.to_dense(bboxes)
  bboxes = tf.reshape(bboxes, [batch_size,-1,4])

  bbox_labels = feature_dict[_transformed_name(_LABEL_KEY)]
  bbox_labels = tf.sparse.to_dense(bbox_labels)
  bbox_labels = tf.one_hot(bbox_labels, 20)

  feature_dict[_transformed_name(_BOX_KEY)] = bboxes
  feature_dict[_transformed_name(_LABEL_KEY)] = bbox_labels
  return feature_dict

def _input_fn(file_pattern: List[Text],
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 200) -> tf.data.Dataset:
  """Generates features and label for tuning/training.

  Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    tf_transform_output: A TFTransformOutput.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
  transformed_feature_spec = (
      tf_transform_output.transformed_feature_spec().copy())

  dataset = tf.data.experimental.make_batched_features_dataset(
      file_pattern=file_pattern,
      batch_size=1,
      features=transformed_feature_spec,
      reader=_gzip_reader_fn,
      )

  dataset = dataset.map(lambda x: prepare_ground_truth(x))

  return dataset

def _build_detection_model():
  print('Building model and restoring weights for fine-tuning...', flush=True)
  num_classes=20
  pipeline_config = '/usr/local/google/home/jingxiangl/ssd_mobilenet_v2_320x320_coco17_tpu-8/pipeline.config'
  checkpoint_path = '/usr/local/google/home/jingxiangl/ssd_mobilenet_v2_320x320_coco17_tpu-8/checkpoint/ckpt-0'

  # Load pipeline config and build a detection model.
  #
  # Since we are working off of a COCO architecture which predicts 90
  # class slots by default, we override the `num_classes` field here to be just
  # one (for our new rubber ducky class).
  configs = config_util.get_configs_from_pipeline_file(pipeline_config)
  model_config = configs['model']
  model_config.ssd.num_classes = num_classes
  model_config.ssd.freeze_batchnorm = True
  detection_model = model_builder.build(
        model_config=model_config, is_training=True)

  fake_model = tf.compat.v2.train.Checkpoint(
            _feature_extractor=detection_model._feature_extractor,
            # _box_predictor=fake_box_predictor
            )
  ckpt = tf.compat.v2.train.Checkpoint(model=fake_model)
  ckpt.restore(checkpoint_path).expect_partial()

  # Run model through a dummy image so that variables are created
  image, shapes = detection_model.preprocess(tf.zeros([1, 300, 300, 3]))
  prediction_dict = detection_model.predict(image, shapes)
  print(prediction_dict.keys())
  _ = detection_model.postprocess(prediction_dict, shapes)
  print('Weights restored!')
  return detection_model

def _decode_and_resize(image_str, target_size):
  """Decodes an encoded image byte string and resize to target size

  Args:
    image_str: The encoded image byte string
    target_size: The target image size

  Returns:
    An image tensor
  """
  image = tf.io.decode_png(image_str, channels=3)
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, target_size)
  return image

# TFX Transform will call this function.
def preprocessing_fn(inputs):
  """tf.transform's callback function for preprocessing inputs.

  Args:
    inputs: map from feature keys to raw not-yet-transformed features.

  Returns:
    Map from string feature key to transformed feature operations.
  """
  outputs = {}

  image_features = tf.map_fn(lambda x: _decode_and_resize(x[0], [300,300]), inputs[_IMAGE_KEY], dtype=tf.float32)
  boxes = inputs[_BOX_KEY]

  box_labels = inputs[_LABEL_KEY]
  image_features = tf.ensure_shape(image_features, (None, 300, 300, 3))
  # image_features = tf.map_fn(tf.keras.applications.mobilenet.preprocess_input,
                            #  image_features, dtype=tf.float32)

  outputs[_transformed_name(_IMAGE_KEY)] = (image_features)
  outputs[_transformed_name(_BOX_KEY)] = boxes
  # TODO(b/157064428): Support label transformation for Keras.
  # Do not apply label transformation as it will result in wrong evaluation.
  # outputs[transformed_name(LABEL_KEY)] = inputs[LABEL_KEY]
  outputs[_transformed_name(_LABEL_KEY)] = box_labels

  return outputs

# Set up forward + backward pass for a single train step.
def get_model_train_step_function(model, optimizer, vars_to_fine_tune):
  """Get a tf.function for training step."""

  # Use tf.function for a bit of speed.
  # Comment out the tf.function decorator if you want the inside of the
  # function to run eagerly.
  # @tf.function
  def train_step_fn(image_tensors,
                    groundtruth_boxes_list,
                    groundtruth_classes_list):
    """A single training iteration.

    Args:
      image_tensors: A list of [1, height, width, 3] Tensor of type tf.float32.
        Note that the height and width can vary across images, as they are
        reshaped within this function to be 640x640.
      groundtruth_boxes_list: A list of Tensors of shape [N_i, 4] with type
        tf.float32 representing groundtruth boxes for each image in the batch.
      groundtruth_classes_list: A list of Tensors of shape [N_i, num_classes]
        with type tf.float32 representing groundtruth boxes for each image in
        the batch.

    Returns:
      A scalar tensor representing the total loss for the input batch.
    """
    shapes = tf.constant(2 * [[300, 300, 3]], dtype=tf.int32)
    model.provide_groundtruth(
        groundtruth_boxes_list=groundtruth_boxes_list,
        groundtruth_classes_list=groundtruth_classes_list)
    print(image_tensors)
    with tf.GradientTape() as tape:
      preprocessed_images = model.preprocess(image_tensors)[0]
      prediction_dict = model.predict(preprocessed_images, shapes)
      losses_dict = model.loss(prediction_dict, shapes)
      total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']
      gradients = tape.gradient(total_loss, vars_to_fine_tune)
      optimizer.apply_gradients(zip(gradients, vars_to_fine_tune))
    return total_loss

  return train_step_fn

# TFX Trainer will call this function.
def run_fn(fn_args: TrainerFnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  train_dataset = _input_fn(fn_args.train_files, tf_transform_output,
                            _TRAIN_BATCH_SIZE)
  eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output,
                           _EVAL_BATCH_SIZE)

  model = _build_detection_model()

  # Select variables in top layers to fine-tune.
  trainable_variables = model.trainable_variables
  to_fine_tune = []
  prefixes_to_train = [
    'BoxPredictor/ConvolutionalClassHead',
    'BoxPredictor/ConvolutionalBoxHead']

  for var in trainable_variables:
    if any([var.name.startswith(prefix) for prefix in prefixes_to_train]):
      to_fine_tune.append(var)
  # mirrored_strategy = tf.distribute.MirroredStrategy()
  # with mirrored_strategy.scope():
  #   model = _build_keras_model()

  optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
  train_step_fn = get_model_train_step_function(
      model, optimizer, to_fine_tune)

  train_iter = iter(train_dataset)

  # 5 batches
  print('Start fine-tuning!', flush=True)
  for idx in range(5):

    # Note that we do not do data augmentation in this demo.  If you want a
    # a fun exercise, we recommend experimenting with random horizontal flipping
    # and random cropping :)
    examples = train_iter.get_next()
    print(examples)
    gt_boxes_list = list(examples[_transformed_name(_BOX_KEY)])
    gt_classes_list = list(examples[_transformed_name(_LABEL_KEY)])
    image_tensors = examples[_transformed_name(_IMAGE_KEY)]

    # Training step (forward pass + backwards pass)
    total_loss = train_step_fn(image_tensors, gt_boxes_list, gt_classes_list)

    if idx % 1 == 0:
      print('batch ' + str(idx) + ' of ' + str(5)
      + ', loss=' +  str(total_loss.numpy()), flush=True)

  keras_inputs = tf.keras.Input(shape=(300,300,3), name=_transformed_name(_IMAGE_KEY))
  keras_outputs = model(keras_inputs)
  keras_model = tf.keras.Model(keras_inputs, keras_outputs)
  print(keras_model.summary())

  signatures = {
      'serving_default':
          _get_serve_image_fn(
              keras_model).get_concrete_function(
                  tf.TensorSpec(shape=[None, 300, 300, 3], dtype=tf.float32, name=_transformed_name(_IMAGE_KEY)))
  }

  temp_saving_model_dir = os.path.join(fn_args.serving_model_dir, 'temp')
  keras_model.save(temp_saving_model_dir, save_format='tf', signatures=signatures)
  # tf.saved_model.save(keras_model, temp_saving_model_dir,signatures=signatures)

  tfrw = rewriter_factory.create_rewriter(
      rewriter_factory.TFLITE_REWRITER, name='tflite_rewriter',
      enable_experimental_new_converter=True)
  converters.rewrite_saved_model(temp_saving_model_dir,
                                 fn_args.serving_model_dir,
                                 tfrw,
                                 rewriter.ModelType.TFLITE_MODEL)

  tf.io.gfile.rmtree(temp_saving_model_dir)
