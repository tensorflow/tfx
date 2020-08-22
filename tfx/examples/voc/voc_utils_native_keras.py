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
import time
from typing import Text, List, Dict, Iterable

import absl
import tensorflow as tf
import tensorflow_transform as tft

from object_detection.utils import label_map_util, config_util, object_detection_evaluation
from object_detection.builders import model_builder

from tfx.components.trainer.rewriting import converters
from tfx.components.trainer.rewriting import rewriter
from tfx.components.trainer.rewriting import rewriter_factory
from tfx.components.trainer.executor import TrainerFnArgs

import tensorflow_model_analysis as tfma
import apache_beam as beam

_TRAIN_DATA_SIZE = 256
_EVAL_DATA_SIZE = 100
_TRAIN_BATCH_SIZE = 32
_EVAL_BATCH_SIZE = 32

_IMAGE_KEY = 'image'
_OBJECT_BOX_KEY = 'objects/bbox'
_OBJECT_CLASS_KEY = 'objects/label'
_FILENAME_KEY = 'image/filename'

# TFLite model rewriter will change the keys in detection dict to
# Identity_{}. Please double check the remapping when using a model
# that's different from the one in this pipeline.
_TFLITE_DETECTION_BOX_KEY = 'Identity'
_TFLITE_DETECTION_CLASS_KEY = 'Identity_1'
_TFLITE_DETECTION_SCORE_KEY = 'Identity_2'
_TFLITE_NUM_DETECTION_KEY = 'Identity_3'

_PIPELINE_CONFIG_PATH = 'voc/model_checkpoints/ssd_mobilenet_v2_320x320_coco17_tpu-8/pipeline.config'
_CHECKPOINT_PATH = 'voc/model_checkpoints/ssd_mobilenet_v2_320x320_coco17_tpu-8/checkpoint/ckpt-0'
_LABEL_MAP_FILE_PATH = 'voc/data/pascal_label_map.pbtxt'

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
    detection_dict = model(image_tensor)
    num_boxes = detection_dict['num_detections']
    detection_boxes = detection_dict['detection_boxes']
    detection_classes = detection_dict['detection_classes']
    detection_scores = detection_dict['detection_scores']

    # For models that includes post-processing operations,
    # MediaPipe requires their output format to be
    # [bboxes, classes, scores, num_detections]
    return [detection_boxes, detection_classes, detection_scores, num_boxes]

  return serve_image_fn

def prepare_ground_truth(feature_dict):
  """Convert groundtruth into the format needed for training"""
  bboxes = feature_dict[_transformed_name(_OBJECT_BOX_KEY)]
  batch_size = tf.shape(bboxes)[0]
  bboxes = tf.sparse.to_dense(bboxes)
  bboxes = tf.reshape(bboxes, [batch_size, -1, 4])

  bbox_labels = feature_dict[_transformed_name(_OBJECT_CLASS_KEY)]
  bbox_labels = tf.sparse.to_dense(bbox_labels)
  bbox_labels = tf.one_hot(bbox_labels, 20)

  feature_dict[_transformed_name(_OBJECT_BOX_KEY)] = bboxes
  feature_dict[_transformed_name(_OBJECT_CLASS_KEY)] = bbox_labels
  return feature_dict

def _input_fn(file_pattern: List[Text],
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 1) -> tf.data.Dataset:
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
      batch_size=batch_size,
      features=transformed_feature_spec,
      reader_num_threads=16,
      parser_num_threads=32,
      reader=_gzip_reader_fn,
      )

  dataset = dataset.map(prepare_ground_truth, num_parallel_calls=16)
  return dataset

def _build_detection_model():
  """Use TF Object Detection API to build a detection model and restore wieghts."""

  # Since we are working off of a COCO architecture which predicts 90
  # class slots by default, we override the `num_classes` field here to be 20
  # (for VOC dataset).
  absl.logging.info('Building model and restoring weights for fine-tuning...')
  num_classes = 20
  configs = config_util.get_configs_from_pipeline_file(_PIPELINE_CONFIG_PATH)
  model_config = configs['model']
  model_config.ssd.num_classes = num_classes
  model_config.ssd.freeze_batchnorm = True
  detection_model = model_builder.build(
      model_config=model_config, is_training=True)

  # We only restore the weights for backbone and box prediction heads.
  # We need to train the class prediction head from scratch since we are finetuning
  # on a different dataset than COCO.
  fake_box_predictor = tf.compat.v2.train.Checkpoint(
      _prediction_heads={
          'box_encodings':
              detection_model._box_predictor._prediction_heads['box_encodings']}
      )

  fake_model = tf.compat.v2.train.Checkpoint(
      _feature_extractor=detection_model._feature_extractor,
      _box_predictor=fake_box_predictor
      )

  ckpt = tf.compat.v2.train.Checkpoint(model=fake_model)
  ckpt.restore(_CHECKPOINT_PATH).expect_partial()

  # Run model through a dummy image so that variables are created
  image, shapes = detection_model.preprocess(tf.zeros([1, 300, 300, 3]))
  prediction_dict = detection_model.predict(image, shapes)
  absl.logging.info(prediction_dict.keys())
  _ = detection_model.postprocess(prediction_dict, shapes)
  absl.logging.info('Weights restored!')
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

  image_features = tf.map_fn(lambda x: _decode_and_resize(x[0], [300, 300]),
                             inputs[_IMAGE_KEY], dtype=tf.float32)

  outputs[_transformed_name(_IMAGE_KEY)] = image_features
  # Note that the object boxes and object labels are Sparse Tensor because
  # different images have different number of objects annotations.
  outputs[_transformed_name(_OBJECT_BOX_KEY)] = inputs[_OBJECT_BOX_KEY]
  outputs[_transformed_name(_OBJECT_CLASS_KEY)] = inputs[_OBJECT_CLASS_KEY]
  # We will need filename as image identifier for the evaluator in custom TFMA metric
  outputs[_transformed_name(_FILENAME_KEY)] = inputs[_FILENAME_KEY]

  return outputs

class CalculateMAP(tfma.metrics.Metric):
  """Custom TFMA metric for caluclating detection model's mAP."""

  def __init__(self, name: Text = 'map'):
    super(CalculateMAP, self).__init__(_calculate_map, name=name)

def _calculate_map(
    name: Text = 'map') -> tfma.metrics.MetricComputations:
  key = tfma.metrics.MetricKey(name=name)
  return [
      tfma.metrics.MetricComputation(
          keys=[key],
          preprocessor=_MAPPreprocessor(),
          combiner=_MAPCombiner(key))
  ]

class _MAPPreprocessor(beam.DoFn):
  """Preprocessor for custom mAP TFMA metric"""
  def process(self, extracts: tfma.Extracts) -> Iterable[tfma.Extracts]:
    extracts_out = {}
    groundtruth = {}
    predictions = {}

    # The evaluator in TF Object Detection API needs to take 1-indexed class labels.
    # we need to shift the labels by 1.
    label_id_offset = 1
    groundtruth['groundtruth_classes'] = extracts['features'] \
                                          ['objects/label_xf'] + label_id_offset
    groundtruth['groundtruth_boxes'] = extracts['features'] \
                                          ['objects/bbox_xf'].reshape(-1, 4)

    predictions['detection_classes'] = extracts['predictions'] \
                                          [_TFLITE_DETECTION_CLASS_KEY] + \
                                          label_id_offset
    predictions['detection_boxes'] = extracts['predictions'] \
                                          [_TFLITE_DETECTION_BOX_KEY]
    predictions['detection_scores'] = extracts['predictions'] \
                                          [_TFLITE_DETECTION_SCORE_KEY]

    extracts_out['filename'] = extracts['features']['image/filename_xf'][0]
    extracts_out['groundtruth'] = groundtruth
    extracts_out['predictions'] = predictions
    yield extracts_out

class _MAPCombiner(beam.CombineFn):
  """Combiner for custom mAP TFMA metric"""
  def __init__(self, metric_key: tfma.metrics.MetricKey):
    self._metric_key = metric_key
    categories = label_map_util.create_categories_from_labelmap(
        _LABEL_MAP_FILE_PATH)
    self.evaluator = object_detection_evaluation.PascalDetectionEvaluator(
        categories)

  def create_accumulator(self) -> List:
    return []

  def add_input(self, accumulator: List, state: Dict) -> List:
    accumulator.append(state)
    return accumulator

  def merge_accumulators(self, accumulators: List) -> List:
    all_predictions = []
    for accumulator in accumulators:
      all_predictions = all_predictions + accumulator
    return all_predictions

  def extract_output(self,
                     accumulator: List) -> Dict[tfma.metrics.MetricKey, float]:
    for example in accumulator:
      # Note: The order of the following two function calss cannot be swapped.
      self.evaluator.add_single_ground_truth_image_info(example['filename'],
                                                        example['groundtruth'])
      self.evaluator.add_single_detected_image_info(example['filename'],
                                                    example['predictions'])
    out = self.evaluator.evaluate()
    print('The model\'s mAP on validation set is')
    print(out)
    return {self._metric_key: out['PascalBoxes_Precision/mAP@0.5IOU']}

# Set up forward + backward pass for a single train step.
def get_model_train_step_function(model, optimizer, vars_to_fine_tune):
  """Get a tf.function for training step."""

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
    shapes = tf.constant(_TRAIN_BATCH_SIZE*[[300, 300, 3]], dtype=tf.int32)
    model.provide_groundtruth(
        groundtruth_boxes_list=groundtruth_boxes_list,
        groundtruth_classes_list=groundtruth_classes_list)
    with tf.GradientTape() as tape:
      preprocessed_images = tf.concat(
          [model.preprocess(image_tensor)[0]
           for image_tensor in image_tensors], axis=0)
      prediction_dict = model.predict(preprocessed_images, shapes)
      losses_dict = model.loss(prediction_dict, shapes)
      total_loss = losses_dict['Loss/localization_loss'] + \
                   losses_dict['Loss/classification_loss']
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
  train_dataset = _input_fn(fn_args.train_files, tf_transform_output, 1)
  model = _build_detection_model()

  # Select variables corresponding to classification heads for fine-tuning.
  # Here we only train the weights for class prediction heads.
  trainable_variables = model.trainable_variables
  to_fine_tune = []
  prefixes_to_train = ['BoxPredictor/ConvolutionalClassHead']

  for var in trainable_variables:
    if any([var.name.startswith(prefix) for prefix in prefixes_to_train]):
      to_fine_tune.append(var)

  optimizer = tf.keras.optimizers.SGD(learning_rate=0.03, momentum=0.9)
  train_step_fn = get_model_train_step_function(
      model, optimizer, to_fine_tune)

  steps_per_epoch = int(_TRAIN_DATA_SIZE / _TRAIN_BATCH_SIZE)
  num_epochs = int(fn_args.train_steps / steps_per_epoch)

  absl.logging.info('Start fine-tuning!')
  for e_idx in range(num_epochs):
    train_iter = iter(train_dataset)
    epoch_start_time = time.time()
    for b_idx in range(steps_per_epoch):
      gt_boxes_list = []
      gt_classes_list = []
      image_tensors = []
      for _ in range(_TRAIN_BATCH_SIZE):
        example = train_iter.get_next()
        gt_boxes_list.append(example[_transformed_name(_OBJECT_BOX_KEY)][0])
        gt_classes_list.append(example[_transformed_name(_OBJECT_CLASS_KEY)][0])
        image_tensors.append(example[_transformed_name(_IMAGE_KEY)])

      # Training step (forward pass + backwards pass)
      total_loss = train_step_fn(image_tensors, gt_boxes_list, gt_classes_list)
      absl.logging.info(
          'epoch {} batch {}, loss={}'.format(e_idx, b_idx, total_loss.numpy()))

    epoch_end_time = time.time()
    absl.logging.info(
        'epoch {} takes {:.2f} minutes'.format(
            e_idx, (epoch_end_time-epoch_start_time)/60))

  # The model provided by TF Object Detection API is a Keras Layer object.
  # we wrap it into a Keras Model.
  keras_inputs = tf.keras.Input(
      shape=(None, None, 3),
      name=_transformed_name(_IMAGE_KEY))
  keras_outputs = model(keras_inputs)
  keras_model = tf.keras.Model(keras_inputs, keras_outputs)

  # Prepare the TFLite model used for serving in MediaPipe
  signatures = {
      'serving_default':
          _get_serve_image_fn(
              keras_model).get_concrete_function(
                  tf.TensorSpec(shape=[1, 300, 300, 3],
                                dtype=tf.float32,
                                name=_transformed_name(_IMAGE_KEY)))
  }

  temp_saving_model_dir = os.path.join(fn_args.serving_model_dir, 'temp')
  keras_model.save(temp_saving_model_dir,
                   save_format='tf',
                   signatures=signatures)

  tfrw = rewriter_factory.create_rewriter(
      rewriter_factory.TFLITE_REWRITER, name='tflite_rewriter',
      enable_experimental_new_converter=True)
  converters.rewrite_saved_model(temp_saving_model_dir,
                                 fn_args.serving_model_dir,
                                 tfrw,
                                 rewriter.ModelType.TFLITE_MODEL)

  tf.io.gfile.rmtree(temp_saving_model_dir)
