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
"""Python source which includes pipeline functions for the Penguins dataset.

The utilities in this file are used to build a model with native Keras or with
Flax.
"""

from typing import List, Optional, Text
import tensorflow as tf
import tensorflow_transform as tft

from tfx import v1 as tfx

from tfx_bsl.public import tfxio

FEATURE_KEYS = [
    'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g'
]
_LABEL_KEY = 'species'

TRAIN_BATCH_SIZE = 20
EVAL_BATCH_SIZE = 10


def transformed_name(key):
  return key + '_xf'


def make_serving_signatures(model,
                            tf_transform_features: tft.TFTransformOutput,
                            serving_batch_size: Optional[int] = None):
  """Returns the serving signatures.

  Args:
    model: the model function to apply to the transformed features.
    tf_transform_features: The transformation to apply to the serialized
      tf.Example.
    serving_batch_size: an optional specification for a concrete serving batch
      size.

  Returns:
    The signatures to use for saving the mode. The 'serving_default' signature
    will be a concrete function that takes a serialized tf.Example, parses it,
    transformes the features and then applies the model.
  """

  model.tft_layer = tf_transform_features.transform_features_layer()

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    """Returns the output to be used in the serving signature."""
    feature_spec = tf_transform_features.raw_feature_spec()
    feature_spec.pop(_LABEL_KEY)
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

    transformed_features = model.tft_layer(parsed_features)

    return model(transformed_features)

  return {
      'serving_default':
          serve_tf_examples_fn.get_concrete_function(
              tf.TensorSpec(
                  shape=[serving_batch_size], dtype=tf.string, name='examples'))
  }


def input_fn(file_pattern: List[Text],
             data_accessor: tfx.components.DataAccessor,
             tf_transform_output: tft.TFTransformOutput,
             batch_size: int) -> tf.data.Dataset:
  """Generates features and label for tuning/training.

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
          batch_size=batch_size, label_key=transformed_name(_LABEL_KEY)),
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

  for key in FEATURE_KEYS:
    # Nothing to transform for the penguin dataset. This code is just to
    # show how the preprocessing function for Transform should be defined.
    # We just assign original values to the transformed feature.
    outputs[transformed_name(key)] = inputs[key]
  # TODO(b/157064428): Support label transformation for Keras.
  # Do not apply label transformation as it will result in wrong evaluation.
  outputs[transformed_name(_LABEL_KEY)] = inputs[_LABEL_KEY]

  return outputs
