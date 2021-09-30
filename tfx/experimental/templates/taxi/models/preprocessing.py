# Copyright 2020 Google LLC. All Rights Reserved.
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
"""TFX taxi preprocessing.

This file defines a template for TFX Transform component.
"""

import tensorflow as tf
import tensorflow_transform as tft

from tfx.experimental.templates.taxi.models import features


def _make_dense(x):
  """Converts to a dense tensor.

  Args:
    x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
      in the second dimension.

  Returns:
    A rank 1 tensor.
  """
  if not isinstance(x, tf.sparse.SparseTensor):
    return x

  return tf.squeeze(
      tf.sparse.to_dense(
          tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1])),
      axis=1)


def preprocessing_fn(inputs):
  """tf.transform's callback function for preprocessing inputs.

  Args:
    inputs: map from feature keys to raw not-yet-transformed features.

  Returns:
    Map from string feature key to transformed feature operations.
  """
  outputs = {}
  for key in features.DENSE_FLOAT_FEATURE_KEYS:
    # Preserve this feature as a dense float, setting nan's to the mean.
    outputs[key] = tft.scale_to_z_score(
        _make_dense(inputs[key]))

  for key in features.VOCAB_FEATURE_KEYS:
    # Build a vocabulary for this feature.
    outputs[key] = tft.compute_and_apply_vocabulary(
        _make_dense(inputs[key]),
        top_k=features.VOCAB_SIZE,
        num_oov_buckets=features.OOV_SIZE)

  for key, num_buckets in zip(features.BUCKET_FEATURE_KEYS,
                              features.BUCKET_FEATURE_BUCKET_COUNT):
    outputs[key] = tft.bucketize(
        _make_dense(inputs[key]),
        num_buckets)

  for key in features.CATEGORICAL_FEATURE_KEYS:
    outputs[key] = _make_dense(inputs[key])

  # Was this passenger a big tipper?
  taxi_fare = _make_dense(inputs[features.FARE_KEY])
  tips = _make_dense(inputs[features.LABEL_KEY])
  outputs[features.LABEL_KEY] = tf.cast(
      tf.greater(tips, tf.multiply(taxi_fare, tf.constant(0.2))), tf.int64)

  return outputs
