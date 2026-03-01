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
"""TFX penguin preprocessing.

This file defines a template for TFX Transform component.
"""

import tensorflow_transform as tft

from tfx.experimental.templates.penguin.models import features


# TFX Transform will call this function.
# TODO(step 3): Define your transform logic in this function.
def preprocessing_fn(inputs):
  """tf.transform's callback function for preprocessing inputs.

  Args:
    inputs: map from feature keys to raw not-yet-transformed features.

  Returns:
    Map from string feature key to transformed feature operations.
  """
  outputs = {}

  # This function is the entry point for your feature engineering with
  # TensorFlow Transform, using the TFX Transform component.  In this example
  # the feature engineering is very simple, only applying z-score scaling.
  for key in features.FEATURE_KEYS:
    outputs[features.transformed_name(key)] = tft.scale_to_z_score(inputs[key])

  # Do not apply label transformation as it will result in wrong evaluation.
  outputs[features.transformed_name(
      features.LABEL_KEY)] = inputs[features.LABEL_KEY]

  return outputs
