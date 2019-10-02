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
"""Implementation of Transform component."""

from __future__ import division
from __future__ import print_function

import tensorflow_transform as tft

from tfx.experimental.templates import common


def preprocessing_fn(inputs):
  """tf.transform's callback function for preprocessing inputs."""
  outputs = inputs.copy()

  # Developer TODO: Implement preprocessing_fn using TF and TF.Transform APIs.

  for key in common.CATEGORICAL_FEATURES:
    _ = tft.vocabulary(
        inputs[key],
        vocab_filename=common.vocabulary_name(key),
        name=common.vocabulary_name(key))

  outputs[common.transformed_name(common.LABEL)] = inputs[common.LABEL] > 0.0

  return outputs
