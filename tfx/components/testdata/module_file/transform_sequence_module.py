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
"""Python source file include taxi pipeline functions and necesasry utils.

For a TFX pipeline to successfully run, a preprocessing_fn and a
_build_estimator function needs to be provided.  This file contains both.

This file is equivalent to examples/chicago_taxi/trainer/model.py and
examples/chicago_taxi/preprocess.py.
"""

import tensorflow_transform as tft


def preprocessing_fn(inputs):
  """tf.transform's callback function for preprocessing inputs.

  Args:
    inputs: map from feature keys to raw not-yet-transformed features.

  Returns:
    Map from string feature key to transformed features.
  """
  return {
      'f_transformed': tft.scale_to_z_score(inputs['f']),
      'seq_s_transformed': tft.compute_and_apply_vocabulary(inputs['seq_s'])
  }


def stats_options_updater_fn(unused_stats_type, stats_options):
  """Callback function for setting pre and post-transform stats options.

  Args:
    unused_stats_type: a stats_options_util.StatsType object.
    stats_options: a tfdv.StatsOptions object.

  Returns:
    An updated tfdv.StatsOptions object.
  """
  return stats_options
