# Lint as: python2, python3
# Copyright 2019 Google LLC. All Rights Reserved.
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
"""Tests for tfx.utils.path_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# Standard Imports

import tensorflow as tf
from tfx.utils import path_utils


class PathUtilsTest(tf.test.TestCase):

  def testEstimatorModelPath(self):
    # Create folders based on Estimator based Trainer output model directory.
    output_uri = os.path.join(self.get_temp_dir(), 'model_dir')
    eval_model_path = os.path.join(output_uri, 'eval_model_dir', '123')
    tf.io.gfile.makedirs(eval_model_path)
    serving_model_path = os.path.join(output_uri, 'serving_model_dir', 'export',
                                      'taxi', '123')
    tf.io.gfile.makedirs(serving_model_path)
    # Test retrieving model folder.
    self.assertEqual(eval_model_path, path_utils.eval_model_path(output_uri))
    self.assertEqual(serving_model_path,
                     path_utils.serving_model_path(output_uri))

  def testKerasModelPath(self):
    # Create folders based on Keras based Trainer output model directory.
    output_uri = os.path.join(self.get_temp_dir(), 'model_dir')
    serving_model_path = os.path.join(output_uri, 'serving_model_dir')
    tf.io.gfile.makedirs(serving_model_path)
    # Test retrieving model folder.
    self.assertEqual(serving_model_path, path_utils.eval_model_path(output_uri))
    self.assertEqual(serving_model_path,
                     path_utils.serving_model_path(output_uri))


if __name__ == '__main__':
  tf.test.main()
