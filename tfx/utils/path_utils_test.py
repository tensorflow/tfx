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

  def setUp(self):
    # Create folders based on current Trainer output model directory.
    self._output_uri = os.path.join(self.get_temp_dir(), 'model_dir')
    self._eval_model_path = os.path.join(self._output_uri, 'eval_model_dir',
                                         'MODEL')
    tf.gfile.MakeDirs(self._eval_model_path)
    self._serving_model_path = os.path.join(
        self._output_uri, 'serving_model_dir', 'export', 'taxi', 'MODEL')
    tf.gfile.MakeDirs(self._serving_model_path)

  def test_model_path(self):
    # Test retrieving model folder.
    self.assertEqual(self._eval_model_path,
                     path_utils.eval_model_path(self._output_uri))
    self.assertEqual(self._serving_model_path,
                     path_utils.serving_model_path(self._output_uri))


if __name__ == '__main__':
  tf.test.main()
