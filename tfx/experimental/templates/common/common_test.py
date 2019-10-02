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
"""Tests for pipeline common modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tfx.experimental.templates import common


class CommonTest(tf.test.TestCase):

  # Developer TODO: Implement adequate unit tests for common modules.

  def testFeatures(self):
    self.assertEqual(type(common.NUMERIC_FEATURES), list)
    self.assertEqual(type(common.CATEGORICAL_FEATURES), list)
    self.assertEqual(type(common.LABEL), str)

  def testHParams(self):
    self.assertIsInstance(common.HPARAMS, tf.contrib.training.HParams)


if __name__ == '__main__':
  tf.test.main()
