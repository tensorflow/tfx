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
"""Tests for tfx.components.transform.executor with sequnce examples.

With the native TF2 code path being exercised.
"""
import os

import tensorflow as tf

from tfx.components.transform import executor_sequence_example_test


class ExecutorWithSequenceExampleV2Test(
    executor_sequence_example_test.ExecutorWithSequenceExampleTest):
  # Should not rely on inherited _SOURCE_DATA_DIR for integration tests to work
  # when TFX is installed as a non-editable package.
  _SOURCE_DATA_DIR = os.path.join(
      os.path.dirname(os.path.dirname(__file__)), 'testdata')

  def _use_force_tf_compat_v1(self):
    return False


if __name__ == '__main__':
  tf.test.main()
