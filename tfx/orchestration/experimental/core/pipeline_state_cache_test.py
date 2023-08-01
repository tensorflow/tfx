# Copyright 2023 Google LLC. All Rights Reserved.
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
"""Tests for tfx.orchestration.experimental.core.pipeline_state_cache."""

import tensorflow as tf
from tfx.orchestration.experimental.core import pipeline_state_cache
from tfx.orchestration.experimental.core import test_utils


class PipelineStateCacheTest(test_utils.TfxTest):

  def setUp(self):
    super().setUp()
    self._test_cache = pipeline_state_cache._LivePipelineCache()

  def test_get_check_signal(self):
    self.assertTrue(self._test_cache.get_check_signal())
    self._test_cache.update_check_signal(False)
    self.assertFalse(self._test_cache.get_check_signal())

  def test_update_check_signal(self):
    self.assertTrue(self._test_cache._live_pipeline_or_check)
    self._test_cache.update_check_signal(False)
    self.assertFalse(self._test_cache._live_pipeline_or_check)


if __name__ == '__main__':
  tf.test.main()
