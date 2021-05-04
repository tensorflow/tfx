# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Tests for tfx.extensions.google_cloud_ai_platform.pusher.component."""

import tensorflow as tf
from tfx.extensions.google_cloud_ai_platform.pusher import component
from tfx.types import channel_utils
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs


class PusherTest(tf.test.TestCase):

  def testConstruct(self):
    self._model = channel_utils.as_channel([standard_artifacts.Model()])
    self._model_blessing = channel_utils.as_channel(
        [standard_artifacts.ModelBlessing()])
    pusher = component.Pusher(
        model=self._model, model_blessing=self._model_blessing)
    self.assertEqual(
        standard_artifacts.PushedModel.TYPE_NAME,
        pusher.outputs[standard_component_specs.PUSHED_MODEL_KEY].type_name)

if __name__ == '__main__':
  tf.test.main()
