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
"""Tests for tfx.components.pusher.component."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tfx.components.pusher import component
from tfx.components.pusher import executor
from tfx.proto import pusher_pb2
from tfx.utils import channel
from tfx.utils import types


class ComponentTest(tf.test.TestCase):

  class _MyCustomPusherExecutor(executor.Executor):
    """Mock class to test custom executor injection."""
    pass

  def setUp(self):
    self.model_export = channel.as_channel(
        [types.TfxArtifact(type_name='ModelExportPath')])
    self.model_blessing = channel.as_channel(
        [types.TfxArtifact(type_name='ModelBlessingPath')])

  def test_construct(self):
    pusher = component.Pusher(
        model_export=self.model_export,
        model_blessing=self.model_blessing,
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory='push_destination')))
    self.assertEqual('ModelPushPath', pusher.outputs['model_push'].type_name)

  def test_construct_no_destination(self):
    with self.assertRaises(ValueError):
      _ = component.Pusher(
          model_export=self.model_export,
          model_blessing=self.model_blessing,
      )

  def test_construct_no_destination_custom_executor(self):
    pusher = component.Pusher(
        model_export=self.model_export,
        model_blessing=self.model_blessing,
        executor_class=self._MyCustomPusherExecutor,
    )
    self.assertEqual('ModelPushPath', pusher.outputs['model_push'].type_name)


if __name__ == '__main__':
  tf.test.main()
