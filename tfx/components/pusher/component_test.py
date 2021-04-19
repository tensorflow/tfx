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

from typing import Text

import tensorflow as tf
from tfx.components.pusher import component
from tfx.components.pusher import executor
from tfx.dsl.components.base import executor_spec
from tfx.orchestration import data_types
from tfx.proto import pusher_pb2
from tfx.types import channel_utils
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs


class ComponentTest(tf.test.TestCase):

  class _MyCustomPusherExecutor(executor.Executor):
    """Mock class to test custom executor injection."""
    pass

  def setUp(self):
    super(ComponentTest, self).setUp()
    self._model = channel_utils.as_channel([standard_artifacts.Model()])
    self._model_blessing = channel_utils.as_channel(
        [standard_artifacts.ModelBlessing()])
    self._infra_blessing = channel_utils.as_channel(
        [standard_artifacts.InfraBlessing()])
    self._push_destination = pusher_pb2.PushDestination(
        filesystem=pusher_pb2.PushDestination.Filesystem(
            base_directory=self.get_temp_dir()))

  def testConstruct(self):
    pusher = component.Pusher(
        model=self._model,
        model_blessing=self._model_blessing,
        push_destination=self._push_destination)
    self.assertEqual(
        standard_artifacts.PushedModel.TYPE_NAME,
        pusher.outputs[standard_component_specs.PUSHED_MODEL_KEY].type_name)

  def testConstructWithParameter(self):
    push_dir = data_types.RuntimeParameter(name='push-dir', ptype=Text)
    pusher = component.Pusher(
        model=self._model,
        model_blessing=self._model_blessing,
        push_destination={'filesystem': {
            'base_directory': push_dir
        }})
    self.assertEqual(
        standard_artifacts.PushedModel.TYPE_NAME,
        pusher.outputs[standard_component_specs.PUSHED_MODEL_KEY].type_name)

  def testConstructNoDestination(self):
    with self.assertRaises(ValueError):
      _ = component.Pusher(
          model=self._model,
          model_blessing=self._model_blessing,
      )

  def testConstructNoDestinationCustomExecutor(self):
    pusher = component.Pusher(
        model=self._model,
        model_blessing=self._model_blessing,
        custom_executor_spec=executor_spec.ExecutorClassSpec(
            self._MyCustomPusherExecutor),
    )
    self.assertEqual(
        standard_artifacts.PushedModel.TYPE_NAME,
        pusher.outputs[standard_component_specs.PUSHED_MODEL_KEY].type_name)

  def testConstruct_InfraBlessingReplacesModel(self):
    pusher = component.Pusher(
        # model=self._model,  # No model.
        model_blessing=self._model_blessing,
        infra_blessing=self._infra_blessing,
        push_destination=self._push_destination)

    self.assertCountEqual(
        pusher.inputs.keys(),
        ['model_blessing', 'infra_blessing'])

  def testConstruct_NoModelAndNoInfraBlessing_Fails(self):
    with self.assertRaisesRegex(ValueError, (
        'Either one of model or infra_blessing channel should be given')):
      component.Pusher(
          # model=self._model,  # No model.
          model_blessing=self._model_blessing,
          # infra_blessing=self._infra_blessing,  # No infra_blessing.
          push_destination=self._push_destination)


if __name__ == '__main__':
  tf.test.main()
