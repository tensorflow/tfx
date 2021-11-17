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
"""Tests for tfx.components.tuner.component."""

import tensorflow as tf
from tfx.extensions.google_cloud_ai_platform.tuner import component
from tfx.proto import trainer_pb2
from tfx.proto import tuner_pb2
from tfx.types import channel_utils
from tfx.types import standard_artifacts


class TunerTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.examples = channel_utils.as_channel([standard_artifacts.Examples()])
    self.schema = channel_utils.as_channel([standard_artifacts.Schema()])
    self.transform_graph = channel_utils.as_channel(
        [standard_artifacts.TransformGraph()])
    self.train_args = trainer_pb2.TrainArgs(num_steps=100)
    self.eval_args = trainer_pb2.EvalArgs(num_steps=50)
    self.tune_args = tuner_pb2.TuneArgs(num_parallel_trials=3)
    self.custom_config = {'key': 'value'}

  def _verify_output(self, tuner):
    self.assertEqual(standard_artifacts.HyperParameters.TYPE_NAME,
                     tuner.outputs['best_hyperparameters'].type_name)

  def testConstructWithCustomConfig(self):
    tuner = component.Tuner(
        examples=self.examples,
        schema=self.schema,
        train_args=self.train_args,
        eval_args=self.eval_args,
        tune_args=self.tune_args,
        module_file='/path/to/module/file',
        custom_config=self.custom_config,
    )
    self._verify_output(tuner)

  def testConstructWithoutCustomConfig(self):
    tuner = component.Tuner(
        examples=self.examples,
        schema=self.schema,
        train_args=self.train_args,
        eval_args=self.eval_args,
        tune_args=self.tune_args,
        module_file='/path/to/module/file',
    )
    self._verify_output(tuner)


if __name__ == '__main__':
  tf.test.main()
