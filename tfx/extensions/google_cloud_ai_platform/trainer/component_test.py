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
"""Tests for tfx.extensions.google_cloud_ai_platform.trainer.component."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tfx.extensions.google_cloud_ai_platform.trainer import component
from tfx.proto import trainer_pb2
from tfx.types import channel_utils
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs


class ComponentTest(tf.test.TestCase):

  def testConstructFromModuleFile(self):
    examples = channel_utils.as_channel([standard_artifacts.Examples()])
    transform_graph = channel_utils.as_channel(
        [standard_artifacts.TransformGraph()])
    super(ComponentTest, self).setUp()
    schema = channel_utils.as_channel([standard_artifacts.Schema()])
    train_args = trainer_pb2.TrainArgs(splits=['train'], num_steps=100)
    eval_args = trainer_pb2.EvalArgs(splits=['eval'], num_steps=50)
    module_file = '/path/to/module/file'
    trainer = component.Trainer(
        module_file=module_file,
        transformed_examples=examples,
        transform_graph=transform_graph,
        schema=schema,
        train_args=train_args,
        eval_args=eval_args)

    self.assertEqual(
        standard_artifacts.Model.TYPE_NAME,
        trainer.outputs[standard_component_specs.MODEL_KEY].type_name)
    self.assertEqual(
        standard_artifacts.ModelRun.TYPE_NAME,
        trainer.outputs[standard_component_specs.MODEL_RUN_KEY].type_name)
    self.assertEqual(
        module_file,
        trainer.spec.exec_properties[standard_component_specs.MODULE_FILE_KEY])

if __name__ == '__main__':
  tf.test.main()
