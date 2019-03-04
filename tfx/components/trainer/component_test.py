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
"""Tests for tfx.components.trainer.component."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tfx.components.trainer import component
from tfx.proto import trainer_pb2
from tfx.utils import channel
from tfx.utils import types


class ComponentTest(tf.test.TestCase):

  def test_construct(self):
    transformed_examples = types.TfxType(type_name='ExamplesPath')
    transform_output = types.TfxType(type_name='TransformPath')
    schema = types.TfxType(type_name='SchemaPath')
    trainer = component.Trainer(
        module_file='/path/to/module/file',
        transformed_examples=channel.as_channel([transformed_examples]),
        transform_output=channel.as_channel([transform_output]),
        schema=channel.as_channel([schema]),
        train_args=trainer_pb2.TrainArgs(num_steps=100),
        eval_args=trainer_pb2.EvalArgs(num_steps=50))
    self.assertEqual('ModelExportPath', trainer.outputs.output.type_name)


if __name__ == '__main__':
  tf.test.main()
